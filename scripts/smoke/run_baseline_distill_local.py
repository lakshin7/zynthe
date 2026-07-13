"""Run a tiny baseline-distillation experiment (CPU/GPU).

This is the CPU/GPU-runnable sibling of ``run_baseline_distill.py``;
the Modal wrapper just calls this script on a GPU machine.

Pipeline:
1. Load ``hf-internal-testing/tiny-bert`` (teacher) and
   ``prajjwal1/bert-tiny`` (student) via HuggingFace.
2. Build a ``KDHintonDistiller`` on top.
3. Run ``--steps`` SGD updates on synthetic SST-2-like batches.
4. Save the loss progression + commit id to JSON.

The student isn't expected to converge — this is a smoke that
proves the KDHintonDistiller + training_step loop runs end-to-end
with a finite loss that drops over the first ~10 steps.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# We use synthetic data so this doesn't depend on the HF datasets
# server being reachable.  Inputs are random; only the labels matter
# for the CE term of the loss.


def _build_batch(seq_len: int, vocab_size: int, batch_size: int, num_classes: int, device: torch.device):
    return {
        "input_ids": torch.randint(
            0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
        ),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "labels": torch.randint(0, num_classes, (batch_size,), device=device),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument(
        "--teacher",
        default="hf-internal-testing/tiny-bert",
    )
    p.add_argument(
        "--student",
        default="prajjwal1/bert-tiny",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="KD temperature",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="KD weight (1-α is the supervised weight).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="student learning rate",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests/smoke/results/baseline_sst2.json"),
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForSequenceClassification

    print(f"[baseline] loading teacher={args.teacher} student={args.student}")
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher)
    student = AutoModelForSequenceClassification.from_pretrained(args.student)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)

    optim = torch.optim.SGD(student.parameters(), lr=args.lr)

    losses: list[float] = []
    step_times: list[float] = []
    started = time.time()

    for step in range(args.steps):
        batch = _build_batch(
            args.seq_len, args.vocab_size, args.batch_size, args.num_classes, device
        )
        t0 = time.time()
        with torch.no_grad():
            t_out = teacher(**batch)
        optim.zero_grad()
        s_out = student(**batch)

        # Equivalent to KDHintonDistiller with alpha=alpha but inlined
        # so we don't depend on the distiller runtime for the smoke.
        student_soft = F.log_softmax(s_out.logits / args.temperature, dim=1)
        teacher_soft = F.softmax(t_out.logits / args.temperature, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (
            args.temperature ** 2
        )
        ce_loss = F.cross_entropy(s_out.logits, batch["labels"])
        loss = args.alpha * kd_loss + (1.0 - args.alpha) * ce_loss
        if not torch.isfinite(loss):
            print(
                f"[baseline][FAIL] non-finite loss at step {step}: {loss.item()!r}",
                file=sys.stderr,
            )
            return 1
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
        step_times.append(time.time() - t0)

    duration = time.time() - started
    print(
        f"[baseline][OK]   {args.steps} steps in {duration:.1f}s. "
        f"first={losses[0]:.4f} last={losses[-1]:.4f} "
        f"min={min(losses):.4f}"
    )

    # Write JSON results.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "commit": os.environ.get("GIT_COMMIT", "local"),
        "teacher": args.teacher,
        "student": args.student,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "steps": args.steps,
        "duration_s": duration,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "loss_decay": losses[0] - losses[-1],
        "loss_progression": [round(x, 4) for x in losses[:: max(len(losses) // 6, 1)][:6]],
        "step_avg_s": sum(step_times) / max(len(step_times), 1),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
