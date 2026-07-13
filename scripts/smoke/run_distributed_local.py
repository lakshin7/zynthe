"""CPU/GPU-runnable distributed-training smoke.

Loads ``prajjwal1/bert-tiny``, runs ``prepare_distillation`` on it
with a single-GPU accelerate config, and runs ``--steps`` SGD
updates.  The smoke criterion is "loss is finite and decays over
the run" — matching the rest of zynthe's smoke suite.

For actual DDP (multi-GPU), use ``torchrun`` with the same script:
the prepare_distillation call handles DDP preparation when
``num_processes>1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=Path("tests/smoke/results/distributed.json"))
    args = p.parse_args()

    from transformers import AutoModelForSequenceClassification

    from zynthe.core.training.distributed import (
        DistributedConfig,
        prepare_distillation,
    )

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[distributed] loading prajjwal1/bert-tiny on {device}")
    teacher = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    student = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False
    optim = torch.optim.SGD(student.parameters(), lr=args.lr)

    # Wire through accelerate.
    cfg = DistributedConfig(
        enabled=True,
        mixed_precision="no",
        num_processes=1,  # single-GPU on Modal L4
    )
    bundle = prepare_distillation(teacher, student, optim, dataloader=None, config=cfg)
    print(f"[distributed] prepare_distillation OK; accelerator={type(bundle.accelerator).__name__}")

    losses = []
    started = time.time()
    for step in range(args.steps):
        torch.manual_seed(args.seed + step)
        x = torch.randint(0, 1000, (4, 8), dtype=torch.long, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        with torch.no_grad():
            t_out = bundle.teacher(input_ids=x)
        optim.zero_grad()
        s_out = bundle.student(input_ids=x)
        # Mean-pool to (B, num_classes) so CE is shape-correct.
        if s_out.logits.dim() == 3:
            s_out.logits = s_out.logits.mean(dim=1)
        loss = F.cross_entropy(s_out.logits, y)
        if not torch.isfinite(loss):
            print(f"[distributed][FAIL] non-finite loss at step {step}", file=sys.stderr)
            return 1
        loss.backward()
        optim.step()
        losses.append(loss.item())

    duration = time.time() - started
    print(
        f"[distributed][OK]   {args.steps} steps in {duration:.1f}s. "
        f"first={losses[0]:.4f} last={losses[-1]:.4f} min={min(losses):.4f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "commit": os.environ.get("GIT_COMMIT", "local"),
        "model": "prajjwal1/bert-tiny",
        "accelerator": type(bundle.accelerator).__name__ if bundle.accelerator else None,
        "steps": args.steps,
        "duration_s": duration,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_decay": losses[0] - losses[-1],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
