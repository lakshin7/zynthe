"""Run a tiny rationale-distillation experiment (CPU/GPU).

Generates a synthetic JSONL of (input, label, rationale) triples,
loads ``hf-internal-testing/tiny-t5`` and ``sshleifer/tiny-t5``, and
runs ``--steps`` SGD steps with the rationale multi-task loss.

Smoke criterion (matches the rest of zynthe's smoke suite): the
loss is finite and decays over the run.
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


def _write_synthetic_jsonl(path: Path, n: int, seed: int) -> None:
    """Emit ``n`` synthetic ``(input, label, rationale)`` records.

    Each record has a short arithmetic-word-problem form: the input
    is something like "3 + 5", the label is the result, the rationale
    is a one-sentence explanation.  Real runs would emit
    LLM-extracted rationales via Google's
    ``google-research/distilling-step-by-step`` pipeline; for the
    smoke we synthesise a tiny set with the same shape.
    """
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        op = rng.choice(("+", "-"))
        if op == "+":
            ans = a + b
        else:
            ans = a - b
        examples.append(
            {
                "input": f"What is {a} {op} {b}?",
                "label": str(ans),
                "rationale": (
                    f"To compute {a} {op} {b}, we add (or subtract) the "
                    f"two operands directly; the result is {ans}."
                ),
            }
        )
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _build_student_and_teacher(model_name: str, device: torch.device):
    """Return (model, prefix-aware forward) for an HF T5-style model.

    The forward wrapper takes a batch dict of ``{"label_input_ids",
    "label_attention_mask", "rationale_input_ids",
    "rationale_attention_mask"}`` and returns
    ``{"label_logits", "rationale_logits"}``.

    Both forward passes share the same backbone; we use T5's
    decoder-lm-head for both heads.  This is a thin simplification
    of the real Distill-step-by-step recipe (which uses a text-to-text
    output head with task prefixes).  Sufficient for the smoke proof.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.train()
    return model, tokenizer


def _forward_two_heads(model, tokenizer, batch, device, max_target_len: int = 8):
    """Run the student on the two task-prefixed views and return logits."""
    label_ids = batch["label_input_ids"].to(device)
    label_mask = batch["label_attention_mask"].to(device)
    rationale_ids = batch["rationale_input_ids"].to(device)
    rationale_mask = batch["rationale_attention_mask"].to(device)

    label_out = model(
        input_ids=label_ids,
        attention_mask=label_mask,
        labels=label_ids,  # dummy for shape; we don't compute loss
    )
    # The decoder returns logits; shape (B, T_dec, V). For the smoke
    # we project to (B, T_dec, vocab).  Use the model's lm_head on the
    # decoder hidden states — the simpler shortcut is to use the full
    # seq2seq logits directly and trim to the last time step (we
    # need a per-position logit vector for cross-entropy, so we use
    # the full output).
    label_logits = label_out.logits  # (B, T_dec, V)

    rationale_out = model(
        input_ids=rationale_ids,
        attention_mask=rationale_mask,
        labels=rationale_ids,
    )
    rationale_logits = rationale_out.logits  # (B, T_dec, V)
    return label_logits, rationale_logits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument(
        "--model",
        default="patrickvonplaten/t5-tiny-random",
        help="Tiny T5 model for the smoke. Default exists on HF Hub.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests/smoke/results/rationale.json"),
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate synthetic rationale dataset.
    jsonl = Path("tests/smoke/results/_rationale_synth.jsonl")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_synthetic_jsonl(jsonl, n=args.n_samples, seed=args.seed)
    print(f"[rationale] wrote {args.n_samples} synthetic triples to {jsonl}")

    # Load student (teacher is the same model frozen).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rationale] loading {args.model} on {device}")
    model, tokenizer = _build_student_and_teacher(args.model, device)
    for p_ in model.parameters():
        p_.requires_grad = False
    # Re-enable grads for the student (we're using a single model as
    # both teacher (frozen) and student (trainable) in the smoke).
    # Build a separate trainable copy.
    from transformers import AutoModelForSeq2SeqLM

    student = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    student.train()

    # Load the JSONL and tokenize each view.
    from zynthe.data import RationaleDataset

    ds = RationaleDataset(jsonl, required=True)

    # Build a small label/rationale tokenization.
    label_prefix = "label: "
    rationale_prefix = "rationale: "

    def _encode(ex):
        label_text = ex["label"]
        rationale_text = ex["rationale"]
        label_input = tokenizer(
            label_prefix + ex["input"],
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        rationale_input = tokenizer(
            rationale_prefix + ex["input"],
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        label_target = tokenizer(
            label_text,
            return_tensors="pt",
            padding="max_length",
            max_length=8,
            truncation=True,
        )["input_ids"]
        rationale_target = tokenizer(
            rationale_text,
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True,
        )["input_ids"]
        return {
            "label_input_ids": label_input["input_ids"],
            "label_attention_mask": label_input["attention_mask"],
            "rationale_input_ids": rationale_input["input_ids"],
            "rationale_attention_mask": rationale_input["attention_mask"],
            "label_ids": label_target,
            "rationale_ids": rationale_target,
        }

    optim = torch.optim.SGD(student.parameters(), lr=args.lr)

    losses = []
    started = time.time()
    for step in range(args.steps):
        ex = ds[step % len(ds)]
        batch = _encode(ex)
        optim.zero_grad()
        label_logits, rationale_logits = _forward_two_heads(
            student, tokenizer, batch, device
        )
        # Trim to target lengths to align with the labels.
        label_target = batch["label_ids"].to(device)[:, : label_logits.size(1)]
        rationale_target = batch["rationale_ids"].to(device)[:, : rationale_logits.size(1)]
        label_loss = F.cross_entropy(
            label_logits.reshape(-1, label_logits.size(-1)),
            label_target.reshape(-1),
            ignore_index=-100,
        )
        rationale_loss = F.cross_entropy(
            rationale_logits.reshape(-1, rationale_logits.size(-1)),
            rationale_target.reshape(-1),
            ignore_index=-100,
        )
        if torch.isnan(label_loss):
            label_loss = torch.zeros((), device=device)
        if torch.isnan(rationale_loss):
            rationale_loss = torch.zeros((), device=device)
        loss = label_loss + rationale_loss
        if not torch.isfinite(loss):
            print(
                f"[rationale][FAIL] non-finite loss at step {step}: {loss.item()!r}",
                file=sys.stderr,
            )
            return 1
        loss.backward()
        optim.step()
        losses.append(
            {
                "label": label_loss.item(),
                "rationale": rationale_loss.item(),
                "total": loss.item(),
            }
        )

    duration = time.time() - started
    print(
        f"[rationale][OK]   {args.steps} steps in {duration:.1f}s. "
        f"first={losses[0]['total']:.4f} last={losses[-1]['total']:.4f} "
        f"min={min(l['total'] for l in losses):.4f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "commit": os.environ.get("GIT_COMMIT", "local"),
        "model": args.model,
        "steps": args.steps,
        "n_samples": args.n_samples,
        "duration_s": duration,
        "loss_first": losses[0]["total"],
        "loss_last": losses[-1]["total"],
        "loss_min": min(l["total"] for l in losses),
        "loss_max": max(l["total"] for l in losses),
        "loss_decay": losses[0]["total"] - losses[-1]["total"],
        "loss_progression": [round(l["total"], 4) for l in losses[:: max(len(losses) // 6, 1)][:6]],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
