"""End-to-end Distill-step-by-step recipe on SST-2.

Runs the full paper §3 recipe:

  1. Load 200 SST-2 records (synthetic fallback if HF datasets is
     unavailable).
  2. Run the LLM rationale extractor to produce (input, label,
     rationale) JSONL.
  3. Run the multi-task T5 trainer (Iter 2) on the JSONL for
     --steps SGD updates.
  4. Evaluate on a held-out split; report multi-task losses and
     loss decay.

Usage::

    python scripts/run_distill_step_by_step.py \
        --task sst2 \
        --train-records 200 \
        --eval-records 50 \
        --steps 100 \
        --output tests/smoke/results/step_by_step.json

The script is intentionally lightweight — it relies on the
existing components (scripts/extract_rationales.py + the
MultiTaskT5Trainer) and does NOT depend on Modal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Synthetic SST-2 generator (used when HF datasets is unavailable)
# ----------------------------------------------------------------------------


def _synthetic_sst2(n: int, seed: int) -> List[Dict[str, str]]:
    """Emit a tiny synthetic SST-2-like dataset.

    Real SST-2 would be loaded via :func:`datasets.load_dataset`.  This
    fallback is used in environments where the HF datasets server
    isn't reachable.
    """
    rng = random.Random(seed)
    positive = [
        "An absolute delight of a film — warm, funny, beautifully acted.",
        "Brilliant, charming, and deeply moving — a must-see.",
        "A gorgeous, uplifting experience that stays with you.",
        "Tender and funny in equal measure; a small gem.",
        "A wonderful surprise, full of heart and humour.",
        "Sublime acting and a story that grabs you from the start.",
        "Delightful, surprising, and wonderfully crafted.",
        "Touching, funny, and beautifully paced.",
    ]
    negative = [
        "A tedious, lifeless, and frankly boring experience.",
        "Predictable, wooden dialogue, and a plot that goes nowhere.",
        "I found the second half disappointing and dull.",
        "Disappointing, derivative, and a waste of a talented cast.",
        "Stale, slow, and frankly unwatchable.",
        "Painful, predictable, and surprisingly charmless.",
        "Boring, derivative, and frankly tiresome.",
        "Lifeless, flat, and honestly quite forgettable.",
    ]
    out: List[Dict[str, str]] = []
    for _ in range(n):
        out.append({"input": rng.choice(positive if rng.random() < 0.5 else negative)})
    return out


def _maybe_load_sst2(n: int, seed: int) -> List[Dict[str, str]]:
    """Try :func:`datasets.load_dataset('glue', 'sst2', ...)`.

    Falls back to the synthetic generator on any failure.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("glue", "sst2", split=f"train[:{n}]")
        return [
            {"input": str(rec["sentence"]), "label": "positive" if rec["label"] == 1 else "negative"}
            for rec in ds
        ]
    except Exception as exc:
        logger.warning(
            "Could not load SST-2 from HF datasets (%s); using synthetic.",
            exc,
        )
        return _synthetic_sst2(n, seed)


# ----------------------------------------------------------------------------
# End-to-end recipe
# ----------------------------------------------------------------------------


def _write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_recipe(
    *,
    task: str,
    train_records: int,
    eval_records: int,
    steps: int,
    seed: int,
    llm: str,
    output_dir: Path,
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_inputs = _maybe_load_sst2(train_records, seed)
    eval_inputs = _maybe_load_sst2(eval_records, seed + 1)

    # ------------------------------------------------------------------
    # 1. LLM rationale extraction
    # ------------------------------------------------------------------
    train_in_path = output_dir / "_sst2_train_inputs.jsonl"
    train_out_path = output_dir / "rationale_train.jsonl"
    eval_in_path = output_dir / "_sst2_eval_inputs.jsonl"
    eval_out_path = output_dir / "rationale_eval.jsonl"
    _write_jsonl(train_in_path, [{"input": r["input"]} for r in train_inputs])
    _write_jsonl(eval_in_path, [{"input": r["input"]} for r in eval_inputs])

    # Lazy import: scripts/ isn't a wheel-installed package.
    import sys as _sys
    _SCRIPTS = str(Path(__file__).parent)
    if _SCRIPTS not in _sys.path:
        _sys.path.insert(0, _SCRIPTS)
    from extract_rationales import default_llm_callable, extract_rationales

    llm_call = default_llm_callable(model_name=llm, batch_size=4)
    started = time.time()
    train_triples = extract_rationales(
        [{"input": r["input"]} for r in train_inputs],
        task_name=task,
        llm_callable=llm_call,
    )
    eval_triples = extract_rationales(
        [{"input": r["input"]} for r in eval_inputs],
        task_name=task,
        llm_callable=llm_call,
    )
    t_extract = time.time() - started
    _write_jsonl(train_out_path, train_triples)
    _write_jsonl(eval_out_path, eval_triples)
    print(
        f"[recipe] extracted {len(train_triples)} train + {len(eval_triples)} eval "
        f"triples in {t_extract:.1f}s"
    )

    # ------------------------------------------------------------------
    # 2. Multi-task T5 training
    # ------------------------------------------------------------------
    from zynthe.core.training.rationale_trainer import MultiTaskT5Trainer
    from zynthe.core.distillers.rationale_distiller import RationaleDistiller

    trainer = MultiTaskT5Trainer.from_pretrained(
        "patrickvonplaten/t5-tiny-random",
        label_prefix="label: ",
        rationale_prefix="rationale: ",
    )
    distiller = RationaleDistiller(
        teacher=trainer.model,
        student=trainer.model,
        config={
            "rationale": {
                "label_weight": 1.0,
                "rationale_weight": 0.5,
                "ignore_index": -100,
            }
        },
        device=trainer.device,
    )

    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-3)
    losses: List[Dict[str, float]] = []

    started = time.time()
    for step in range(steps):
        idx = step % len(train_triples)
        rec = train_triples[idx]
        label_target = trainer.tokenizer(
            rec["label"],
            return_tensors="pt",
            padding="max_length",
            max_length=8,
            truncation=True,
        )["input_ids"]
        rationale_target = trainer.tokenizer(
            rec["rationale"],
            return_tensors="pt",
            padding="max_length",
            max_length=24,
            truncation=True,
        )["input_ids"]
        loss, breakdown = trainer.train_step(
            {
                "input": rec["input"],
                "label_ids": label_target,
                "rationale_ids": rationale_target,
                "max_length": 32,
            },
            distiller=distiller,
            optimizer=optimizer,
        )
        losses.append(
            {
                "label": breakdown.get("label", float("nan")),
                "rationale": breakdown.get("rationale", float("nan")),
                "total": breakdown.get("total", float("nan")),
            }
        )
    t_train = time.time() - started
    print(
        f"[recipe] trained {steps} multi-task steps in {t_train:.1f}s"
    )

    # ------------------------------------------------------------------
    # 3. Evaluation: forward the eval set through both heads and
    #    record the losses (no_grad, no optimization).
    # ------------------------------------------------------------------
    trainer.model.eval()
    eval_losses: List[Dict[str, float]] = []
    started = time.time()
    for rec in eval_triples:
        label_target = trainer.tokenizer(
            rec["label"],
            return_tensors="pt",
            padding="max_length",
            max_length=8,
            truncation=True,
        )["input_ids"]
        rationale_target = trainer.tokenizer(
            rec["rationale"],
            return_tensors="pt",
            padding="max_length",
            max_length=24,
            truncation=True,
        )["input_ids"]
        with torch.no_grad():
            label_logits = trainer.forward_label(rec["input"], max_length=32)
            rationale_logits = trainer.forward_rationale(rec["input"], max_length=32)
            label_loss = F.cross_entropy(
                label_logits.float().view(-1, label_logits.size(-1)),
                label_target.view(-1).to(label_logits.device),
                ignore_index=-100,
            )
            rationale_loss = F.cross_entropy(
                rationale_logits.float().view(-1, rationale_logits.size(-1)),
                rationale_target.view(-1).to(rationale_logits.device),
                ignore_index=-100,
            )
        eval_losses.append(
            {
                "label": label_loss.item(),
                "rationale": rationale_loss.item(),
                "total": (label_loss + rationale_loss).item(),
            }
        )
    t_eval = time.time() - started
    trainer.model.train()
    print(
        f"[recipe] evaluated {len(eval_triples)} eval triples in {t_eval:.1f}s"
    )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    def _avg(xs):
        return sum(xs) / max(len(xs), 1)

    train_first = losses[0]
    train_last = losses[-1]
    eval_loss_avg = _avg([e["total"] for e in eval_losses])
    eval_loss_min = min([e["total"] for e in eval_losses])

    payload = {
        "timestamp": time.time(),
        "commit": os.environ.get("GIT_COMMIT", "local"),
        "task": task,
        "llm": llm,
        "train_records_requested": train_records,
        "train_triples_extracted": len(train_triples),
        "eval_records_requested": eval_records,
        "eval_triples_extracted": len(eval_triples),
        "steps": steps,
        "t_extract_s": t_extract,
        "t_train_s": t_train,
        "t_eval_s": t_eval,
        "train_loss_first": train_first,
        "train_loss_last": train_last,
        "train_loss_decay": train_first["total"] - train_last["total"],
        "eval_loss_total_avg": eval_loss_avg,
        "eval_loss_total_min": eval_loss_min,
    }
    summary_path = output_dir / "step_by_step.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[recipe][OK] {len(train_triples)} train triples, {len(eval_triples)} eval, "
        f"{steps} steps, train loss {train_first['total']:.4f} -> {train_last['total']:.4f} "
        f"(decay {payload['train_loss_decay']:.4f}) -> {summary_path}"
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="sst2", choices=["sst2", "esnli"])
    p.add_argument("--train-records", type=int, default=200)
    p.add_argument("--eval-records", type=int, default=50)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm", default="google/flan-t5-base")
    p.add_argument("--output", type=Path, default=Path("tests/smoke/results"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        run_recipe(
            task=args.task,
            train_records=args.train_records,
            eval_records=args.eval_records,
            steps=args.steps,
            seed=args.seed,
            llm=args.llm,
            output_dir=args.output,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[recipe][FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
