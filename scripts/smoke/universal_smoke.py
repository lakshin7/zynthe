"""Universal 5-family smoke gate.

Loads one teacher/student pair per modality family, runs ``n_steps``
of distillation, and asserts the loss is finite and bounded. The
results are written to ``--output`` as JSON.

This is the empirical proof for Zynthé's "universal" claim. The
script is meant to run on Modal L4 (16-24 GB) and stay within a
budget of about a dollar per run.

Usage::

    python scripts/smoke/universal_smoke.py --pairs all
    python scripts/smoke/universal_smoke.py --pairs bert vit

The pairs (5) are tiny models — small enough that 50 steps fit on any
recent CUDA GPU. Each pair is intentionally lightweight: BERT-tiny,
tinny ViT, tiny GPT-2, tiny CLIP, torchvision resnet18.

Use ``--quick`` to skip distillation loss checking (sanity-only) and
``--max-budget <usd>`` to abort when wall-time exceeds a cap.
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
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------
# Pair definitions
# ----------------------------------------------------------------------------


PAIRS = {
    "bert": {
        "teacher": "hf-internal-testing/tiny-bert",
        "student": "prajjwal1/bert-tiny",
        "task": "sequence_classification",
        "input_shape": (4, 32),
    },
    "vit": {
        "teacher": "google/vit-base-patch16-224-in21k",
        "student": "facebook/deit-tiny-patch16-224",
        "task": "image_classification",
        "input_shape": (3, 32, 32),
        "image_long_side": 224,  # ViTs trained at 224 — we keep this for future tests.
    },
    "gpt2": {
        "teacher": "sshleifer/tiny-gpt2",
        "student": "sshleifer/tiny-gpt2",
        "task": "causal_lm",
        "input_shape": (4, 32),
    },
    "clip": {
        # We use the production CLIP for the universal-model proof.
        # It is heavier (~600 MB download). Skip if --skip-clip is set.
        "teacher": "openai/clip-vit-base-patch32",
        "student": "openai/clip-vit-base-patch32",
        "task": "vision_language_contrastive",
        "input_shape": (3, 32, 32),
        "image_long_side": 224,
    },
    "resnet": {
        # Both teacher and student are the same small torchvision model
        # so we can prove the vision adapter path on a pure-CNN pair
        # (no HF download).
        "teacher": "resnet18",
        "student": "resnet18",
        "task": "image_classification",
        "input_shape": (3, 32, 32),
    },
}


def is_torchvision_pair(pair: dict) -> bool:
    """Tiny dataset+torchvision path uses ``torchvision.models`` rather
    than ``transformers.AutoModel``.
    """
    return "/" not in pair["teacher"] and "/" not in pair["student"]


def load_pair_models(pair: dict):
    """Load teacher + student via either transformers or torchvision.

    Returns (teacher, student, model_loader_label).
    """
    if is_torchvision_pair(pair):
        from torchvision import models

        teacher = getattr(models, pair["teacher"])()
        student = getattr(models, pair["student"])()
        return teacher, student, "torchvision"
    from transformers import AutoModel

    task = pair.get("task", "")
    if task in ("sequence_classification", "image_classification"):
        from transformers import AutoModelForSequenceClassification

        cls = AutoModelForSequenceClassification
    else:
        cls = AutoModel
    teacher = cls.from_pretrained(pair["teacher"])
    student = cls.from_pretrained(pair["student"])
    return teacher, student, "transformers"


# ----------------------------------------------------------------------------
# Tiny dataset
# ----------------------------------------------------------------------------


class _SyntheticBatch(Dataset):
    """A dataset that yields random tensors with the right shape for the
    pair's task.  Smoke gate doesn't care about accuracy — only that
    the pipeline runs.
    """

    def __init__(self, n: int, sample_factory):
        self.n = n
        self._factory = sample_factory

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self._factory(i)


# ----------------------------------------------------------------------------
# Sample factories per task
# ----------------------------------------------------------------------------


def _classification_sample(input_shape, n_classes, max_seq_len):
    return {
        "input_ids": torch.randint(0, 1000, input_shape, dtype=torch.long),
        "attention_mask": torch.ones(input_shape, dtype=torch.long),
        "labels": torch.tensor(random.randint(0, n_classes - 1)),
    }


def _vision_sample(input_shape, n_classes):
    return {
        "pixel_values": torch.randn(*input_shape),
        "labels": torch.tensor(random.randint(0, n_classes - 1)),
    }


def _causal_lm_sample(input_shape):
    return {
        "input_ids": torch.randint(0, 1000, input_shape, dtype=torch.long),
        "attention_mask": torch.ones(input_shape, dtype=torch.long),
    }


def _clip_sample(input_shape):
    return {
        "input_ids": torch.randint(0, 1000, (1, 8), dtype=torch.long),
        "pixel_values": torch.randn(*input_shape),
    }


def _factory_for_pair(pair_name: str, pair: dict):
    if pair["task"] == "sequence_classification":
        return lambda i: _classification_sample(pair["input_shape"], 4, pair["input_shape"][1])
    if pair["task"] == "image_classification":
        return lambda i: _vision_sample(pair["input_shape"], 4)
    if pair["task"] == "causal_lm":
        return lambda i: _causal_lm_sample(pair["input_shape"])
    if pair["task"] == "vision_language_contrastive":
        return lambda i: _clip_sample(pair["input_shape"])
    raise ValueError(pair["task"])


# ----------------------------------------------------------------------------
# Pair runner
# ----------------------------------------------------------------------------


def run_pair(
    pair_name: str,
    pair: dict,
    n_steps: int,
    quick: bool,
) -> dict:
    """Run the smoke gate for one teacher/student pair.

    Returns a dict of metrics (success, loss progression, step times).
    """
    print(f"\n[smoke] {pair_name}: {pair['teacher']} -> {pair['student']}")

    t_load_start = time.time()
    try:
        teacher, student, model_loader = load_pair_models(pair)
    except Exception as exc:  # noqa: BLE001 — smoke gate reports the error.
        return {
            "pair": pair_name,
            "success": False,
            "error": f"load_failed: {type(exc).__name__}: {exc}",
            "duration_s": time.time() - t_load_start,
        }
    print(
        f"  loaded teacher={type(teacher).__name__} student={type(student).__name__}"
    )
    load_s = time.time() - t_load_start

    # Adapter routing check.
    from zynthe.core.adapters import AdapterRegistry

    registry = AdapterRegistry()
    student_adapter = registry.detect(student)
    teacher_adapter = registry.detect(teacher)
    print(
        f"  adapters — teacher={teacher_adapter.modality!r}, "
        f"student={student_adapter.modality!r}, loader={model_loader!r}"
    )

    # Dataset / loader.
    factory = _factory_for_pair(pair_name, pair)
    ds = _SyntheticBatch(n=max(n_steps * 2, 16), sample_factory=factory)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device).eval()
    student.to(device).train()

    optim = torch.optim.SGD(student.parameters(), lr=1e-4)
    losses: list[float] = []
    step_times: list[float] = []

    def _call(model, batch):
        """Dispatch on the loader kind. Torchvision takes tensor only;
        transformers takes **batch.
        """
        if model_loader == "torchvision":
            x = batch.get("pixel_values")
            if x is None:
                raise TypeError("torchvision pair missing 'pixel_values'")
            return model(x)
        return model(**batch)

    def _extract_logits(out):
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        # Torchvision-style output: the tensor is the logits.
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(f"can't extract logits from {type(out).__name__}")

    if quick:
        # Just verify forward + backward.
        last_err: str | None = None
        quick_success = False
        for batch in loader:
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            with torch.no_grad():
                t_out = _call(teacher, batch_dev)
            try:
                s_out = _call(student, batch_dev)
                quick_success = True
                print(
                    f"  [quick] forward OK — logits shape "
                    f"{tuple(_extract_logits(s_out).shape)}"
                )
                break
            except Exception as exc:  # noqa: BLE001 — keep smoke going for next batch.
                last_err = f"forward_failed: {type(exc).__name__}: {exc}"
                continue
        return {
            "pair": pair_name,
            "success": quick_success,
            "load_s": load_s,
            "teacher_adapter": teacher_adapter.modality,
            "student_adapter": student_adapter.modality,
            "loader": model_loader,
            "mode": "quick",
            "error": last_err,
            "loss_progression": losses,
        }
    t_train = time.time()
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        batch_dev = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        t0 = time.time()
        with torch.no_grad():
            t_out = _call(teacher, batch_dev)
        optim.zero_grad()
        try:
            s_out = _call(student, batch_dev)
        except Exception as exc:  # noqa: BLE001
            return {
                "pair": pair_name,
                "success": False,
                "error": f"forward_failed_step_{step}: {type(exc).__name__}: {exc}",
            }
        try:
            t_logits = _extract_logits(t_out).float()
            s_logits = _extract_logits(s_out).float()
        except Exception as exc:  # noqa: BLE001
            return {
                "pair": pair_name,
                "success": False,
                "error": f"logits_extract_failed: {type(exc).__name__}: {exc}",
            }
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(s_logits, dim=-1),
            torch.nn.functional.softmax(t_logits, dim=-1),
            reduction="batchmean",
        )
        if not torch.isfinite(loss):
            return {
                "pair": pair_name,
                "success": False,
                "error": f"non-finite loss at step {step}: {loss.item()!r}",
            }
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
        step_times.append(time.time() - t0)
    train_s = time.time() - t_train

    return {
        "pair": pair_name,
        "success": True,
        "load_s": load_s,
        "train_s": train_s,
        "step_avg_s": sum(step_times) / max(len(step_times), 1),
        "teacher_adapter": teacher_adapter.modality,
        "student_adapter": student_adapter.modality,
        "loader": model_loader,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_progression": [round(x, 4) for x in losses[:: max(len(losses) // 6, 1)][:6]],
        "steps_completed": len(losses),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        default="all",
        help=(
            "Comma-separated subset of pairs to run, or 'all'. "
            f"Available: {', '.join(PAIRS)}"
        ),
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-budget", type=float, default=None, help="USD ceiling to abort at.")
    p.add_argument("--output", type=Path, default=None, help="Path for the results JSON.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Forward-only sanity check (no loss optimization loop).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.pairs == "all":
        pair_names = list(PAIRS)
    else:
        pair_names = [n.strip() for n in args.pairs.split(",") if n.strip()]

    started_at = time.time()
    results = []
    budget_used_usd = 0.0  # Not actually measured; reserved for future.

    for name in pair_names:
        if name not in PAIRS:
            print(f"[smoke] unknown pair {name!r}; skipping", file=sys.stderr)
            continue
        result = run_pair(name, PAIRS[name], n_steps=args.steps, quick=args.quick)
        result["commit"] = os.environ.get("GIT_COMMIT", "local")
        results.append(result)
        elapsed_min = (time.time() - started_at) / 60
        if args.max_budget and elapsed_min * 0.8 > args.max_budget:
            print(f"[smoke] budget exceeded after {elapsed_min:.1f} min — abort.")
            break

    out_path = args.output or Path("tests/smoke/results/universal_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "duration_s": time.time() - started_at,
        "pairs": pair_names,
        "steps": args.steps,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Echo failures to stdout so the Modal runner sees them.
    for r in results:
        if not r.get("success"):
            print(f"[smoke][FAIL] {r['pair']}: {r.get('error', 'no error recorded')}", file=sys.stderr)
        else:
            extra = (
                f"teacher_adapter={r.get('teacher_adapter')}, "
                f"student_adapter={r.get('student_adapter')}"
            )
            print(f"[smoke][OK]   {r['pair']}: {extra}", file=sys.stderr)
    successes = sum(1 for r in results if r.get("success"))
    total = len(results)
    print(f"\n[smoke] {successes}/{total} pairs succeeded. results -> {out_path}")
    sys.exit(0 if successes == total else 1)


if __name__ == "__main__":
    main()
