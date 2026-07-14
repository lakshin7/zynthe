"""PTQ numerics-parity benchmark (Phase 5 Iteration 4).

Compares fp32 vs dynamic-int8 inference on a tiny sentiment
classification task (prajjwal1/bert-tiny, synthetic data).  Records
the per-tensor difference in logits + the size delta.

This is a "small" benchmark — we measure how much int8 deviates from
fp32 *per logit*, on a deterministic synthetic input.  It is NOT a
GLUE accuracy number; that needs a real labelled dataset and lands
in the v1.0 plan-talk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        default="prajjwal1/bert-tiny",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests/smoke/results"),
    )
    args = p.parse_args()

    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(args.seed)

    print(f"[ptq-parity] loading {args.model}")
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(args.model)
    model_fp32.eval()
    size_fp32 = sum(p.numel() for p in model_fp32.parameters()) * 4 / (1024 ** 2)

    # Synthetic input: a deterministic batch.
    input_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])

    with torch.no_grad():
        out_fp32 = model_fp32(input_ids=input_ids, attention_mask=attention_mask)
    logits_fp32 = out_fp32.logits.float().numpy().tolist()

    # Quantize (dynamic int8) and re-forward.
    print("[ptq-parity] applying quantize_dynamic (int8 Linear)")
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()
    size_int8 = sum(p.numel() for p in model_int8.parameters()) * 4 / (1024 ** 2)

    with torch.no_grad():
        out_int8 = model_int8(input_ids=input_ids, attention_mask=attention_mask)
    logits_int8 = out_int8.logits.float().numpy().tolist()

    # Per-tensor diff stats.
    diffs: list[float] = []
    for r_fp32, r_int8 in zip(logits_fp32, logits_int8):
        for v_fp32, v_int8 in zip(r_fp32, r_int8):
            diffs.append(abs(v_fp32 - v_int8))
    abs_max = max(diffs)
    abs_mean = sum(diffs) / len(diffs)

    # Argmax agreement: did the predicted class change?
    pred_fp32 = max(range(len(logits_fp32[0])), key=lambda i: logits_fp32[0][i])
    pred_int8 = max(range(len(logits_int8[0])), key=lambda i: logits_int8[0][i])
    agree = pred_fp32 == pred_int8
    print(
        f"[ptq-parity] fp32 size {size_fp32:.2f} MB | int8 size {size_int8:.2f} MB | "
        f"argmax agree: {agree} (fp32={pred_fp32} int8={pred_int8}) | "
        f"|diff| max={abs_max:.4f} mean={abs_mean:.4f}"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "commit": os.environ.get("GIT_COMMIT", "local"),
        "model": args.model,
        "size_mb_fp32": size_fp32,
        "size_mb_int8_labelled_fp32": size_int8,
        "abs_diff_max": abs_max,
        "abs_diff_mean": abs_mean,
        "logits_fp32": logits_fp32,
        "logits_int8": logits_int8,
        "pred_fp32": pred_fp32,
        "pred_int8": pred_int8,
        "argmax_agree": bool(agree),
        "seed": args.seed,
    }
    out_path = args.output / "ptq_parity.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ptq-parity][OK] summary -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
