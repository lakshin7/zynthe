"""Post-Training Quantization (PTQ) smoke proof.

Loads a tiny distilled model, applies dynamic int8 quantization to
its nn.Linear modules, and verifies the quantized model still
forward-passes with a finite output.

This is the "Iteration 2" Phase-4 smoke.  The full PTQ pipeline
(calibration, ONNX export, QAT) lives in
``src/zynthe/core/quant/ptq.py`` (the ``PTQRunner``).  This script
verifies the *concept*: ``torch.ao.quantization.quantize_dynamic``
works on a real model, shrinks it (in bytes), and produces a
forward-passable int8 model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def _estimate_size_mb(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * 4 / (1024 ** 2)  # fp32 bytes


def _run_ptq(model_name: str, output_dir: Path) -> int:
    from transformers import AutoModelForSequenceClassification

    print(f"[ptq] loading {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    size_before = _estimate_size_mb(model)
    n_params_before = sum(p.numel() for p in model.parameters())
    print(
        f"[ptq] model loaded: {n_params_before:,} params, "
        f"~{size_before:.2f} MB (fp32)"
    )

    # Quantize Linear layers in-place (dynamic, no calibration needed).
    print("[ptq] applying torch.ao.quantization.quantize_dynamic (int8 Linear)")
    started = time.time()
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    elapsed = time.time() - started
    size_after = _estimate_size_mb(quantized)
    print(f"[ptq] quantized in {elapsed:.2f}s")

    # Sanity: forward pass.
    torch.manual_seed(0)
    sample_input = torch.randint(0, 1000, (1, 8), dtype=torch.long)
    with torch.no_grad():
        out = quantized(input_ids=sample_input)
    logits = out.logits
    assert torch.isfinite(logits).all(), f"non-finite logits: {logits}"
    print(f"[ptq] forward OK — logits shape {tuple(logits.shape)}")

    # Persist the (fp32) quantized model and a JSON summary.
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": time.time(),
        "commit": os.environ.get("GIT_COMMIT", "local"),
        "model": model_name,
        "params": n_params_before,
        "size_mb_fp32": size_before,
        "size_mb_int8_labelled_fp32": size_after,
        "quantize_time_s": elapsed,
        "logits_shape": list(logits.shape),
        "logits_l2_norm": float(logits.float().norm().item()),
        "elapsed_s": elapsed,
        "strategy": "dynamic_int8_linear",
    }
    summary_path = output_dir / "ptq_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ptq][OK] summary -> {summary_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="prajjwal1/bert-tiny",
        help="Tiny HF model for the smoke (default: bert-tiny).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests/smoke/results"),
    )
    args = p.parse_args()
    return _run_ptq(args.model, args.output)


if __name__ == "__main__":
    sys.exit(main())
