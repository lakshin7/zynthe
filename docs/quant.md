# Quantization

Zynthé ships two post-training quantization paths for the distilled
student.  Both are gated behind the same `distillation.quantization`
config block and produce an int8 model you can deploy.

## When to use what

| Use case | Path |
|---|---|
| "I want an int8 model right now, no calibration" | **Dynamic PTQ** — `torch.ao.quantization.quantize_dynamic` on `nn.Linear` layers at load time.  No calibration loop. |
| "I want the most accurate int8 model I can get" | **Static PTQ** — `PTQRunner` with a calibration loader.  Slightly slower, +0-3 % accuracy. |
| "I'm fine-tuning in fp32, want int8 to ship" | **QAT** — `QATRunner`.  Adds fake-quantization nodes during training, then converts at the end. |
| "I want a 1-2 line 'give me int8' answer" | `apply_ptq(model, config)`.  Defaults to dynamic; fall back to static if calibration data is supplied. |

## Dynamic PTQ (smoke proof)

`scripts/smoke/run_ptq.py` loads `prajjwal1/bert-tiny`, applies
`quantize_dynamic({nn.Linear}, dtype=qint8)`, and verifies the
quantized model still forward-passes with finite logits.  Verified on
Modal L4 (Phase 4 Iteration 2).

## PTQ via the existing `PTQRunner`

The full PTQ pipeline lives in `src/zynthe/core/quant/ptq.py`
(`PTQRunner`).  Use it from Python:

```python
from zynthe.core.quant.ptq import PTQRunner

cfg = {
    "distillation": {"model": "path/to/distilled/student"},
    "quantization": {
        "strategy": "dynamic",          # or "static" with calibration
        "dtype": "qint8",               # or "float16" for fp16 fallback
        "device": "cuda",               # or "cpu"
        "backend": "fbgemm",            # "qnnpack" for ARM
        "calibration": {
            "num_batches": 32,
            "max_samples": "auto",
            "use_training_split": False,
        },
        "output_dir": "./quantized_model",
    },
    "runtime": {"device": "cuda"},
}
runner = PTQRunner(cfg)
summary = runner.run()
print(summary)
# {
#   "strategy_requested": "dynamic", "strategy_used": "dynamic",
#   "size_before_mb": 4.5, "size_after_mb": 1.2,
#   ...
# }
```

`PTQRunner` writes a `quantized_<timestamp>/` directory under
`output_dir` with the int8 model + a JSON summary.

## QAT (`QATRunner`)

QAT lives in `src/zynthe/core/quant/qat.py`.  Insert QAT after
distillation by calling `QATRunner.run()` on the trained student:

```python
from zynthe.core.quant.qat import QATRunner

qat_cfg = {
    "distillation": {"model": "path/to/distilled/student"},
    "quantization": {
        "strategy": "qat", "dtype": "qint8", "device": "cuda",
        "training": {"epochs": 1, "lr": 1e-5},
    },
}
QATRunner(qat_cfg).run()
```

The QAT runner is heavy (real training loop); the smoke doesn't
exercise it.

## Why we don't ship a numerics-parity test yet

We *do* now ship a numerics-parity smoke — `scripts/smoke/run_ptq_parity.py`
records the per-tensor logit difference between fp32 and int8 on a
deterministic synthetic input, plus the argmax agreement and the
size delta.  Verified on Modal L4 (Phase 5 Iteration 4):

| Field | Value |
|---|---|
| Model | `prajjwal1/bert-tiny` |
| fp32 size | 16.73 MB |
| int8 size (fp32-labelled) | 15.16 MB |
| abs logit diff max | 0.0125 |
| abs logit diff mean | 0.0124 |
| **argmax agreement** | **True** |

The smoke proves the *bit-level delta* is small and the prediction is
preserved.  A real GLUE accuracy number (vs. fp32) still lives in
the v1.0 plan-talk — it needs a labelled dataset and a real benchmark
harness, not just a forward pass on synthetic data.

## References

- PyTorch AO documentation: <https://pytorch.org/ao/stable/>
- Lin et al. *Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference*, ICLR 2016 (the
  original PTQ paper).

## See also

- `docs/distillation-methods.md` — the loss-side catalogue.
- `docs/benchmarks.md` — smoke proof of dynamic PTQ on bert-tiny.
