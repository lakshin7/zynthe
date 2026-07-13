# Distributed Training (Phase 5 Iteration 1)

Zynthé integrates with HuggingFace `accelerate` to provide a
single-flag move from single-GPU to multi-GPU DDP.

## Quick start

```python
from zynthe.core.training.distributed import (
    DistributedConfig,
    prepare_distillation,
)

cfg = DistributedConfig(
    enabled=True,
    mixed_precision="bf16",
    num_processes=2,
    gradient_accumulation_steps=4,
)
bundle = prepare_distillation(teacher, student, optimizer, loader, config=cfg)

# bundle.teacher, bundle.student, bundle.optimizer, bundle.loader
# are now wrapped in accelerator.prepare(...).
```

`enabled=False` (the default) makes `prepare_distillation` a no-op
that returns the inputs unchanged, so your existing training loop
runs unchanged when distributed training is off.

## `DistributedConfig` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | when True, prepare through accelerate. |
| `mixed_precision` | `"no"` / `"fp16"` / `"bf16"` | `"no"` | `accelerate` autocast. |
| `num_processes` | `int` / `"auto"` | `"auto"` | accelerate discovers from env. |
| `mixed_precision_dtype` | `str` / `None` | `None` | explicit dtype override. |
| `gradient_accumulation_steps` | `int` | `1` | micro-batches per optimiser step. |
| `cpu` | `bool` | `False` | force CPU even when GPUs are present. |
| `extra` | `dict` | `{}` | extra kwargs passed to `Accelerator(**kwargs)`. |

## Running DDP via `torchrun`

```bash
torchrun --nproc_per_node=2 scripts/smoke/run_distributed_local.py
```

The script's `prepare_distillation(..., config=DistributedConfig(enabled=True, num_processes="auto"))`
hands the distiller/optimizer/dataloader to `accelerate.prepare(...)`.
accelerate detects the world size from `torchrun`'s environment and
wraps the model in `DistributedDataParallel` automatically.

## Smoke proof

`scripts/smoke/run_distributed.py` runs the local script on Modal L4
(single-GPU, since the Modal runner can only request a single GPU
per function).  20 SGD steps, loss 0.69 → 0.66.

A real DDP run (multi-GPU) requires a multi-GPU Modal function.  The
plan for that is in `docs/HANDOFF.md` ("DDP via `torchrun`" section).

## Why we don't ship a numerics-parity test for DDP

DDP on a single tiny pair over 20 steps is more about confirming
the *infrastructure* (accelerate is wired, the bundle survives
`step()` calls) than measuring training speed.  Numerical parity
is identical to single-GPU for tiny models; the wins from DDP
appear only at scale.  That scale is what `docs/quant.md` is for
(int8 quantisation smoke) and what Phase-5 Iteration 4 will do for
PTQ numerics parity.

## References

- HuggingFace `accelerate`: <https://huggingface.co/docs/accelerate>
- PyTorch DDP: <https://pytorch.org/docs/stable/notes/ddp.html>
