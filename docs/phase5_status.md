# Phase 5 — Status

This document is updated after each Phase 5 iteration.  Phase 5
covers DX, distributed training, and the docs site — the items
needed to call zynthe "deployable" and "release-ready" for v1.0.

## Done

### Iteration 4 — PTQ numerics-parity benchmark

- New `scripts/smoke/run_ptq_parity.py` — runs the same input
  through fp32 and dynamic-int8, records argmax agreement, abs
  diff max / mean, and size delta.  Writes JSON.
- New `scripts/smoke/run_ptq_parity_modal.py` — Modal L4 wrapper.
- New `tests/test_ptq_parity.py` — runs the local script and
  asserts the JSON summary has the expected fields.
- Verified on Modal L4: argmax agreement `True`, |diff| max 0.0125.
- `docs/quant.md` updated with the parity smoke table.
- `docs/benchmarks.md` updated with the parity row.

### Iteration 3 — docs site

- `mkdocs.yml` nav extended with: Distillation Methods,
  Quantization, Distributed Training, Benchmarks, Phase 3 Status,
  Phase 4 Status, Plan-Talk Handoff, Audit.
- New `tests/test_mkdocs_nav.py` — parses `mkdocs.yml` and asserts
  every referenced .md file exists on disk; phase3_status and
  HANDOFF must be present in the nav.

### Iteration 2 — typed Plan/Stage DSL + 5 new presets

- New `Plan` and `Stage` dataclasses in
  `zynthe.core.distillers.presets`.  Validation: Stage rejects
  empty loss / negative weight / zero epochs; Plan rejects empty
  name / empty stages / zero epochs.
- `Plan.to_dict` produces the same shape as `get_preset(name)` so
  the rest of the pipeline-builder code is unchanged.  `Plan.from_preset`
  round-trips legacy dicts.
- 5 new presets registered in `PRESET_LIBRARY`:
  - `compression_max` (heavy KD + feature L2)
  - `fidelity_first` (KD + CRD + PKT + aux heads)
  - `vision_default` (KD + feature L2 + attention)
  - `causal_lm_default` (KD with learnable τ + rationale)
  - `multimodal_default` (KD + feature + contrastive)
- 17 new tests pin the DSL.

### Iteration 1 — distributed-training integration

- New `DistributedConfig` and `prepare_distillation` in
  `zynthe.core.training.distributed` (re-exported from
  `zynthe.training`).
- `prepare_distillation` is a no-op when `enabled=False`; when
  `enabled=True`, runs through `accelerator.prepare(...)`.
- 6 new tests pin: defaults, validation, no-op, accelerate path.
- New `scripts/smoke/run_distributed.py` + `_local.py` — Modal L4
  smoke proves the integration on a single GPU.
- New `docs/distributed.md` — quick-start, config field reference,
  torchrun usage, smoke proof, references.

### Iteration 0 — scaffolding

- Created `docs/phase5_status.md` to track progress.

## Done

- [x] Iteration 1: DDP / accelerate integration
- [x] Iteration 2: typed Plan/Stage DSL + 5 presets
- [x] Iteration 3: docs site (MkDocs nav)
- [x] Iteration 4: PTQ numerics-parity benchmark

## Remaining (next phases, not Phase 5)

- [ ] v1.0 plan-talk: real GLUE benchmark, scripts/modal/ archive removal,
  coverage floor lift, mkdocs deploy to GitHub Pages.
- [ ] Numerics-parity test with **labelled** data (currently synthetic).

## Upcoming

- v1.0 plan-talk — see `docs/HANDOFF.md` for input.
