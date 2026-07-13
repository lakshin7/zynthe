# Phase 4 — Status

This document is updated after each Phase 4 iteration.  Phase 4
implements throughput hardening (bf16 / torch.compile /
gradient_checkpointing) and a quantization (PTQ) proof.

## Done

### Iteration 2 — Quantization (PTQ) smoke + docs/quant.md

- New `scripts/smoke/run_ptq.py` — dynamic int8 PTQ on a tiny HF
  model via `torch.ao.quantization.quantize_dynamic`.  Verified on
  Modal L4: 4.39 M params quantized in 0.15 s, forward OK.
- New `scripts/smoke/run_ptq_modal.py` — Modal wrapper.
- New `tests/test_quant_smoke.py` — unit guard that
  `PTQRunner.run()` is wired.
- New `docs/quant.md` — workflow doc covering dynamic / static /
  QAT paths, with explicit deferral of numerics-parity benchmarking
  to Phase 5.

### Iteration 1 — Throughput hardening (bf16 / compile / grad-ckpt)

- `BaseDistiller` now opt-in supports three config flags under
  `config['distill']`:
  - `precision: 'bf16'` — `torch.autocast(bfloat16)` around the
    student forward + loss computation.  Teacher stays in fp32
    (frozen soft targets stay numerically stable).  No-op on CPU.
  - `compile: true` — wraps the student in `torch.compile`.  Falls
    back to eager on failure with a warning.
  - `grad_checkpointing: true` — calls HF
    `student.gradient_checkpointing_enable()` if the student
    supports it.  Plain `nn.Module` students no-op silently.
- 9 new tests in `tests/test_throughput_flags.py` pin the flag
  contract (records the flag, doesn't break forward, calls the HF
  hook on HF students, keeps eager on plain students).
- Verified on Modal L4: 244/244 unit tests pass under
  `-W error::UserWarning`.

### Iteration 0 — scaffolding

- Created `docs/phase4_status.md` to track progress.

## Done

- [x] Iteration 1: throughput flags (bf16 / compile / grad-ckpt)
- [x] Iteration 2: PTQ smoke + docs/quant.md

## Remaining (next phases, not Phase 4)

- [ ] Phase 5: Distributed training (accelerate + DDP smoke)
- [ ] Phase 5: Preset DSL rewrite
- [ ] Phase 5: Docs site / `mkdocs` publish
- [ ] Phase 5: Numerics-parity benchmark for PTQ (int8 vs fp32)

## Upcoming

- Phase 5: DX / DDP / docs.  See `docs/HANDOFF.md` for the v1.0
  plan-talk input.
