# Phase 3 — Status

This document is updated after each Phase 3 iteration. It tracks what's
been shipped, what is still in progress (against the *zynthe report*),
and what's queued next.

Phase 3 implements the report's §"Theoretical Enhancements"
(lines 201-225) end-to-end, plus a smoke-scale run of the report's
§"Proposed Experiments Plan" Experiment #1 (Baseline Distillation),
plus a paper-aligned §Iteration 8 (Distill step-by-step, Hsieh et al.
2023).

Phase 3 plan reference — plan approved, scope:

| # | Idea | Source line |
|---|------|-------------|
| 1 | Contrastive Distillation (InfoNCE / CRD) | 205-207 |
| 2 | Relational Distillation (pairwise cosine) | 209-211 |
| 3 | Attention to Embedding (translator projection) | 213 |
| 4 | Intermediate Classifier Distillation (aux heads) | 215-217 |
| 5 | Entropy-based Regularizer | 219 |
| 6 | Dynamic Temperature | 221 |
| 7 | MoE / NAS | 223 (out of scope per the report) |
| 8 | Distill step-by-step (rationale multi-task) | Hsieh et al. 2023 (paper) |

Plus the baseline-distillation smoke experiment (report §227-249,
Experiment #1) and the docs/benchmarks.md update.

---

## Done

### Iteration 8 — RationaleDistiller (Distill step-by-step) [Hsieh et al. 2023]

- New `RationaleDistiller` (text-to-text multi-task): `CE_label +
  λ·CE_rationale`.  Dict-input contract: `student_outputs={'label_logits',
  'rationale_logits'}`, `targets={'label_ids', 'rationale_ids'}`.
  ignore_index support; NaN-safe for all-ignore batches.
- New `RationaleDataset` — JSONL of `(input, label, rationale)`
  triples.  Permissive / strict modes; no third-party deps.
- New `docs/distillation-methods.md` — curated table of every
  distiller + method citations.
- New `scripts/smoke/run_rationale_distill.py` + `_local.py` —
  Modal + local smoke proof.  Verified on Modal L4: 20 steps in
  1.9 s, loss 18.46 → 17.34.
- Wired as `'rationale'` and `'distill_step_by_step'` in
  `DistillerRegistry`.  Exposed from `core.distillers` and
  `core.data`.
- 12 new tests pin: closed-form math, weight asymmetry,
  rationale_weight=0, ignore_index handling, gradient flow,
  TypeError on non-dict inputs, dataset roundtrip, missing-key
  strict / permissive, missing-file, registry wiring.

### Iteration 7 — full smoke gate verification on Modal L4

- Modal-verified: 234/234 unit tests pass under `-W error::UserWarning`.
- Modal-verified: 5/5 universal-model smoke gate pairs succeed
  (bert / vit / gpt2 / clip / resnet) at Modal commit `e2c0dec`.

### Iteration 6 — baseline-distillation smoke experiment + docs/benchmarks.md [Experiment #1, §227-249]

- Added `scripts/smoke/run_baseline_distill.py` (Modal wrapper) and
  `scripts/smoke/run_baseline_distill_local.py` (CPU/GPU-runnable
  sibling).
- 30 SGD steps on Modal L4 in 1.3s; first loss 0.3526 → last loss
  0.3384; smoke criterion (finite + decay > 0) met.
- Wrote `docs/benchmarks.md` documenting the result + the Phase-2
  universal-model smoke gate.

### Iteration 5 — KD-Hinton entropy + dynamic temperature [§219, §221]

- New `entropy_regularizer_weight` config knob (default 0.0): when
  enabled, adds `|H(σ(s/T)) − H(σ(t/T))|` to the KD loss.
- New `dynamic_temperature: 'learnable'` config knob: registers an
  `nn.Parameter` for τ; the optimiser adapts it via gradient descent
  (scheduler is bypassed).  τ is clamped to `[0.1, 10.0]`.
- 5 new tests; total KD-Hinton tests: 15 (all green).

### Iteration 4 — AuxHeadDistiller (intermediate classifiers) [§215-217, Experiment #4]

- New `AuxHeadDistiller`: lazy aux heads attached to configured
  student layers, each a small MLP; loss = mean of per-layer CE on
  labels.
- Wired as `'aux_head'` and `'aux'` in DistillerRegistry.
- 9 new tests pin aux head output shape, loss composition, strict
  layer match, FP16 stability, gradient flow, pooling.

### Iteration 3 — ProjectionDistiller (translator projection) [§213]

- New `ProjectionDistiller`: learnable translator MLP maps student
  features to teacher's hidden dim; MSE loss on aligned hidden states.
- Wired as `'projection'` and `'translator'` in DistillerRegistry.
- 10 new tests pin translator dim, closed-form MSE reference,
  mismatched widths, strict layer match, FP16 stability, gradient
  flow, pooling.

### Iteration 2 — RelationalDistiller (PKT) [§209-211, Experiment #3]

- New `RelationalDistiller`: PKT-style loss — pairwise cosine
  similarity matrix between student and teacher features; MSE
  between the two similarity matrices.
- Wired as `'relational'` and `'pkt'` in DistillerRegistry.
- 10 new tests pin pairwise cosine properties, closed-form MSE
  reference, FP16 stability, mismatched widths, strict layer match.

### Iteration 1 — ContrastiveDistiller (CRD) [§205-207, Experiment #2]

- New `ContrastiveDistiller`: InfoNCE-style loss on projected
  student/teacher features with in-batch negatives; optional
  memory bank for negative samples.
- Wired as `'contrastive'` and `'crd'` in DistillerRegistry.
- 12 new tests pin pooling, L2-normalisation, reference value,
  FP16 stability, gradient flow, memory bank eviction, strict
  layer match.

### Iteration 0 — scaffolding

- Created `docs/phase3_status.md` to track progress.

## Done

- [x] Iteration 1: ContrastiveDistiller + tests
- [x] Iteration 2: RelationalDistiller + tests
- [x] Iteration 3: ProjectionDistiller + tests
- [x] Iteration 4: AuxHeadDistiller + tests
- [x] Iteration 5: KDHinton entropy + dynamic-τ extensions
- [x] Iteration 6: baseline-distillation smoke + benchmarks.md
- [x] Iteration 7: full smoke gate on Modal L4
- [x] Iteration 8: RationaleDistiller (Distill step-by-step)

## Remaining (next phases, not Phase 3)

- [ ] Phase 4: bf16 autocast + torch.compile + gradient_checkpointing
- [ ] Phase 4: Quantization (PTQ benchmarks, docs/quant.md)
- [ ] Phase 5: Distributed training (accelerate + DDP smoke)
- [ ] Phase 5: Preset DSL rewrite + docs site

## Upcoming

- Phase 4: throughput hardening (bf16 / torch.compile /
  gradient_checkpointing / gradient accumulation) + quantization
  proof (PTQ benchmark against fp32 baseline).  Will land under a
  similar per-iteration status file: `docs/phase4_status.md`.
