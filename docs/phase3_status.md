# Phase 3 — Status

This document is updated after each Phase 3 iteration. It tracks what's
been shipped, what is still in progress (against the *zynthe report*),
and what's queued next.

Phase 3 implements the report's §"Theoretical Enhancements"
(lines 201-225) end-to-end, plus a smoke-scale run of the report's
§"Proposed Experiments Plan" Experiment #1 (Baseline Distillation).

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

Plus the baseline-distillation smoke experiment (report §227-249,
Experiment #1) and the docs/benchmarks.md update.

---

## Done

### Iteration 1 — ContrastiveDistiller (CRD) [report §205-207]

- Added `ContrastiveDistiller` in `src/zynthe/core/distillers/contrastive_distiller.py`:
  - Per-sample projection heads (student + teacher) — small 2-layer MLPs with L2-normalised output.
  - InfoNCE loss on (student, teacher) projections with in-batch negatives; optional memory bank.
  - Pooling for 2-D / 3-D / 4-D feature tensors.
  - Phase-0 `strict_layer_match` flag raises `ConfigError` on missing layers.
- Wired into `DistillerRegistry` as `'contrastive'` and `'crd'`.
- Exposed `ContrastiveDistiller` from `zynthe.core.distillers`.
- Added `tests/test_contrastive_distiller.py` with 12 tests:
  - Pooling across 2-D / 3-D / 4-D shapes.
  - L2-normalisation of projection-head output.
  - InfoNCE reference value (closed-form vs the distiller).
  - 2-D feature path.
  - Batch size 1 returns zero (degenerate).
  - Projection heads trainable; teacher backbone still frozen.
  - Memory bank grows up to its size then evicts oldest.
  - Strict layer match raises `ConfigError`.
  - FP16 stability under extreme inputs.
  - Gradient flow through student projection only (teacher detached).
- Verified on Modal L4: 12/12 pass with `-W error::UserWarning`.

### Iteration 0 — scaffolding

- Created `docs/phase3_status.md` to track progress.

## Pending (aligned with plan)

- [ ] Iteration 2: `RelationalDistiller` + tests (plan #2, Experiment #3)
- [ ] Iteration 3: `ProjectionDistiller` + tests (plan #3)
- [ ] Iteration 4: `AuxHeadDistiller` + tests (plan #4, Experiment #4)
- [ ] Iteration 5: `KDHintonDistiller` entropy + dynamic-τ extensions (plan #5, #6)
- [ ] Iteration 6: baseline-distillation smoke experiment + `docs/benchmarks.md` (Experiment #1)
- [ ] Iteration 7: full smoke gate on Modal L4 — verification round

## Upcoming

- Iteration 2 next: `RelationalDistiller` from the report's §209-211 with pairwise cosine matrix loss on student/teacher features.
