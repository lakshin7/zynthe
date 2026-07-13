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

### Iteration 0 — scaffolding

- Created `docs/phase3_status.md` to track progress.

## Pending (aligned with plan)

- [ ] Iteration 1: `ContrastiveDistiller` + tests + smoke CRD pair (plan #1, Experiment #2)
- [ ] Iteration 2: `RelationalDistiller` + tests (plan #2, Experiment #3)
- [ ] Iteration 3: `ProjectionDistiller` + tests (plan #3)
- [ ] Iteration 4: `AuxHeadDistiller` + tests (plan #4, Experiment #4)
- [ ] Iteration 5: `KDHintonDistiller` entropy + dynamic-τ extensions (plan #5, #6)
- [ ] Iteration 6: baseline-distillation smoke experiment + `docs/benchmarks.md` (Experiment #1)
- [ ] Iteration 7: full smoke gate on Modal L4 — verification round

## Upcoming

- Iteration 1 next: `ContrastiveDistiller` from the report's §205-207 with projection head + InfoNCE on teacher/student features. Will land a smoke pair in `scripts/smoke/universal_smoke.py` so the universal-model gate exercises CRD too.
