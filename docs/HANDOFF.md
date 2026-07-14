# Zynthé — v1.0 Plan-Talk Handoff

**Last updated:** Phase 5 close.  Read this when picking up zynthe
in a new session.

## Where zynthe is right now

- **Phase 0** (foundation) — done.
- **Phase 1** (test backbone) — done.
- **Phase 2** (universal model support) — done.
- **Phase 3** (SOTA losses + paper-aligned) — done.
- **Phase 4** (throughput + quant) — done.
- **Phase 5** (DX / DDP / docs / parity) — done.

**278 unit tests, all green on Modal L4 under `-W error::UserWarning`.**

## Test inventory (208 test functions across 27 files)

```
tests/test_accuracy_improvements.py        2 tests
tests/test_adapters.py                     9 tests
tests/test_adapters_extended.py          21 tests
tests/test_attention_transfer.py         11 tests
tests/test_aux_head_distiller.py          7 tests
tests/test_base_distiller.py              8 tests
tests/test_causal_lm_engine.py           10 tests
tests/test_contrastive_distiller.py      10 tests
tests/test_dataloaders.py                 2 tests
tests/test_ddp_script.py                   2 tests
tests/test_distributed.py                  6 tests
tests/test_distillers.py                   2 tests
tests/test_distillers_compute_loss.py      6 tests
tests/test_feature_distiller.py           11 tests
tests/test_imports.py                      2 tests
tests/test_kd_hinton.py                   10 tests
tests/test_mkdocs_nav.py                   3 tests
tests/test_optional_imports.py             3 tests
tests/test_pipeline_builder.py            9 tests  (tests/test_pipelines/)
tests/test_multi_stage_pipeline.py        5 tests  (tests/test_pipelines/)
tests/test_preflight.py                    6 tests
tests/test_preprocessing.py                2 tests
tests/test_presets_dsl.py                 14 tests
tests/test_projection_distiller.py         8 tests
tests/test_ptq_parity.py                   1 test
tests/test_quant_smoke.py                  1 test
tests/test_rationale_distiller.py        12 tests
tests/test_relational_distiller.py         8 tests
tests/test_similarity_transfer.py         10 tests
tests/test_throughput_flags.py            9 tests
```

Group summary:
- 9 distillers + KD-Hinton extensions: 11 files, ~78 tests
- 9-adapter registry: 2 files, ~30 tests
- Pipeline / Multi-stage: 2 files, 14 tests
- 5 family universal-model smoke: 1 file (test_optional_imports.py)
- Throughput flags: 1 file, 9 tests
- DDP / distributed: 1 file, 6 tests (+2 DDP script tests)
- Preset DSL: 1 file, 14 tests
- Docs site (MkDocs nav): 1 file, 3 tests
- PTQ numerics-parity: 1 file, 1 test (+1 PTQRunner smoke test)
- Resource probe / preflight: 1 file, 6 tests
- Dataloaders / preprocessing / accuracy: ~6 tests
- Optional imports / version: 5 tests
- Compute-loss integration: 6 tests
- Migration / smoke gating: 12 tests

## Modal examples (run from repo root on Modal L4)

### Run the full test suite (single-GPU)

```bash
modal run scripts/modal/run_tests.py --gpu L4
```

### Universal-model 5-family smoke gate

```bash
modal run scripts/smoke/run_smoke_modal.py --gpu L4 --pairs all --steps 2
# Subsets:
modal run scripts/smoke/run_smoke_modal.py --gpu L4 --pairs bert
modal run scripts/modal/run_smoke.py --gpu L4 --pairs bert
# Quick mode (no optimisation loop):
modal run scripts/modal/run_smoke.py --gpu L4 --pairs bert --quick
```

### Distillation smokes (single-GPU)

```bash
# Baseline distillation (KD-Hinton): tiny BERT → bert-tiny, 30 steps
modal run scripts/smoke/run_baseline_distill.py --gpu L4 --steps 30

# Rationale distillation (Distill step-by-step): tiny T5, 20 steps
modal run scripts/smoke/run_rationale_distill.py --gpu L4 --steps 20

# Distributed-training (single-GPU accelerator): bert-tiny, 20 steps
modal run scripts/smoke/run_distributed.py --gpu L4 --steps 20

# PTQ smoke (dynamic int8 quantisation): bert-tiny
modal run scripts/smoke/run_ptq_modal.py --gpu L4

# PTQ numerics-parity: fp32 vs int8, argmax agreement
modal run scripts/smoke/run_ptq_parity_modal.py --gpu L4
```

### Multi-GPU DDP (torchrun)

```bash
# 2 L4 GPUs (capped at 5):
modal run scripts/smoke/run_distributed_ddp.py --gpus 2 --steps 10
# 4 L4 GPUs:
modal run scripts/moke/run_distributed_ddp.py --gpus 4 --steps 20
```

**Caveat observed:** the Modal L4 image ships kernel 4.19.0, below
accelerate's recommended 5.5.0.  The DDP smoke reaches the
`rank=N world_size=N` print stage and then stalls.  Single-GPU
`Accelerator` is fully green on the same Modal L4 (loss 0.68 → 0.65
in 20 steps).  For full DDP verification, run on a 5.5+ kernel host
(local machine or a Modal image with `--kernel-version` set).

## Status docs (read these in order)

1. `docs/phase3_status.md` — 8 iterations, all theoretical enhancements + paper.
2. `docs/phase4_status.md` — 2 iterations (throughput + quant).
3. `docs/phase5_status.md` — 4 iterations (DX / DDP / docs / parity).
4. `docs/benchmarks.md` — every Modal-verified smoke result in one place.
5. `docs/audit.md` — Phase 0 audit.
6. `docs/distillation-methods.md` — catalogue of every distiller.
7. `docs/adapters.md` — adapter registry + detection heuristics.
8. `docs/quant.md` — quantisation workflow.
9. `docs/distributed.md` — accelerate/DDP usage.
10. `docs/usage_guide.md` / `docs/api_reference.md` — entry points.

## Outstanding from the *zynthe report*

- [x] Report §201-225 (Theoretical Enhancements 1-6) — covered.
- [x] Report §227-249 (Experiments 1-6) — smoke proof, not full benchmark.
- [ ] **Real GLUE benchmark** — needs a labelled dataset run on Modal
  A100.  Was deferred from Phase 5.
- [ ] **Coverage floor** — `pyproject fail_under = 0` (we've never set
  it to ≥ 25 %).  The next plan-talk should set this.
- [ ] **`scripts/modal/` archive** — currently committed.  After
  maintainer sign-off, `git rm -r scripts/modal/` to clean up.
- [ ] **Long-running CI** — no Modal CI workflow in `.github/`.  Needs
  GitHub secrets `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` to enable.
- [ ] **`zynthe_report.md`** — historical, archive in `docs/reports/`.

## v1.0 plan-talk input

If the next session is a v1.0 plan-talk, the natural scope is:

1. Real GLUE benchmark (e.g. SST-2 on tiny-bert, or full bert-base
   on Modal A100).  ~$5-15 of Modal credit.
2. Coverage floor lift to ≥ 25 % (run a coverage report, see what's
   still uncovered, close gaps).
3. `scripts/modal/` archive decision: keep or remove.
4. GitHub Actions Modal CI workflow.
5. MkDocs deploy to GitHub Pages (workflow already wired, never used).
6. Archive `zynthe_report.md` to `docs/reports/`.
7. `pyproject.toml` version bump → 0.3.0 + PyPI release.

If after v1.0 we have budget, the natural follow-up is the next
plan-talk (maintenance, more benchmarks, etc.).
