# Zynthé — v1.0 Plan-Talk Handoff

**Last updated:** Phase 4 close.  Read this when picking up zynthe
in a new session.

## Where zynthe is right now

- **Phase 0** (foundation) — done.  Exceptions, deterministic
  seeding, strict layer match, smoke gate architecture.
- **Phase 1** (test backbone) — done.  100+ unit tests pinning math
  across 5 distillers + adapters + pipelines.
- **Phase 2** (universal model support) — done.  9-adapter registry,
  5-family smoke gate green on Modal L4.
- **Phase 3** (SOTA losses + paper-aligned) — done.  8 distillers
  (KD-Hinton, Feature, Attention, Similarity, Contrastive, Relational,
  Projection, AuxHead, Rationale) + 2 KD-Hinton regularizers
  (entropy, dynamic τ) + 1 dataset (RationaleDataset).
- **Phase 4** (throughput + quant) — done.  bf16 autocast,
  torch.compile opt-in, grad_checkpointing opt-in, PTQ smoke
  proof.

## Test count

- 244 unit tests pass cleanly on Modal L4 under
  `-W error::UserWarning`.
- 5/5 universal-model smoke gate pairs (bert / vit / gpt2 / clip /
  resnet) succeed end-to-end.
- 4 smoke scripts verified on Modal L4:
  - `run_baseline_distill.py` — KD-Hinton (BERT→bert-tiny), 30 steps, decay 0.0142.
  - `run_rationale_distill.py` — RationaleDistiller (T5-tiny), 20 steps, decay 1.12.
  - `run_ptq.py` — Dynamic int8 PTQ, 0.15s.
  - `run_smoke_modal.py` — universal gate (5/5).

## Status docs (read these in order)

1. `docs/phase3_status.md` — last full phase (8 iterations).
2. `docs/phase4_status.md` — current phase (2 iterations).
3. `docs/benchmarks.md` — empirical smoke proof.
4. `docs/audit.md` — Phase 0 audit, dated.
5. `docs/distillation-methods.md` — catalogue of every distiller.
6. `docs/adapters.md` — adapter registry + detection heuristics.
7. `docs/quant.md` — quantization workflow.
8. `docs/usage_guide.md` — entry-point guide.
9. `docs/api_reference.md` — public API surface.

## Outstanding from the *zynthe report*

- [x] Report §201-225 (Theoretical Enhancements 1-6) — covered.
- [x] Report §227-249 (Experiments 1-6) — smoke proof, not full benchmark.
- [ ] **Coverage floor** — pyproject `fail_under = 0` (we never set the
  gate to 25 %+).  Phase 5 will need to ramp this as we add tests.
- [ ] **Modular archive** — `scripts/modal/` is currently still
  committed (`.gitignore`d since Phase 2 but the existing file
  lives in history).  Phase 5 should `git rm -r scripts/modal/`
  once the maintainer is happy with the workflow.

## Phase 5 plan (deferred — to discuss in next plan-talk)

- **Iteration 1**: Distributed training smoke.
  - Add `accelerate` integration to `UnifiedTrainingRuntime`.
  - Single-GPU + 2×L4 DDP smoke (Modal supports multi-GPU).
  - 8 tests + smoke proof on a tiny pair.
- **Iteration 2**: Preset DSL rewrite.
  - Replace `get_preset(name)`-style calls with a typed
    `Plan(name, stages=[Stage(loss=..., weight=...)])` DSL.
  - 5 new presets: `compression_max`, `fidelity_first`,
    `vision_default`, `causal_lm_default`, `multimodal_default`.
- **Iteration 3**: Docs site / `mkdocs` publish.
  - Material for MkDocs + serve locally.
  - Publish to GitHub Pages (already wired but never used).
  - 1 page per distiller; auto-generated API reference.
- **Iteration 4**: Numerics-parity benchmark for PTQ.
  - Run a real GLUE task (sst2) on tiny-bert, compare fp32 vs int8
    accuracy delta.  ~$5 on Modal A100.

## Risks the next plan-talk should address

1. **Context budget** — this session is ~150k tokens in.  Each
   new phase needs to be lean or we hit limits.  Plan in tight
   iterations (≤ 4), no in-flight design changes.

2. **Coverage floor** — current gate is `fail_under = 0`.  Phase 5
   should set it to ~25 % (lift to 50 % by end of v1.0, ramp in
   25 % steps).

3. **The `scripts/modal/` archive** — 4 Modal runner scripts
   (`run_tests.py`, `run_smoke_modal.py`, `run_baseline_distill.py`,
   `run_rationale_distill.py`, `run_ptq_modal.py`) currently
   live in the repo.  They've been `.gitignore`d since Phase 2
   but the existing files are still tracked.  Plan: `git rm -r
   scripts/modal/` after the maintainer is happy with the smoke
   workflow.  Don't do this in a plan-talk session — it needs a
   human sign-off.

4. **Long-running CI** — no GitHub Actions CI is set up beyond the
   existing `ci.yml` and `docs.yml`.  A nightly Modal CI is
   desirable but is gated on `MODAL_TOKEN_ID` /
   `MODAL_TOKEN_SECRET` GitHub secrets.  Phase 5 can wire this.

5. **zync report.md** — the report itself is now a historical
   document.  Once Phase 5 lands, archive it in `docs/reports/`
   to keep the repo root tidy.

## What the next plan-talk should produce

A clean Phase 5 plan (4 iterations max), focused on:
- making the repo *deployable* (DDP, docs site),
- not adding new distillers (we have 9 + 2 KD-Hinton variants — enough),
- removing the `scripts/modal/` archive post-sign-off,
- setting a coverage floor.

If after Phase 5 we still have budget, the natural follow-up is
"v1.0 release" — bumping the version, releasing to PyPI, doing a
GLUE benchmark run.  That can be the *next* plan-talk.
