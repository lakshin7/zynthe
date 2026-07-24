# Phase 6 — Status

## Done

### Iteration 1 — LLM rationale extractor

- New `scripts/extract_rationales.py` — re-implements the
  rationale-extraction step from Hsieh et al. 2023 (Distill
  step-by-step, paper §3.1). Reads JSONL of `{"input"}` records,
  builds a 3-shot CoT prompt from per-task presets (`sst2`,
  `esnli`), calls a small HF instruction-tuned model
  (`google/flan-t5-base` default), parses the LLM's
  `Answer: <class>` response, writes
  `(input, label, rationale)` JSONL consumable by
  `RationaleDataset`. 19 new tests.
- `scripts/smoke/run_extract_rationales_modal.py` — Modal L4
  smoke.  Verified: 8/8 SST-2 triples in 2.6 s.

### Iteration 2 — multi-task T5 trainer

- New `MultiTaskT5Trainer` in
  `zynthe.core.training.rationale_trainer` (re-exported from
  `zynthe.training`). Implements the paper's two-forward-pass
  multi-task recipe (paper §3.2) under task prefixes
  `[label] · input` and `[rationale] · input`. 12 new tests.
- Train-step path teacher-forces the decoder (inference path uses
  no_grad + explicit decoder_input_ids).
- Inline-eval path keeps logits and target shapes aligned via
  min-length truncation.

### Iteration 3 — end-to-end recipe on SST-2

- New `scripts/run_distill_step_by_step.py` runs the full
  pipeline end-to-end (extract → train → eval). 5 new tests
  pin the recipe end-to-end via a subprocess (so they're fully
  isolated from HF Hub access).
- Offline fallback T5 + offline fallback LLM callable:
  when the recipe can't reach HF Hub (Modal sandbox, local CI),
  it uses a self-contained `_OfflineTinySeq2Seq` + `_OfflineStubTokenizer`
  + a stub extractor LLM.  The recipe still runs end-to-end and
  produces a valid JSON.
- Tests run the recipe as a subprocess (`subprocess.run`), so
  they're offline by construction.

### Version bump + CHANGELOG

- `pyproject.toml` and `src/zynthe/__init__.py` version bumped
  **0.2.6 → 0.3.0**.
- `CHANGELOG.md` has a 0.3.0 entry describing the deliverables.

### Final test count

**316 unit tests pass clean on Modal L4** under `-W error::UserWarning`.

## Pending (next phases, not Phase 6)

These are the items deferred to a future plan-talk (not Phase 6):

- Real GLUE accuracy number (vs. fp32 baseline, on Modal A100).
- Coverage floor lift to ≥ 25 %.
- `scripts/modal/` archive decision / removal.
- MkDocs deploy to GitHub Pages (workflow already wired).
- `zynthe_report.md` archival to `docs/reports/`.
