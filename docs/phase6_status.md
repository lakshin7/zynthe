# Phase 6 — Status

This document is updated after each Phase 6 iteration.  Phase 6
implements the end-to-end Distill step-by-step recipe (paper §3.1
+ §3.2) on real GLUE-style data — closing the v1.0 claim.

## Done

### Iteration 1 — LLM rationale extractor

- New `scripts/extract_rationales.py` — re-implements the
  rationale-extraction step from
  `google-research/distilling-step-by-step` (paper §3.1).  Reads
  JSONL of `{"input"}` records, builds a 3-shot CoT prompt from a
  per-task preset (`sst2`, `esnli`), calls a small HF
  instruction-tuned model (`google/flan-t5-base` default), parses
  the LLM's `Answer: <class>` response, writes
  `(input, label, rationale)` JSONL consumable by
  `RationaleDataset`.
- Per-task presets: `SST2_PRESET` (positive/negative),
  `ESNLI_PRESET` (entailment/neutral/contradiction).  Few-shot
  exemplars are inline.
- The LLM call is decoupled from the extractor function
  (`llm_callable` parameter) so tests pass a deterministic stub —
  no model load required for the test suite.
- New `scripts/smoke/run_extract_rationales_modal.py` — Modal L4
  wrapper.  Verified: 8/8 SST-2 triples extracted in 2.6 s.
- New `tests/test_extract_rationales.py` (19 tests) — pins
  prompt-construction, label-parsing, rationale-parsing, malformed
  output handling, max_records behaviour, CLI roundtrip.

## Pending (aligned with plan)

- [ ] Iteration 2: multi-task T5 trainer + tests
- [ ] Iteration 3: SST-2 end-to-end benchmark
- [ ] Iteration 4: 0.3.0 version bump + CHANGELOG

## Upcoming

- Iteration 2 next: the multi-task T5 trainer that runs the
  paper's `[label] · input` + `[rationale] · input` two-forward-pass
  recipe and feeds the dict-output contract into
  `RationaleDistiller.compute_loss`.
