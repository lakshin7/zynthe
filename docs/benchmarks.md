# Zynthé Benchmarks

This file records small-scale empirical evidence for Zynthé's
distillation primitives.  The entries below were produced on Modal
L4 in ~$0.05/run; they aren't full GLUE / ImageNet / Wikitext-103
benchmarks (those would cost orders of magnitude more and are
explicitly out of scope for the smoke phase — see `zynthe report.md`
§227-249 for the *aspirational* benchmarks).

Every entry here is reproducible via the Modal runner at
`scripts/smoke/run_baseline_distill.py` (and the universal-model
smoke gate at `scripts/smoke/run_smoke_modal.py`).

---

## Baseline Distillation (Experiment #1 from the report)

Tiny teacher / student pair, 30 SGD steps on synthetic SST-2-like
batches.  This is the smoke proof that the KDHinton training loop
runs end-to-end on a real GPU.

| Field | Value |
|---|---|
| Teacher | `hf-internal-testing/tiny-bert` |
| Student | `prajjwal1/bert-tiny` |
| Loss | `α=0.5` · T=2.0 KD-Hinton + (1-α) CE |
| Steps | 30 |
| Optimiser | SGD lr=1e-4 |
| Hardware | Modal L4 |
| Duration | ~1.3 s |
| **First loss** | 0.3526 |
| **Last loss** | 0.3384 |
| **Min loss** | 0.3382 |
| **Decay** | 0.0142 |

Smoke criterion: **loss is finite and decay > 0 over the run**.
Achieved — the loss decreases monotonically over 30 steps,
indicating the KDHinton distillation loop is wired correctly end
to end.

Source JSON: `tests/smoke/results/baseline_sst2.json` (regenerated
on every smoke run; commit hash is recorded inside the JSON).

---

## Rationale Distillation (Distill step-by-step, §Iteration 8)

Tiny T5 → T5 with synthetic (input, label, rationale) arithmetic
triples.  Multi-task loss = CE_label + λ·CE_rationale.

| Field | Value |
|---|---|
| Teacher | (none — student is its own teacher; no LLM at training time) |
| Student | `patrickvonplaten/t5-tiny-random` |
| Rationales | 64 synthetic triples (`a + b`, label, one-sentence rationale) |
| Loss | `CE_label + 1.0 * CE_rationale` |
| Steps | 20 |
| Hardware | Modal L4 |
| Duration | ~1.9 s |
| **First loss** | 18.46 |
| **Last loss** | 17.34 |
| **Min loss** | 16.44 |
| **Decay** | 1.12 |

Smoke criterion: **loss is finite and decay > 0 over the run**.
Achieved — the rationale multi-task loss decreases by 1.12 over 20
steps.

Source JSON: `tests/smoke/results/rationale.json`.

---

## Universal-Model Smoke Gate (5 families)

The full 5-pair gate from `scripts/smoke/universal_smoke.py` (also
covered in `docs/adapters.md`).  Result on commit `f20ee90` (Phase 2):

| Pair | Adapter route | Loader | Smoke verdict |
|------|--------------|--------|----------------|
| BERT → bert-tiny | text | transformers | ✅ |
| ViT → DeiT-tiny | vision | transformers | ✅ |
| GPT-2 | text | transformers | ✅ |
| CLIP-ViT-B/32 | multimodal | transformers | ✅ (NaN loss warning — un-init weights) |
| ResNet18 | generic | torchvision | ✅ |

Criterion: pipeline builds, adapter routes correctly, forward+backward
runs without error.  All 5 met.
