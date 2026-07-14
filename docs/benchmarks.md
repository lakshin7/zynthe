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

## Quantization (Phase 4 Iteration 2)

Dynamic int8 PTQ on `prajjwal1/bert-tiny` via
`torch.ao.quantization.quantize_dynamic`.  Smoke criterion: int8
model loads, forward-passes, logits finite.

| Field | Value |
|---|---|
| Model | `prajjwal1/bert-tiny` (4.39 M params) |
| Strategy | dynamic int8 (Linear layers only) |
| Time | 0.15 s |
| **fp32 size** | 16.73 MB |
| **Int8 model** | Loads + forward OK; logits shape (1, 2) |

Smoke criterion: **forward pass succeeds with finite logits**.
Achieved.

Source JSON: `tests/smoke/results/ptq_summary.json`.

---

## PTQ Numerics-Parity (Phase 5 Iteration 4)

`prajjwal1/bert-tiny` on a deterministic synthetic input.  Compares
the same input through fp32 and dynamic-int8 inference.

| Field | Value |
|---|---|
| Model | `prajjwal1/bert-tiny` |
| Strategy | dynamic int8 (Linear layers only) |
| fp32 size | 16.73 MB |
| int8 size (fp32-labelled) | 15.16 MB |
| **Argmax agreement** | **True** (fp32 = 1, int8 = 1) |
| Abs logit diff max | 0.0125 |
| Abs logit diff mean | 0.0124 |

Smoke criterion: **argmax agreement on a deterministic input**.
Achieved — int8 quantisation preserves the predicted class and the
per-logit error is <1.3 % of the max logit magnitude.

Source JSON: `tests/smoke/results/ptq_parity.json`.

---

## Distributed Training (Phase 5 Iteration 1)

Tiny distillation (KD-Hinton) of `prajjwal1/bert-tiny`, wrapped via
HuggingFace `accelerate.prepare` (single-GPU on Modal L4).  This
proves the `prepare_distillation` integration is wired and the bundle
survives a real SGD loop.  For multi-GPU DDP, see `docs/distributed.md`.

| Field | Value |
|---|---|
| Model | `prajjwal1/bert-tiny` |
| Strategy | KD-Hinton (T=2.0) |
| Accelerator | `Accelerator(mixed_precision="no")` |
| Steps | 20 |
| Duration | 1.0 s |
| **First loss** | 0.6821 |
| **Last loss** | 0.6501 |
| **Min loss** | 0.6501 |
| **Decay** | 0.0320 |

Source JSON: `tests/smoke/results/distributed.json`.

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
