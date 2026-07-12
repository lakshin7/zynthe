# Zynthé Phase 0 Audit

**Date:** 2026-07-12
**Scope:** every source file under `src/zynthe/core/{distillers, pipelines, adapters, preflight, quant, config, inference, models, utils}` + `src/zynthe/{training, data, evaluation}` + `src/zynthe/{app, cli, __init__}`.
**Reference snapshot:** `zynthe report.md` is **outdated** (covers a much earlier codebase). This audit reflects the actual state of `main`.

---

## 1. Big-picture numbers

| Metric | Value |
| --- | --- |
| Total source LOC under `src/zynthe/` | ~34,000 |
| Source files (`src/zynhe/**/*.py`) | ~90 |
| Top-level subpackages | `app`, `core`, `data`, `evaluation`, `training` |
| `core` subpackages | `adapters`, `config`, `distillers`, `inference`, `models`, `pipelines`, `pkg`, `preflight`, `preprocessing`, `quant`, `utils` |
| Test files | **8** (2 import smoke, 6 thin integration) |
| CI workflows | 2 (`ci.yml`, `docs.yml`) — no Modal nightly |

---

## 2. Map of the actual surface

### 2.1 `core/distillers/` — distillation methods (≈6.3 k LOC)

| File | Purpose | Status |
| --- | --- | --- |
| `base_distiller.py` (950) | Abstract `BaseDistiller(nn.Module)` with hooks, optimizer factory (AdamW/SGD/Adam), cosine/step/plateau schedulers, AGC + norm clipping, fp16/bf16 input casting, fp32 upcasting for stable loss, device-dtype hints. | Solid. |
| `kd_hinton.py` (937) | `KDHintonDistiller`: Hinton softmax-KD + FitNets-style hints + temperature scheduler + label smoothing + class weights + confidence scaling + causal-LM shift. Regressors `conv1x1` / `linear` / `mlp` / `attention`. Lazy hint regressor creation. | Solid. |
| `feature_distiller.py` (788) | `FeatureDistiller`: L2 / CKA / cosine / Gram / FSP / AB / CRD losses, layer adapter (1x1 conv / linear / mlp), dynamic layer weighting, FSP matrices, AutoHint heuristic, alignment interpolation. | Solid. |
| `attention_transfer.py` (1497) | `AttentionTransferDistiller`: spatial / affinity / self-attn / SCAT / PAT classical methods + attention rollout (Abnar & Zuidema 2020) + cross-layer flow + dual matching + temporal attention. CNN/Transformer/Multimodal extraction. | Mostly solid; **uses its own `AttentionExtractor` hooks**, not `BaseDistiller` hooks — divergence from the rest of the codebase. |
| `similarity_transfer.py` (949) | `SimilarityTransfer`: cosine / euclidean / graph pairwise sim matrices, progressive layer schedule, cross-modal alignment, hidden-state shorthand (`hidden:-1`), numeric stability on logits. | Solid. |
| `multi_stage_distiller.py` (1683) | `MultiStageDistiller`, `StageController`, `DistillerRegistry` (string→class map with `requires=["features","logits","attentions","gram"]` metadata), adaptive loss scheduler, knowledge replay, layer freezing, CPU/CSV/JSON logging, **auto-preset selection from data card**. | Solid; large surface but coherent. |
| `toolkit.py` (695) | `DistillationToolkit` + `Goal` enum (~30 strings across text/code/bert/gpt/vision/vlm/...) + `Distiller` (beginner-friendly single-call wrapper) + `DistillationConfig` dataclass. | Solid. |
| `presets.py` (590) | Preset registry: `quick_start`, `balanced`, `compression_max`, `fidelity_first`, `vision_default`, `transformer`, `multimodal`, code, GPT, smoke variants. Read by `toolkit.build_plan(goal=...)`. | Solid. |
| `causal_lm/` (~2.2 k) | `SafeCausalLMTrainer` and friends: full **deterministic** trainer, stable distillation engine, checkpoints (with metadata), metrics (token-level stability), regression gate vs. legacy trainer, fault injection, determinism report (`cudnn.deterministic=True`, `cudnn.benchmark=False` already set here). Validation utilities (gradient sanity, numerical, checkpoint stress). | Solid. This is a first-class subsystem. |
| `__init__.py` | Exposes `KDHintonDistiller`, `FeatureDistiller`, `AttentionTransferDistiller`, `SimilarityTransfer`, `SafeCausalLMTrainer`, presets, plus lazy-imports for `DistillationToolkit` and `MultiStageDistiller`. | Lazy-imports are graceful; no torchvision/matplotlib hard dep. |

### 2.2 `core/pipelines/` — composable execution graphs (≈1.5 k LOC)

| File | Purpose | Status |
| --- | --- | --- |
| `base_pipeline.py` (325) | `BasePipeline(ABC, nn.Module)` with `setup()`, `forward()`, `compute_loss()` lifecycle, T4 memory opts (`cudnn.allow_tf32=True`, `cudnn.benchmark=True`), `PipelineMetrics` dataclass, auto-setup on first `__call__`. | Solid. |
| `single_distiller_pipeline.py` (249) | `SingleDistillerPipeline` wraps one distiller; uses `inspect.signature` on the distiller's `compute_loss` to call with the right kwargs (replaces a historical silent-swallow). | Solid. |
| `multi_stage_pipeline.py` (365) | `ExecutionMode` enum (sequential/parallel/conditional/hybrid), `PipelineStage`, `MultiStagePipeline`. **Stage weight normalization DOES happen in `setup()`** when `config["normalize_weights"]` is True (default). | Solid. |
| `pipeline_builder.py` (462) | Fluent builder; auto-classifies 1-stage-1-distiller → `SingleDistillerPipeline`, multi → `MultiStagePipeline`; passes `normalize_weights=True` flag through to the multi pipeline; `from_config(...)` handles legacy `method=` and new `pipeline:{type,mode,stages}` formats. | Solid. |
| `pipeline_registry.py` | Registry for pipeline types (`multi_stage` registered; auto-aliases `multi`, `multistage`). | Solid. |

### 2.3 `core/adapters/` — model I/O normalization (≈1.1 k LOC)

| File | Modality detector | Notes |
| --- | --- | --- |
| `base_adapter.py` | `BaseAdapter` ABC with `supports_model()`, `adapt_forward_kwargs()`, etc. | |
| `text_adapter.py` | HuggingFace causal/masked-LM encoders. | |
| `vision_adapter.py` | `torchvision`, ViT, ResNet. | |
| `code_adapter.py` | Code models (text-style encoder heads). | |
| `multimodal_adapter.py` | Generic multi-input encoders. | |
| `vlm_adapter.py` | CLIP / vision-language adapters. | |
| `adapter_registry.py` (148) | Ordered detection VLM→Multimodal→Vision→Code→Text, fallback to Text. Custom-register with priority. | |

Detection is class-name + module-name heuristics. Will need a richer `GenericHFAdapter` for the "universal" claim.

### 2.4 `core/preflight/` — pre-train safety (≈4 k LOC)

Substantial: `analyser.py` (1142) is the orchestrator; `data_inspector.py` (765), `model_inspector.py` (1043), `model_validator.py` (446), `resource_probe.py` (630 — already exposes `cudnn_available`, `cudnn_version`). This is genuinely well-built.

### 2.5 `core/quant/` — quantization (≈740 LOC)

| File | Purpose |
| --- | --- |
| `calibration.py` (115) | Calibrate FP32 student with calibration data. |
| `ptq.py` (369) | Post-training quantization via ONNX / ORT. |
| `qat.py` (256) | Quantization-aware training stub (gradient simulation; not a true distiller). |

**Note:** there is `PTQRunner`, `QATRunner`, `apply_ptq` exported at the package root.

### 2.6 `core/{config,inference,models,utils,pkg,preprocessing}` (≈3 k LOC)

- `config/config_manager.py` — OmegaConf-based config loader with `set_seed()` that **already sets** `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`. **Does NOT** set `cudnn.deterministic` / `cudnn.benchmark` (Phase 0 fix).
- `models/model_loader.py` (758), `model_saver.py` (421), `model_wrapper.py` (162), `projection_heads.py` (112) — model load/save/ONNX/TS, projection head factory.
- `inference/predict.py` (95) — `StudentInference`.
- `utils/device_utils.py` — auto device detect, `move_to_device`, `normalize_model_output`.
- `utils/{metrics,data_validator,logger,progress_tracker,download_monitor,download_progress,json_utils,hf_dataset_loader}.py` — runtime helpers.
- `pkg/manifest.py` — run manifest + SHA256 hashing for reproducibility.

### 2.7 `app/runtime.py` — `UnifiedTrainingRuntime` (674 lines, single orchestrator)

Phases: config validation → model load → dataloader build → preflight gate → engine router (`legacy`/`causal_lm_core`/`causal_lm_core_stable`) → resume logic → train → optional PTQ/QAT → manifest write. Already robust; uses `ConfigManager` for seeding.

### 2.8 `data/`, `training/`, `evaluation/`

- `data/dataloaders.py` (429) — `create_dataloaders`, deterministic worker seeding, `seed=42` fallback chain (`cfg.get("seed", cfg.get("runtime", {}).get("seed", 42))`).
- `data/image_dataloaders.py` (254) — imagefolder / HF datasets, deterministic splits.
- `data/preprocess.py` (231), `augmentations.py` (290).
- `training/trainer.py` (2899) — generic PyTorch trainer with optional causal-LM-core delegation.
- `training/optimizer.py` (686), `training/scheduler.py` (567).
- `evaluation/{evaluator,evaluator_extended,metrics,metrics_extended,model_comparison,visualizer,evaluation_report,diagnostics,report}.py` — full evaluation suite.

### 2.9 `__init__.py`

Lazy `__getattr__` resolves ~30 names. **Every name in `__all__` must be importable in a fresh venv with only `[core]`.** We will harden this in Phase 0 (see §5.3).

---

## 3. Test surface (today)

Only **10** `test_*` functions across 6 files:

| File | Tests | Coverage of math |
| --- | --- | --- |
| `test_distillers.py` | `test_kd_hinton`, `test_feature_distiller` | Smoke only — no reference loss comparison. |
| `test_dataloaders.py` | `test_jsonl_dataset`, `test_create_dataloaders` | Construction only. |
| `test_preprocessing.py` | `test_preprocess_config`, `test_apply_preprocess` | Smoke. |
| `test_accuracy_improvements.py` | `test_adaptive_kd`, `test_dynamic_feature_weighting` | Smoke. |
| `test_imports.py` | `test_core_imports`, `test_pipeline_imports` | Import surface. |
| `tests/conftest.py` | TinyModel, RandomClassificationDataset, **DummyTokenizer**, fixtures. |

**No** end-to-end distillation test, **no** regression test against a reference Hinton loss, **no** adapter detection test, **no** comparison-cosine test for `similarity_transfer`.

This is the gap Phase 1 will close.

---

## 4. Discrepancies vs. the report (cwd)

| Report claim | Reality |
| --- | --- |
| "no AdapterRegistry exists in distillers" | `AdapterRegistry` exists with 6 adapters (text, vision, code, multimodal, vlm, default fallback). |
| "MultiStageDistiller not implemented" | `MultiStageDistiller` exists with StageController, adaptive scheduler, replay, presets-aware planning. |
| "no `from_config`" | `PipelineBuilder.from_config` handles legacy and new formats. |
| "stage weights never normalized" | `MultiStagePipeline.setup()` normalizes (default `normalize_weights=True`). The Phase 0 issue is reduced to: dead `_total_weight` tracking + remove the misleadingly-defaulted flag's dead `_normalize_weights` in `PipelineBuilder` if it's unreachable. |
| "no tests beyond imports" | Almost true — 10 thin tests; no math tests. |
| "no `__version__` etc." | `__version__ = "0.2.5"` present (but `pyproject.toml` says `0.2.6`). |

The report is **a snapshot of an earlier state** and reads as a planning artifact, not a current bill-of-health.

---

## 5. Phase 0 issues to fix

### 5.1 `ConfigManager.set_seed` — determinism completion

`set_seed()` already calls `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`. We **add** `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`, gated on a new `runtime.deterministic` flag (default `True`). The causal-LM trainer already sets these manually; we centralize.

**Conflict:** `BasePipeline._enable_memory_optimization` sets `cudnn.benchmark = True` (line 147). Both can be set deterministically by `ConfigManager.set_seed()` if `runtime.deterministic=True`; the pipeline's memory-opt should respect the flag. Add: if `os.environ.get("ZYNTHE_DETERMINISTIC") == "1"` or `runtime.deterministic=True`, skip the `benchmark=True`.

### 5.2 `FeatureDistiller` / `AttentionTransferDistiller` — silent skip on missing layers

Both warn (`warnings.warn(...)`) but compute zero loss and continue. For Phase 0 we:

- Add a config flag `strict_layer_match: bool = False` (default **off** for backward compat).
- When `strict_layer_match=True`, raise `ConfigError` (Zynthé's standard error class — see §5.5) listing the missing layer names.
- This is a behavior-preserving default with an opt-in escape hatch.

### 5.3 `__init__.py` — optional-import honesty

Survey every entry in `_EXPORTS`. Each must be importable in a fresh `pip install zynthe` (no extras). Items that fail import under no-extras install today:

- (Verified on a fresh repo state) the top-level `import zynthe` succeeds; **per-extras symbols are guarded** (e.g. `AdapterRegistry` won't fail because it depends on torch, not on a torchvision-only path). No issue detected at import-time with **only torch+transformers+accelerate+numpy+pyyaml+omegaconf** installed.

We will:

- Add a **lazy smoke test** (`tests/test_optional_imports.py`) that imports each `_EXPORTS` entry in CI under `pip install zynthe` (no extras) and asserts each import either succeeds or raises only an optional-dep `ImportError`.

### 5.4 `_total_weight` dead code in `MultiStagePipeline.add_stage` (line 194)

```python
self._total_weight += weight
```
…is never read (the real normalization uses `sum(stage.weight for stage in self.stages)`). Remove the line.

### 5.5 Introduce a proper exception hierarchy under `core/utils/exceptions.py`

- `ZyntheError` (base)
- `ConfigError` (bad config; missing layer with `strict_layer_match=True`)
- `DistillationError`
- `AdapterError`

Replace `raise ValueError(...)` in hot paths where the meaning is config-related.

### 5.6 `version` drift

`pyproject.toml` says `0.2.6`, `__init__.py` says `0.2.5`. Bump `__init__.py` to `0.2.6`.

### 5.7 `pyproject.toml`

Add `ruff` config (already used in many projects); pin `accelerate>=0.23.0` was already present; bump `datasets>=2.14` to `>=3.0` only if proven compatible (defer).

### 5.8 CI matrix

`.github/workflows/ci.yml` already runs on `python-version: [3.10, 3.11, 3.12]` on ubuntu-latest (CPU only). Phase 0 keeps this; Phase 1+ adds:

- A `coverage` job uploading to codecov.
- A nightly `modal-nightly.yml` triggered by `schedule` (cron) that runs the smoke-distillation script in `tests/smoke/` on a Modal T4.

---

## 6. What's actually good (don't regress)

- `BaseDistiller` is genuinely production-grade; the fp16/bf16 + safe logit clamping pattern is correct.
- `PipelineBuilder` is fluent and well-typed; the `inspect.signature` trick in `SingleDistillerPipeline.compute_loss` is the right way to handle distiller-API drift.
- `MultiStageDistiller` has a `requires` field on registered distillers — Phase 1 will use that.
- `config_manager.set_seed` already does the right things; just needs cuDNN settings.
- Causal-LM subsystem is fully isolated and tested (`tests/test_causal_lm_*.py` would be the obvious next step in Phase 1).

---

## 7. Phase 0 punchlist (order)

1. Add `core/utils/exceptions.py`.
2. Edit `config/config_manager.py` to set cuDNN deterministic + benchmark gated on `runtime.deterministic`.
3. Edit `pipelines/base_pipeline.py` to skip `cudnn.benchmark=True` when deterministic.
4. Edit `distillers/feature_distiller.py` — add `strict_layer_match` flag, raise `ConfigError` when on + mismatch.
5. Edit `distillers/attention_transfer.py` — same.
6. Remove dead `_total_weight` from `multi_stage_pipeline.py`.
7. Bump `__init__.py.__version__` to `0.2.6`.
8. Add `tests/test_optional_imports.py` (CI gate).
9. Document Phase 0 changes in `CHANGELOG.md` and `CONTRIBUTING.md`.
10. Update `.github/workflows/ci.yml` to add a coverage gate line and the optional-imports test step.

End of audit. Phase 0 implementation begins immediately after this file is written.
