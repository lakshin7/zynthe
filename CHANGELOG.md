# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `zynthe.core.utils.exceptions` — `ZyntheError`, `ConfigError`, `DistillationError`, `AdapterError`, `PreflightError`, `QuantizationError`, `RegistryError`. Library exceptions no longer leak generic `ValueError` / `RuntimeError`.
- `ConfigError` raised by `FeatureDistiller` and `AttentionTransferDistiller` when `strict_layer_match=true` and configured layer names do not exist on the models. Default behavior (warn + skip) is preserved.
- `runtime.deterministic` config flag (env override: `ZYNTHE_DETERMINISTIC`). When `True`, `ConfigManager.set_seed()` pins `torch.backends.cudnn.deterministic=True` and disables `cudnn.benchmark`, while still auto-detecting device.
- `tests/test_optional_imports.py` — CI gate that walks every name in `zynthe.__all__` and asserts each one is importable in a fresh `pip install zynthe` install (no extras).

### Changed
- `BasePipeline._enable_memory_optimization()` now respects the deterministic flag: when `cudnn.deterministic=True` we keep `cudnn.benchmark=False` instead of forcing autotune on.
- `MultiStagePipeline` no longer maintains the dead `_total_weight` accumulator (the normalization routine recomputes the sum from `self.stages` instead, as documented in the function header).
- `__version__` bumped from `0.2.5` to `0.2.6` to match `pyproject.toml`.

### Docs
- `docs/audit.md` — Phase 0 audit of every source file under `src/zynthe/`, with discrepancies vs. `zynthe report.md`, the Phase 0 punchlist, and a written description of the gap between reported and actual state.

## [0.2.6] - 2026-05-12
### Added
- GitHub Actions CI workflow for linting and tests.
- GitHub Actions Docs workflow for deploying MkDocs to GitHub Pages.
- Issue templates for bug reports and feature requests.
- `balanced.yaml` preset configuration matching the Kaggle SST-2 distillation example.
- Full ONNX export support using `optimum[onnxruntime]`.

## [0.2.5] - 2026-05-12
### Added
- Multi-stage distillation framework with `DistillationToolkit`.
- Support for Text, Code, Vision, and Multimodal models via `AdapterRegistry`.
- Dynamic learning rate (auto_lr) and Automatic Gradient Clipping (AGC).
- Hardware-agnostic preflight checks.
- Post-Training Quantization (PTQ) support.
