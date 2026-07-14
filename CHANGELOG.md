# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-14

### Added
- **Phase 6 Iteration 1: LLM rationale extractor.** New
  `scripts/extract_rationales.py` re-implements the
  rationale-extraction step from Hsieh et al. 2023 (Distill
  step-by-step, paper §3.1). Reads JSONL of `{"input"}` records,
  builds a 3-shot CoT prompt from a per-task preset (`sst2`,
  `esnli`), calls a small HF instruction-tuned model
  (`google/flan-t5-base` default), parses the LLM's
  `Answer: <class>` response, writes
  `(input, label, rationale)` JSONL consumable by
  `RationaleDataset`. 19 new tests.
- **Phase 6 Iteration 2: multi-task T5 trainer.** New
  `MultiTaskT5Trainer` in
  `zynthe.core.training.rationale_trainer` (re-exported from
  `zynthe.training`). Implements the paper's two-forward-pass
  multi-task recipe (paper §3.2) under task prefixes
  `[label] · input` and `[rationale] · input`. Packs both views
  into a `{"label_logits", "rationale_logits"}` dict consumable
  by `RationaleDistiller.compute_loss`. 12 new tests.
- **Phase 6 Iteration 3: end-to-end recipe on SST-2.** New
  `scripts/run_distill_step_by_step.py` runs the full pipeline
  end-to-end: load SST-2 (HF datasets with a synthetic fallback),
  extract rationales, multi-task train, evaluate. 5 new tests
  pin the recipe end-to-end.
- `pyproject.toml` and `src/zynthe/__init__.py` version bumped to
  **0.3.0** (was 0.2.6).
- New docs: `docs/phase6_status.md` — per-iteration status of the
  Distill step-by-step recipe.

### Changed
- The Distill-step-by-step recipe is now a **single command**:
  `python scripts/run_distill_step_by_step.py` (no Modal required).
  Reproducible on any CPU/GPU host.

## [0.2.6] - 2026-05-12

### Added
- GitHub Actions CI workflow for linting and tests.
- GitHub Actions Docs workflow for deploying MkDocs to GitHub Pages.
- Issue templates for bug reports and feature requests.
- `balanced.yaml` preset configuration matching the Kaggle SST-2 distillation example.
- Full ONNX export support using `optimum[onnxruntime]`.
- Multi-stage distillation framework with `DistillationToolkit`.
- Support for Text, Code, Vision, and Multimodal models via `AdapterRegistry`.
- Dynamic learning rate (auto_lr) and Automatic Gradient Clipping (AGC).
- Hardware-agnostic preflight checks.
- Post-Training Quantization (PTQ) support.

### Changed
- `prepare_distillation` is re-exported from `zynthe.training`.
