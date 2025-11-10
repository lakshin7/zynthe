# Repository Structure and Component Guide

This document provides an end-to-end description of the `knowledge-distillation-toolkit` repository to help new contributors and reviewers understand how the project is organised, how the data and models flow through the system, and where to extend core functionality.

## Top-Level Layout

| Path | Purpose |
| --- | --- |
| `app/` | Typer-based CLI entrypoints for running distillation, quantisation, and evaluation routines. |
| `configs/` | YAML configuration presets (e.g. `default.yaml`, device-specific variants) consumed by the configuration manager. |
| `core/` | Core library code including configuration utilities, distillers, model wrappers, quantisation helpers, and shared utilities. |
| `data/` | Data processing utilities, JSONL datasets, and augmentation scripts for IMDB sentiment distillation experiments. |
| `docs/` | Project documentation, design notes, quick-start guides, reports, and publishable artefacts. |
| `evaluation/` | Metric computation, benchmarking utilities, report generation, and visualisation helpers for teacher vs. student analysis. |
| `examples/` | Minimal runnable scripts and notebooks illustrating distillation and evaluation workflows. |
| `experiments/` | Auto-generated experiment artefacts (checkpoints, logs, reports) arranged by timestamp. |
| `exports/` | Reserved for packaging and exporting distilled/quantised models. |
| `test/` | Automated tests, regression suites, and smoke checks used in CI/CD. |
| `training/` | Training orchestration modules (optimizer, scheduler, trainer) for supervised fine-tuning and distillation loops. |
| `requirements.txt` | Python dependency manifest covering training, explainability, and quantisation stacks. |

## Configuration and Experiment Management (`core/config`)

- `config_manager.py`: Central `ConfigManager` class responsible for loading default and user-specified YAML files, merging overrides, resolving runtime options (device, seeds, explainability toggles), and materialising experiment directories with captured metadata (Git SHA, environment info, resolved config snapshot).
- `__init__.py`: Exposes configuration classes for simplified imports.

## Distillation Engines (`core/distillers`)

- `base_distiller.py`: Abstract base class defining the distillation contract (hooks, forward pass, loss aggregation) shared by specialised distillers.
- `kd_hinton.py`: Implements classical knowledge distillation (KL divergence between teacher and student logits) following Hinton et al.
- `attention_transfer.py`: Aligns teacher and student attention maps via L2 loss with optional feature retrieval.
- `feature_distiller.py`: Registers forward hooks on specified layers and minimises feature-wise MSE between teacher and student representations.
- `similarity_transfer.py`: Computes cosine similarity-based losses for representation alignment.
- `multi_stage_distiller.py`: Orchestrates multi-phase distillation pipelines by instantiating a sequence of distillers governed by configuration rules, enabling mixed strategies.

## Models and Persistence (`core/models`)

| File | Role |
| --- | --- |
| `model_loader.py` | Loads teacher/student pairs from Hugging Face Hub based on configuration (handles device placement, tokenizer loading, model type inference). |
| `model_wrapper.py` | Enterprise-grade wrapper that standardises forward calls, quantisation, saving, and summarisation with logging. |
| `projection_heads.py` | Lightweight MLP projection heads for feature distillation adapters. |
| `model_saver.py` | Helper utilities for checkpointing and restoring PyTorch/HF models and tokenisers. |
| `__init__.py` | Simplified module exports. |

## Quantisation (`core/quant`)

- `ptq.py`: Post-training quantisation runners supporting dynamic quantisation with automatic fallbacks (e.g. FP16 when qint8 is unavailable) and device-aware backend selection.
- `qat.py`, `calibration.py`: Stubs/placeholders for future quantisation-aware training and calibration pipelines; designed for extension with enterprise requirements.
- `__init__.py`: Package init for quant modules.

## Utilities and Explainability (`core/utils`, `core/explainability`)

- `core/utils/`: Contains shared helpers (logging, metrics, file IO). Inspect individual modules when extending.
- `core/explainability/`: Houses explainability integrations (e.g. SHAP, LIME) referenced in configuration defaults and evaluation flows.

## CLI Application Layer (`app/`)

- `main.py`: Typer CLI exposing commands such as `distill` and `quantize`. Dynamically imports distiller/quantisation modules, wires configuration, and handles runtime errors with informative logging.
- `__init__.py`: Package initialiser for CLI exposure.

## Data Pipeline (`data/`)

- Contains raw/sampled IMDB datasets (`*.jsonl`), augmentation scripts (`augmentations.py`), preprocessing utilities (`preprocess.py`), and dataset loaders (`dataloaders.py`).
- Designed for extension to custom datasets; configuration references determine train/validation split paths.

## Evaluation Suite (`evaluation/`)

- `metrics.py`: Aggregates accuracy, F1, precision, recall, ROC AUC, confusion matrices, and produces JSON/plot outputs.
- `evaluator.py`: Runs batched evaluation loops with device-aware fallbacks and optional explainability hooks.
- `benchmark.py` & `tasks/`: Provide benchmarking harnesses and task definitions for cross-task evaluation.
- `model_comparison.py`, `visualizer.py`: Generate side-by-side teacher vs. student analyses, charts, and visual reports.
- `report.py`: Emits Markdown/HTML experiment summaries with tables and embedded plots.

## Examples and Notebooks (`examples/`)

- `minimal_distill.py`: Executable example demonstrating config loading, model initialisation, single-batch Hinton KD loss computation, and model export.
- `minimal_eval.py`: Blueprint for running evaluation on distilled models.
- `compare_teacher_student.py`: Scripted comparison workflow for analysing teacher/student performance.
- `Teacher_vs_Student_Comparison.ipynb`: Notebook offering interactive visual inspection.

## Testing (`test/`)

- Contains smoke tests (`test_distill.py`, `test_quant.py`, etc.) ensuring minimal workflows execute without runtime errors.
- `compare_models.py`, `run_comparison.py`: Integration tests to validate reporting and comparison pipelines.

## Training Loop (`training/`)

- `trainer.py`: High-level trainer orchestrating epochs, gradient accumulation, early stopping, and logging.
- `optimizer.py` & `scheduler.py`: Encapsulate optimiser and LR scheduler construction from configuration.

## Documentation (`docs/`)

- Includes strategic documents (`design.md`, `msme_playbook.md`), quick start guides, comparison summaries, and this structure overview.
- New publishable artefacts (e.g., IEEE-formatted paper) should be added here.

## Workflow Summary

1. **Configuration**: Users customise YAML files under `configs/` and launch jobs via the CLI or examples.
2. **Model Loading**: `core/models` modules instantiate teacher/student/tokenizer triplets on the detected device.
3. **Distillation/Training**: `core/distillers` and `training/` coordinate distillation strategies and optimisation loops.
4. **Quantisation (Optional)**: `core/quant` modules apply PTQ/QAT and device-specific optimisations.
5. **Evaluation & Reporting**: `evaluation/` modules generate metrics, plots, and reports stored under `experiments/`.
6. **Outputs**: Artifacts (checkpoints, reports, logs) are persisted in timestamped experiment directories.

## Extensibility Guidelines

- **Adding a new distiller**: Implement a class in `core/distillers/`, inherit from `BaseDistiller`, and register via configuration.
- **Supporting new model families**: Extend `model_loader.py` and add dedicated wrappers/heads in `core/models/`.
- **Expanding datasets**: Add preprocessing scripts to `data/` and reference new JSONL paths in configs.
- **Enhancing quantisation**: Fill out `quant/qAT.py` or create additional runners; expose via CLI to maintain parity.

This overview should equip contributors, reviewers, and auditors with a clear roadmap of the repository's architecture and extension points.
