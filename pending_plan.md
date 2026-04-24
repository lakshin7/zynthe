# Pending Implementation Plan

Last updated: 2026-04-21
Source baseline: implementation_plan.md

## Goal
Track implementation progress against the foundation plan and keep only actionable remaining work.

## Current Status Snapshot
- Phase 1: Mostly complete
- Phase 2: Complete
- Phase 3: Complete
- Phase 4: Mostly complete
- Phase 5: Complete
- Phase 6: Complete

## Phase Checklist

### Phase 1: Code Health And Static Analysis
- [x] Finalize typing around optional fallback symbols in distiller exports.
- [ ] Re-run lint and mypy gates as a strict baseline after all pending tasks land.

### Phase 2: Evaluation And Visualization Refactor
- [x] Make evaluator modality-aware with explicit paths for text, vision, and multimodal.
- [x] Add a report generation path returning a standardized EvaluationReport.
- [x] Move duplicated diagnostics logic into a shared helper used by evaluator and trainer.
- [x] Extend visual dashboard to conditionally render runtime, calibration, metrics, and training curves based on available data.
- [x] Implement plot_extended_metrics.
- [x] Reduce scattered plotting calls in trainer end-of-fit flow and use one consolidated dashboard/report path.

### Phase 3: Model Save And Resume Integration
- [x] Expand save_training_run bundle to include model/tokenizer, config snapshot, metrics history, checkpoint state, and evaluation report artifact.
- [x] Complete export matrix support for ONNX, TorchScript, SafeTensors, GGUF, and BitNet.
- [x] Implement resume restoration contract for model/optimizer/scheduler/scaler and metadata (epoch/step/best metric).

### Phase 4: CLI Unification
- [x] Align command behavior to planned contracts for distill/evaluate/export/compare/info.
- [x] Replace compare placeholder with actual comparison flow.
- [x] Ensure export command supports multi-format selection and output behavior.
- [ ] Keep shared loading/config/device logic in helper functions to reduce command body size.

### Phase 5: Vision And Multimodal Pipeline
- [x] Generalize image dataloader into a universal factory supporting cifar10/cifar100/stl10/imagenet/image_folder/hf datasets.
- [x] Add explicit config entries for dataset routing via data.image_dataset.
- [x] Ensure trainer/evaluator can handle vision-first batch keys in configured modality paths.
- [x] Add planned configs: configs/vision_cifar10.yaml and configs/multimodal_clip.yaml.
- [x] Add top-1/top-5 vision metrics and guard text-only logic by modality.

### Phase 6: Testing And Validation
- [x] Add CPU-safe fixtures for tiny model and mock dataloader patterns.
- [x] Add tests/test_evaluation_report.py (serialization, markdown, dashboard resilience).
- [x] Expand tests/test_model_saver.py for bundle structure and export paths.
- [x] Add vision-route coverage in tests/test_pipeline_refactor.py.
- [ ] Add integration notebooks: notebooks/test_distillation_colab.ipynb and notebooks/test_vision_colab.ipynb.

## Remaining Work
1. Run strict lint and mypy baselines after broader repo cleanup (current repo has many pre-existing style/type issues outside this change set).
2. Refactor CLI shared helpers to reduce duplication between app/main.py and app/main_new.py.
3. Add and verify the missing Colab notebook artifacts, then expand them into full text and vision end-to-end distillation runs once the runtime inputs are finalized.

## Completion Criteria
- Lint and type checks pass for touched modules.
- Unit tests pass for report, saver/export, and pipeline routing paths.
- CLI smoke checks pass for distill/evaluate/export/compare/info.
- Distillation run emits consistent model, checkpoint, and evaluation artifacts.
- Colab notebooks run end-to-end for text and vision validation.
