#!/usr/bin/env python3
"""
Zynthé v0.2.4 — Comprehensive Feature Demo
=============================================

Exercises every public API symbol with dry-run-safe operations.
Works on CPU-only machines (e.g. Dell Latitude 7490).

Usage:
    pip install -e ".[eval]"
    python examples/full_demo.py
"""
from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

# Results tracker
PASSED: list[str] = []
FAILED: list[str] = []


def section(name: str):
    """Decorator that runs a test section and tracks pass/fail."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            try:
                fn()
                PASSED.append(name)
                print(f"  ✅ {name} — PASSED")
            except Exception as exc:
                FAILED.append(name)
                print(f"  ❌ {name} — FAILED: {exc}")
                traceback.print_exc()
        return wrapper
    return decorator


# =========================================================================
# Section 1: Import Verification
# =========================================================================
@section("1. Import Verification")
def test_imports():
    import zynthe
    print(f"  zynthe v{zynthe.__version__}")

    from zynthe import (
        # Primary
        DistillationToolkit,
        # Distillers
        KDHintonDistiller, AttentionTransferDistiller,
        FeatureDistiller, SimilarityTransfer, SafeCausalLMTrainer,
        # Presets
        list_presets, describe_preset, get_preset,
        # Adapters
        AdapterRegistry,
        # Config
        ConfigManager,
        # Models
        ModelBundle, ModelLoader, ModelWrapper,
        get_device, load_models, model_summary,
        save_model, load_model, save_checkpoint, load_checkpoint,
        export_onnx, export_torchscript, CheckpointMetadata,
        ProjectionHead, ProjectionHeadFactory,
        # Evaluation
        Evaluator, DualEvaluator, CurriculumEvaluator,
        EvaluationReport, ModelComparator,
        # Inference
        StudentInference,
        # Pipelines
        PipelineBuilder, PipelineRegistry, get_registry,
        # Preflight
        PreflightAnalyzer, run_preflight_check,
        # Quantization
        PTQRunner, QATRunner, apply_ptq,
        # Runtime
        UnifiedTrainingRuntime, RuntimeOptions, RuntimeResult,
    )

    # Sub-package imports
    from zynthe.core.adapters import (
        TextModelAdapter, CodeModelAdapter, VisionModelAdapter,
        MultimodalModelAdapter, VLMModelAdapter,
    )
    from zynthe.data import create_dataloaders, JsonlDataset, TextAugmenter
    from zynthe.evaluation.visualizer import (
        plot_training_curves, plot_teacher_student_comparison,
        plot_distillation_gap, plot_evaluation_dashboard,
        plot_metric_grid, plot_calibration_curve, plot_runtime_profile,
    )
    print(f"  All {len(zynthe.__all__)} public symbols imported successfully")


# =========================================================================
# Section 2: Presets
# =========================================================================
@section("2. Presets")
def test_presets():
    from zynthe import list_presets, describe_preset, get_preset

    names = list_presets()
    print(f"  Available presets: {names}")
    assert len(names) >= 5, f"Expected >=5 presets, got {len(names)}"

    for name in names:
        desc = describe_preset(name)
        cfg = get_preset(name)
        stages = cfg.get("distillation", {}).get("stages", [])
        print(f"    {name}: {len(stages)} stage(s) — {desc[:60]}")


# =========================================================================
# Section 3: Adapter Registry
# =========================================================================
@section("3. Adapter Detection")
def test_adapters():
    from zynthe import AdapterRegistry
    from zynthe.core.adapters import (
        TextModelAdapter, CodeModelAdapter, VisionModelAdapter,
        MultimodalModelAdapter, VLMModelAdapter,
    )

    registry = AdapterRegistry()
    available = registry.list_available()
    print(f"  Modalities: {available}")
    assert len(available) >= 5

    # Explicit get
    for name in ["text", "code", "vision", "multimodal", "vlm"]:
        adapter = registry.get(name)
        print(f"    get('{name}') → {adapter.__class__.__name__}")


# =========================================================================
# Section 4: DistillationToolkit (dry run)
# =========================================================================
@section("4. DistillationToolkit")
def test_toolkit():
    import torch
    import torch.nn as nn
    from zynthe import DistillationToolkit

    # Tiny dummy models
    teacher = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    student = nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 2))

    toolkit = DistillationToolkit(teacher, student, device="cpu")
    print(f"  Device: {toolkit._resolve_device()}")

    # Build plans
    plan_quick = toolkit.build_plan(goal="quick")
    n_q = len(plan_quick.get("distillation", {}).get("stages", []))
    print(f"  quick plan: {n_q} stage(s)")

    plan_balanced = toolkit.build_plan(goal="balanced")
    n_b = len(plan_balanced.get("distillation", {}).get("stages", []))
    print(f"  balanced plan: {n_b} stage(s)")

    plan_full = toolkit.build_plan(goal="full")
    n_f = len(plan_full.get("distillation", {}).get("stages", []))
    print(f"  full plan: {n_f} stage(s)")

    # Preview
    toolkit.preview(plan_balanced)

    # Dry run
    dry = toolkit.run(plan_quick, dry_run=True)
    assert isinstance(dry, dict), "Dry run should return plan dict"
    print(f"  dry_run=True returned plan with keys: {list(dry.keys())}")

    # Save plan
    out_path = Path("experiments/_test_plan.json")
    toolkit.save_plan(plan_balanced, out_path)
    assert out_path.exists()
    out_path.unlink()
    print(f"  save_plan → JSON verified")


# =========================================================================
# Section 5: EvaluationReport
# =========================================================================
@section("5. EvaluationReport")
def test_eval_report():
    from zynthe import EvaluationReport

    report = EvaluationReport(
        loss=0.42,
        metrics={"accuracy": 0.91, "f1": 0.89, "precision": 0.90, "recall": 0.88},
        diagnostics={"warnings": []},
        modality="text",
        model_name="test-student",
        task_type="classification",
        metadata={"num_samples": 128},
    )
    print(f"  Report: {report.model_name}, loss={report.loss}, acc={report.metrics['accuracy']}")

    # JSON round-trip
    json_path = Path("experiments/_test_report.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_json(json_path)
    loaded = EvaluationReport.load_json(json_path)
    assert loaded.loss == report.loss
    json_path.unlink()
    print(f"  JSON round-trip verified")

    # Markdown export
    md_path = Path("experiments/_test_report.md")
    report.save_markdown(md_path)
    assert md_path.exists()
    md_path.unlink()
    print(f"  Markdown export verified")


# =========================================================================
# Section 6: Visualization (synthetic data)
# =========================================================================
@section("6. Visualization")
def test_visualization():
    from zynthe.evaluation.visualizer import (
        plot_training_curves, plot_distillation_gap, plot_metric_grid,
    )

    out_dir = Path("experiments/_test_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    train_losses = [2.1, 1.8, 1.5, 1.3, 1.1]
    val_losses = [2.2, 1.9, 1.6, 1.5, 1.4]
    metrics = {
        "accuracy": [0.5, 0.6, 0.7, 0.75, 0.8],
        "f1": [0.48, 0.58, 0.68, 0.73, 0.78],
    }
    plot_training_curves(train_losses, val_losses, metrics, str(out_dir / "training_curves.png"))
    assert (out_dir / "training_curves.png").exists()

    # Distillation gap
    teacher_m = {"accuracy": 0.92, "f1": 0.90, "precision": 0.91}
    student_m = {"accuracy": 0.85, "f1": 0.83, "precision": 0.84}
    plot_distillation_gap(teacher_m, student_m, str(out_dir / "gap.png"))
    assert (out_dir / "gap.png").exists()

    # Metric grid
    plot_metric_grid(metrics, str(out_dir / "grid.png"))
    assert (out_dir / "grid.png").exists()

    # Cleanup
    import shutil
    shutil.rmtree(out_dir)
    print(f"  3 plot types generated and verified")


# =========================================================================
# Section 7: StudentInference (CPU)
# =========================================================================
@section("7. StudentInference")
def test_inference():
    import torch
    import torch.nn as nn
    from zynthe import StudentInference

    # HF-style model that accepts **kwargs and returns .logits
    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, input_ids, attention_mask=None, **kwargs):
            # input_ids is (batch, 32) from MockTokenizer — matches Linear(32, 16)
            logits = self.net(input_ids.float())  # (batch, 2)
            return type('Out', (), {'logits': logits})()

    model = TinyClassifier()

    # Minimal tokenizer mock
    class MockTokenizer:
        def __call__(self, texts, **kwargs):
            n = len(texts)
            return {
                "input_ids": torch.randint(0, 100, (n, 32)),
                "attention_mask": torch.ones(n, 32, dtype=torch.long),
            }

    infer = StudentInference(model, MockTokenizer(), device=torch.device("cpu"))
    results = infer.predict(["Hello world", "Test input"], batch_size=2)
    assert len(results) == 2
    assert "label_id" in results[0]
    assert "prob" in results[0]
    assert "probs" in results[0]
    print(f"  Predicted {len(results)} samples")
    print(f"    Sample: label_id={results[0]['label_id']}, prob={results[0]['prob']:.4f}")


# =========================================================================
# Section 8: PipelineBuilder
# =========================================================================
@section("8. PipelineBuilder")
def test_pipeline_builder():
    from zynthe import PipelineBuilder

    builder = PipelineBuilder()
    builder = (
        builder
        .add_stage("logit", weight=0.7)
        .add_distiller("kd_hinton", temperature=4.0, alpha=0.8)
        .add_stage("features", weight=0.3)
        .add_distiller("feature")
        .with_mode("sequential")
        .with_name("TestPipeline")
    )
    print(f"  Builder: {repr(builder)}")
    print(f"  Pipeline configured (build requires actual models, skipping)")


# =========================================================================
# Section 9: Quantization (PTQ)
# =========================================================================
@section("9. Quantization (PTQ)")
def test_ptq():
    import torch
    import torch.nn as nn
    from zynthe import apply_ptq

    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    model.eval()

    quantized = apply_ptq(model, device="cpu", mode="dynamic")

    # Compare sizes
    def param_bytes(m):
        return sum(t.numel() * t.element_size() for t in m.state_dict().values() if torch.is_tensor(t))

    orig_kb = param_bytes(model) / 1024
    quant_kb = param_bytes(quantized) / 1024
    print(f"  Original:  {orig_kb:.1f} KB")
    print(f"  Quantized: {quant_kb:.1f} KB")
    print(f"  Reduction: {(1 - quant_kb/orig_kb)*100:.1f}%")


# =========================================================================
# Section 10: Model Save/Load
# =========================================================================
@section("10. Model Save/Load")
def test_model_save_load():
    import torch
    import torch.nn as nn
    from zynthe import save_model, save_checkpoint, load_checkpoint, CheckpointMetadata

    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    save_dir = Path("experiments/_test_save")

    # Save model
    save_model(model, str(save_dir / "model"))
    assert (save_dir / "model").exists()
    print(f"  save_model → OK")

    # Save checkpoint with metadata
    optimizer = torch.optim.Adam(model.parameters())
    meta = CheckpointMetadata(epoch=5, metrics={"accuracy": 0.85})
    save_checkpoint(
        model, optimizer, path=str(save_dir / "ckpt.pt"),
        metadata=meta,
    )
    assert (save_dir / "ckpt.pt").exists()
    print(f"  save_checkpoint → OK")

    # Load checkpoint
    ckpt_data, ckpt_meta = load_checkpoint(model, optimizer, path=str(save_dir / "ckpt.pt"))
    print(f"  load_checkpoint → epoch={ckpt_meta.epoch}, metrics={ckpt_meta.metrics}")

    # Cleanup
    import shutil
    shutil.rmtree(save_dir)


# =========================================================================
# Section 11: Preflight Analysis
# =========================================================================
@section("11. Preflight Analysis")
def test_preflight():
    import torch.nn as nn
    from zynthe import PreflightAnalyzer

    teacher = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    student = nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 2))

    config = {
        "model": {"name": "test-teacher", "student_name": "test-student", "type": "classifier"},
        "data": {"train_path": "dummy.jsonl"},
        "train": {"batch_size": 8, "epochs": 2},
        "distillation": {"method": "kd_hinton"},
    }

    analyzer = PreflightAnalyzer(
        teacher_model=teacher,
        student_model=student,
        config=config,
    )

    # Config validation only (doesn't need data files)
    validation = analyzer.validate_config()
    print(f"  Config valid: {validation['is_valid']}")
    print(f"  Errors: {len(validation['errors'])}")
    print(f"  Warnings: {len(validation['warnings'])}")
    print(f"  Info: {len(validation['info'])}")


# =========================================================================
# Section 12: ConfigManager
# =========================================================================
@section("12. ConfigManager")
def test_config_manager():
    from zynthe import ConfigManager

    # Load a bundled config
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        cm = ConfigManager(config_path=str(config_path))
        cfg = cm.resolved_config
        print(f"  Loaded config: {config_path.name}")
        print(f"    model.name = {cfg.get('model', {}).get('name')}")
        print(f"    distillation.method = {cfg.get('distillation', {}).get('method')}")
    else:
        print(f"  Config not found at {config_path}, skipping")


# =========================================================================
# SUMMARY
# =========================================================================
def main():
    print("\n" + "=" * 60)
    print("  ZYNTHÉ COMPREHENSIVE FEATURE DEMO")
    print("=" * 60)

    tests = [
        test_imports,
        test_presets,
        test_adapters,
        test_toolkit,
        test_eval_report,
        test_visualization,
        test_inference,
        test_pipeline_builder,
        test_ptq,
        test_model_save_load,
        test_preflight,
        test_config_manager,
    ]

    for test in tests:
        test()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  ✅ Passed: {len(PASSED)}/{len(PASSED)+len(FAILED)}")
    for name in PASSED:
        print(f"     ✅ {name}")

    if FAILED:
        print(f"\n  ❌ Failed: {len(FAILED)}/{len(PASSED)+len(FAILED)}")
        for name in FAILED:
            print(f"     ❌ {name}")

    print()
    if not FAILED:
        print("  🎉 ALL FEATURES VERIFIED!")
    else:
        print(f"  ⚠️  {len(FAILED)} feature(s) need attention")
    print("=" * 60)

    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
