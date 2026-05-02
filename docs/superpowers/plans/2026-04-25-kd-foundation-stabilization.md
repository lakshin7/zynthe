# KD Foundation Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create git commits unless the user explicitly requests them in this session.

**Goal:** Stabilize Zynthe's practical KD foundation for text classification, causal LM, and vision classification while keeping quantization and multimodal support explicitly non-blocking.

**Architecture:** Keep the existing runtime/trainer/pipeline structure, but fix the concrete broken seams: modality-aware model loading, batch propagation through pipeline loss dispatch, canonical model persistence imports, non-interactive trainer behavior, and an always-available evaluation dashboard artifact. CLI polish is limited to a white `ZYNTHE` banner and does not change command semantics.

**Tech Stack:** Python, PyTorch, Transformers, Typer, pytest, existing Zynthe pipeline/adapters/evaluation modules.

---

### Task 1: Model Loader Modality Fixes

**Files:**
- Modify: `core/models/model_loader.py`
- Test: `tests/test_model_loader_modality.py`

- [ ] **Step 1: Write failing tests**

```python
def test_model_loader_accepts_teacher_name_alias():
    loader = ModelLoader({"model": {"teacher_name": "teacher-a", "student_name": "student-b", "type": "causal_lm"}}, device="cpu")
    spec = loader._build_spec(use_agent=False, data_samples=None)
    assert spec.teacher_name == "teacher-a"

def test_model_loader_uses_image_classification_model_and_processor(monkeypatch):
    monkeypatch.setattr(model_loader, "AutoModelForImageClassification", FakeVisionModel)
    monkeypatch.setattr(model_loader, "AutoImageProcessor", FakeProcessor)
    loader = ModelLoader({"model": {"name": "teacher-vit", "student_name": "student-vit", "type": "vision"}}, device="cpu")
    teacher, student, processor = loader.load()
    assert isinstance(teacher, FakeVisionModel)
    assert isinstance(student, FakeVisionModel)
    assert isinstance(processor, FakeProcessor)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_model_loader_modality.py -v`
Expected: FAIL because `teacher_name` is not accepted and vision uses generic `AutoModel`/`AutoTokenizer`.

- [ ] **Step 3: Implement minimal loader fix**

Use `model.name or model.teacher_name`, import `AutoModelForImageClassification`, import `AutoImageProcessor`, and route `model.type in {"vision", "image", "image_classification"}` to the image classification model class and image processor.

- [ ] **Step 4: Verify**

Run: `python -m pytest tests/test_model_loader_modality.py -v`
Expected: PASS.

---

### Task 2: Preserve Modality Batch Keys Through Pipeline Loss Dispatch

**Files:**
- Modify: `core/pipelines/single_distiller_pipeline.py`
- Modify: `training/trainer.py`
- Test: `tests/test_pipeline_refactor.py`

- [ ] **Step 1: Write failing test**

```python
def test_single_distiller_pipeline_passes_pixel_values_to_loss():
    teacher = TinyVisionModel()
    student = TinyVisionModel()
    pipeline = SingleDistillerPipeline(PixelAwareDistiller(teacher, student, device=torch.device("cpu")))
    batch = {"pixel_values": torch.randn(2, 3, 4, 4), "labels": torch.tensor([0, 1])}
    loss, _ = pipeline(batch)
    assert loss.item() >= 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_pipeline_refactor.py::TestVisionPipelineRouting::test_single_distiller_pipeline_passes_pixel_values_to_loss -v`
Expected: FAIL because `pixel_values` is not passed to the distiller compute-loss call.

- [ ] **Step 3: Implement minimal batch propagation**

Add `pixel_values`, `image`, and the original `batch` to `SingleDistillerPipeline.compute_loss()` available kwargs. In `Trainer._compute_distillation_loss()`, pass the full filtered batch instead of constructing a text-only batch.

- [ ] **Step 4: Verify**

Run: `python -m pytest tests/test_pipeline_refactor.py::TestVisionPipelineRouting::test_single_distiller_pipeline_passes_pixel_values_to_loss -v`
Expected: PASS.

---

### Task 3: Trainer Runtime Safety And Artifacts

**Files:**
- Modify: `training/trainer.py`
- Test: existing trainer/pipeline tests where practical

- [ ] **Step 1: Replace canonical persistence import**

Change `from core.models.model_loader import ModelSaver` to `from core.models.model_saver import ModelSaver`.

- [ ] **Step 2: Remove CLI-interactive validation prompt**

Replace `input("Data validation failed...")` with config-controlled behavior: `train.data_validation_fail_policy: abort|warn`, default `warn`.

- [ ] **Step 3: Save dashboard unconditionally**

When `self.final_report` exists, populate metadata with `train_losses`, `val_losses`, and `metrics_history`, then call `plot_evaluation_dashboard()` and `plot_extended_metrics()` at the end of `fit()` outside the teacher-comparison branch.

- [ ] **Step 4: Verify focused tests**

Run: `python -m pytest tests/test_evaluation_report.py tests/test_model_saver.py tests/test_pipeline_refactor.py -v`
Expected: PASS.

---

### Task 4: CLI Banner And Persistence Import

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_cli_banner.py`

- [ ] **Step 1: Write failing banner test**

```python
def test_banner_text_contains_large_zynthe():
    assert "ZYNTHE" in banner_text()
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_cli_banner.py -v`
Expected: FAIL because `banner_text()` does not exist.

- [ ] **Step 3: Implement banner**

Add `banner_text()` returning a compact ASCII `ZYNTHE` wordmark and `_print_banner()` rendering it in bold white via Rich. Call `_print_banner()` at the start of each Typer command.

- [ ] **Step 4: Use canonical model saver in export**

Change export command to import `ModelSaver` from `core.models.model_saver`.

- [ ] **Step 5: Verify**

Run: `python -m pytest tests/test_cli_banner.py -v`
Expected: PASS.

---

### Task 5: Final Verification

**Files:**
- No direct code edits.

- [ ] **Step 1: Run targeted unit tests**

Run: `python -m pytest tests/test_model_loader_modality.py tests/test_cli_banner.py tests/test_pipeline_refactor.py tests/test_evaluation_report.py tests/test_model_saver.py tests/test_optimizer_and_overrides.py -v`
Expected: PASS.

- [ ] **Step 2: Run CLI smoke check**

Run: `python app/main.py info --config configs/cpu_smoke_test.yaml`
Expected: command exits `0`, prints the white `ZYNTHE` banner, and shows configuration/model information or a controlled warning.

- [ ] **Step 3: Report remaining non-goals**

Report that GGUF/BitNet remain scaffolds, Muon remains experimental/not default, Unsloth is deferred to causal-LM LoRA workflows, and full CLIP multimodal KD is not claimed complete.

---

## Self-Review

- Spec coverage: covers loader, trainer, evaluator artifacts, CLI banner, and persistence import. Quantization and full multimodal KD are intentionally excluded from this implementation batch.
- Placeholder scan: no TBD/TODO instructions remain; deferred items are explicit non-goals.
- Type consistency: `teacher_name`, `pixel_values`, `EvaluationReport`, `ModelSaver`, and `banner_text` are used consistently across tasks.
