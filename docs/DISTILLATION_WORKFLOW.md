# Knowledge Distillation Workflow - Complete Phase Guide

> **Reference Document**: This workflow is the canonical guide for the 9-phase distillation process.
> All code implementations must follow this sequence.

---

## 🔄 PHASE 0: Environment & Configuration Setup

**🧠 Objective**: Create a reproducible, environment-aware configuration system.

**✅ Steps**:

1. **Environment Check**
   - Detects device (MPS, CUDA, CPU)
   - Logs PyTorch version, hardware info, Git commit
   - Tools: `core/config/config_manager.py`

2. **Load Config File** (`configs/default.yaml`)
   - Contains training parameters, model names, data paths
   - Quantization and explainability flags
   - Tools: `ConfigManager.load_config()`

3. **Create Experiment Folder**
   - Timestamped directory under `experiments/`
   - Subfolders: `logs/`, `checkpoints/`, `tensorboard/`, `snapshots/`, `student_model/`
   - Tools: `ConfigManager.create_experiment_dir()`

4. **Seed Setup**
   - Sets deterministic RNG for reproducibility
   - Tools: `torch.manual_seed()`, `random.seed()`, `numpy.random.seed()`

**📦 Output**: 
- `resolved_config.yaml` and preflight metadata snapshot stored
- Experiment directory created with all subfolders

---

## 🔍 PHASE 1: Preflight Analysis & Model Loading

**🧠 Objective**: Validate configuration and load models with compatibility checks.

**✅ Steps**:

### 1.1 Config Validation (NEW - CRITICAL)
```python
from core.preflight.analyser import PreflightAnalyzer

# VALIDATE CONFIG BEFORE MODEL LOADING
config_validator = PreflightAnalyzer(
    teacher_model=None,  # Not loaded yet
    student_model=None,  # Not loaded yet
    dataset=None,        # Not loaded yet
    config=cfg_manager.resolved_config
)

# Validate config structure
validation_report = config_validator.validate_config()

if not validation_report['is_valid']:
    print("❌ Config validation failed:")
    for error in validation_report['errors']:
        print(f"  • {error}")
    exit(1)
```

**Config validation checks**:
- ✅ `model.name` (teacher) is specified
- ✅ `model.student_name` is specified  
- ✅ `data.train_path` exists and is accessible
- ✅ `data.val_path` exists and is accessible
- ✅ `distillation.method` is valid
- ✅ Device preferences are valid
- ✅ Batch size is reasonable for available memory

### 1.2 Teacher & Student Loading
```python
from core.models.model_loader import load_models

teacher, student, tokenizer = load_models(cfg_manager, device)
```

- Loaded via `core/models/model_loader.py` using Hugging Face
- Device-placed and wrapped using `ModelWrapper`
- **Error handling**: Missing model names, network errors, incompatible architectures

### 1.3 Tokenizer Initialization
- Each model loads its native tokenizer (ensures text token compatibility)
- Padding token added if missing

### 1.4 Preflight Analyzer (Model Compatibility)
```python
from core.preflight.analyser import run_preflight_check

report = run_preflight_check(
    teacher_model=teacher,
    student_model=student,
    dataset=train_dataset,
    config=cfg_manager.resolved_config,
    save_report=True,
    output_dir=experiment_dir / "preflight"
)

if not report['can_proceed']:
    print("❌ Preflight checks failed:")
    for blocker in report['blockers']:
        print(f"  • {blocker}")
    exit(1)
```

**Preflight checks**:
- ✅ Detects compatible layers (hidden states, attention heads, dimensions)
- ✅ Logs correlation between teacher and student layers
- ✅ Suggests best distillation modes: Logit, Feature, Attention, or Hybrid
- ✅ Validates model architectures are compatible
- ✅ Checks memory requirements vs available resources

**📦 Output**: 
- Preflight readiness and layer mapping report
- `preflight_report.json` saved to experiment directory
- Model summary logged

---

## 📊 PHASE 2: Dataset Preparation

**🧠 Objective**: Prepare data for efficient training and evaluation.

**✅ Steps**:

1. **Dataset Loading** (e.g., IMDB Reviews)
   - JSONL → Tokenized batches (train/val) via `core/data/preprocess.py`
   - Tools: `load_and_preprocess_data()`

2. **DataLoader Creation**
   - Batching handled with collate functions (pads, truncates, tokenizes)
   - Tools: `create_dataloaders()`
   - Optimizations: `num_workers`, `pin_memory` from preflight

3. **Augmentation** (Optional)
   - For image or multimodal setups, basic augmentations applied here
   - Tools: `core/data/augmentations.py`

4. **Data Validation**
   - Preflight `DataInspector` validates:
     - ✅ Sample structure matches model expectations
     - ✅ Label distribution is balanced
     - ✅ No missing values or corrupt samples
     - ✅ Batch sizes fit in memory

**📦 Output**: 
- `train_loader`, `val_loader` objects ready for the trainer

---

## 🎯 PHASE 3: Distillation Engine Initialization

**🧠 Objective**: Initialize the correct distiller type (Logit, Feature, Attention, or Multi-Stage).

**✅ Steps**:

1. **BaseDistiller** (Core Framework)
   - Abstracts student–teacher forward pass
   - Provides unified loss aggregation, device management, and hook registry
   - Tools: `core/distillers/base_distiller.py`

2. **Specific Distiller** (e.g., Hinton KD, Attention Transfer)
   - Inherits from `BaseDistiller`
   - Defines `compute_loss()` with relevant sub-losses (KLDiv, MSE, CosineSim)
   - Tools: `core/distillers/kd_hinton.py`, `core/distillers/attention_distiller.py`

3. **LossComposer**
   - Aggregates weighted sub-losses for complex distillation modes
   - E.g., Logit + Feature + Attention
   - Tools: `core/distillers/loss_composer.py`

4. **Multi-Stage Distiller**
   - Orchestrates multiple distillation strategies sequentially
   - Tools: `core/distillers/multi_stage_distiller.py`

**📦 Output**: 
- Ready-to-train distillation pipeline
- Loss components configured

---

## 🏋️ PHASE 4: Training Loop (Distillation Phase)

**🧠 Objective**: Train the student under teacher supervision.

**✅ Steps**:

1. **Forward Pass**
   - Teacher (frozen) → generates logits/features/attentions
   - Student → computes its own outputs

2. **Loss Calculation**
   - `compute_loss()` fuses:
     - **Logit KD Loss**: Soft targets from teacher (KLDiv)
     - **Feature Loss**: L2 between feature maps
     - **Attention Loss**: Cosine/L2 between attention matrices
     - **Hybrid/Custom Loss**: Config-driven combinations

3. **Backward & Optimize**
   - Gradients from student only (teacher frozen)
   - Optimizer & LR Scheduler steps handled in `training/trainer.py`

4. **Logging & Early Stopping**
   - Track metrics each epoch: Loss, Accuracy, F1, etc.
   - Stop if validation loss plateaus
   - Tools: `training/trainer.py`, TensorBoard

5. **Checkpointing**
   - Save best model based on validation loss
   - Save optimizer state for resuming

**📦 Output**: 
- Trained student model with best weights checkpoint
- Training history (losses, metrics per epoch)

---

## 📈 PHASE 5: Evaluation & Metrics Generation

**🧠 Objective**: Validate model performance and compare teacher vs student.

**✅ Steps**:

1. **Evaluation Run**
   - Uses `evaluation/evaluator.py` to run both models
   - Computes metrics via `evaluation/metrics.py`

2. **Metrics Collected**:
   - Accuracy, F1, Precision, Recall
   - Per-class confusion matrix
   - Precision/Recall/F1 per class
   - ROC/AUC (if applicable)

3. **Teacher–Student Comparison**
   - Both models evaluated side by side
   - Reports delta improvements/degradations
   - Tools: `evaluation/model_comparison.py`

4. **Visualization** (`visualizer.py`)
   - Confusion matrix
   - Training/validation curves
   - Attention map overlays (if enabled)
   - Layer-wise activation similarity heatmap

**📦 Output**: 
- Metrics JSON + Plots in experiment folder
- `evaluation_report.json`

---

## 🗜️ PHASE 6: Quantization Stage (Optional PTQ/QAT)

**🧠 Objective**: Compress student model while preserving accuracy.

**✅ Steps**:

1. **Device-Aware PTQ Runner**
   - Detects if `torch.qint8` supported
   - Falls back to FP16 or CPU quantization on MPS
   - Tools: `core/quant/ptq.py`

2. **Modes Supported**:
   - Dynamic Quantization
   - Float16 Quantization
   - Static Quantization

3. **Evaluation Post-Quantization**
   - Computes accuracy drop, latency, model size reduction
   - Tools: `evaluation/benchmark.py`

4. **Report Generation**
   - Plots compression vs. accuracy trade-off

**📦 Output**: 
- Quantized model saved to `/student_model_quantized/`
- Quantization report

---

## 🔬 PHASE 7: Explainability (Optional)

**🧠 Objective**: Interpret what knowledge the student retained.

**✅ Steps**:

1. **SHAP / LIME Integration**
   - Explains predictions by showing most influential tokens/features
   - Tools: `core/explainability/shap_explainer.py`

2. **Attention Heatmaps** (Grad-CAM / Rollout)
   - Visualize focus regions
   - Tools: `core/explainability/attention_viz.py`

3. **Comparison**
   - Overlay teacher and student attentions

**📦 Output**: 
- Explainability visuals stored in `experiments/.../explainability/`

---

## 📋 PHASE 8: Reporting & Export

**🧠 Objective**: Package, document, and prepare for deployment or sharing.

**✅ Steps**:

1. **Final Report** (`evaluation/report.py`)
   - Summarizes:
     - Experiment details
     - Training curves
     - Confusion matrices
     - Quantization results
     - Teacher vs Student visual comparison

2. **Model Export**
   - Saves as Hugging Face-compatible checkpoint
   - Optional ONNX/CoreML/TensorRT export for deployment
   - Tools: `core/pkg/exporter.py`

3. **Docs Update**
   - Auto-append summary to `/docs/experiment_log.md`

**📦 Output**: 
- Full experiment folder ready for presentation or publication
- `final_report.pdf` or `final_report.html`

---

## 🎨 PHASE 9: Visualization and Showcasing

**🧠 Objective**: Create visually engaging and interpretable outputs for public sharing.

**✅ Steps**:

1. Export visual plots (loss curves, heatmaps)
2. Include configuration summaries (YAML, runtime info)
3. Add snapshots of the quantization and attention overlays
4. Use in LinkedIn / Paper posts

**📦 Output**: 
- Share-ready visuals and summaries

---

## 🚀 Phase 2.0 Preview (Upcoming Additions)

| Feature | Description | Purpose |
|---------|-------------|---------|
| **LoRA / QLoRA Integration** | Adapter-based fine-tuning post-distillation | Efficient low-rank compression |
| **Multi-modal Attention Distillation** | Handle vision-language models (e.g., Qwen-VL, CLIP) | Unified MM KD |
| **Dynamic Loss Composer** | Auto-tunes weights for multi-loss KD | Smarter optimization |
| **Cloud-backed UI Dashboard** | Visual monitoring of training runs | Enterprise observability |

---

## 🔧 Integration Checklist

### Current Implementation Status:

- [x] **Phase 0**: Environment & Config Setup (`ConfigManager`)
- [ ] **Phase 1**: Config Validation (MISSING - needs implementation)
- [x] **Phase 1**: Model Loading (`load_models`)
- [ ] **Phase 1**: Preflight Analysis (EXISTS but NOT CALLED in main.py)
- [x] **Phase 2**: Dataset Preparation (`dataloaders.py`)
- [x] **Phase 3**: Distillation Engine (`multi_stage_distiller.py`)
- [x] **Phase 4**: Training Loop (`trainer.py`)
- [x] **Phase 5**: Evaluation (`evaluator.py`)
- [x] **Phase 6**: Quantization (`ptq.py`)
- [x] **Phase 7**: Explainability (`explainability/`)
- [x] **Phase 8**: Reporting (`report.py`)
- [ ] **Phase 9**: Visualization (PARTIAL - needs enhancement)

### Critical Gaps to Fix:

1. ✅ **Config Validation** (Phase 1) - Add to preflight analyzer
2. ✅ **Preflight Integration** (Phase 1) - Call analyzer in `main.py`
3. ⚠️ **Error Handling** - Missing teacher/student model validation before loading

---

## 📝 Usage Example

```python
# Complete workflow execution
from core.config.config_manager import ConfigManager
from core.preflight.analyser import run_preflight_check
from core.models.model_loader import load_models
from data.dataloaders import create_dataloaders
from core.distillers.multi_stage_distiller import MultiStageDistiller

# PHASE 0: Setup
cfg_manager = ConfigManager("configs/default.yaml")
cfg = cfg_manager.resolved_config

# PHASE 1: Preflight & Model Loading
# 1.1 Config Validation (NEW)
from core.preflight.config_validator import validate_config_structure
validation = validate_config_structure(cfg)
if not validation['is_valid']:
    raise ValueError(f"Invalid config: {validation['errors']}")

# 1.2 Load Models
teacher, student, tokenizer = load_models(cfg_manager, cfg_manager.device())

# 1.3 Preflight Analysis
train_loader, val_loader = create_dataloaders(cfg, tokenizer)
report = run_preflight_check(
    teacher_model=teacher,
    student_model=student,
    dataset=train_loader.dataset,
    config=cfg
)

if not report['can_proceed']:
    raise RuntimeError(f"Preflight failed: {report['blockers']}")

# PHASE 2-4: Distillation
distiller = MultiStageDistiller(
    teacher=teacher,
    student=student,
    config=cfg,
    train_loader=train_loader,
    val_loader=val_loader,
    device=cfg_manager.device()
)

training_report = distiller.run()

# PHASE 5: Evaluation
# (automatic in distiller.run())

# PHASE 6: Quantization
if cfg.get('quantization', {}).get('enable'):
    from core.quant.ptq import apply_ptq
    quantized_model = apply_ptq(student, mode='float16', device=cfg_manager.device())

# PHASE 8: Export
distiller.save_models(experiment_dir)
```

---

## 🎯 Key Principles

1. **Fail Fast**: Validate early (config → models → data) before expensive operations
2. **Reproducibility**: Seed everything, log everything
3. **Transparency**: Preflight reports explain what will happen
4. **Optimization**: Preflight auto-tunes batch sizes, precision, workers
5. **Flexibility**: Config-driven, not hardcoded

---

**Last Updated**: October 23, 2025  
**Maintainer**: Knowledge Distillation Toolkit Team
