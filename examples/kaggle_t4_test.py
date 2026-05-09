#!/usr/bin/env python3
"""
Zynthé v0.2.2 — Kaggle T4 GPU Test
====================================

Copy-paste this into a Kaggle notebook cell.
Supports: T4 x1 (single GPU) and T4 x2 (dual GPU).

Kaggle Setup:
  1. New Notebook → Accelerator → GPU T4 x1 (or x2)
  2. Paste this entire script into a single cell
  3. Run
"""

# ============================================================
# Cell 1: Install zynthe from TestPyPI
# ============================================================
import subprocess, sys

def install():
    """Install zynthe + deps."""
    cmds = [
        # Install from TestPyPI (fallback to real PyPI for deps)
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--index-url", "https://test.pypi.org/simple/",
         "--extra-index-url", "https://pypi.org/simple/",
         "zynthe==0.2.4"],
        # Eval extras
        [sys.executable, "-m", "pip", "install", "--quiet",
         "matplotlib", "seaborn", "scikit-learn"],
    ]
    for cmd in cmds:
        subprocess.check_call(cmd)

install()
print("✅ Installation complete")

# ============================================================
# Cell 2: Environment Detection
# ============================================================
import torch
import os

num_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "None"
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9 if num_gpus > 0 else 0

print(f"\n{'='*60}")
print(f"  ENVIRONMENT")
print(f"{'='*60}")
print(f"  GPUs:       {num_gpus}x {gpu_name}")
print(f"  VRAM:       {vram_gb:.1f} GB per GPU")
print(f"  PyTorch:    {torch.__version__}")
print(f"  CUDA:       {torch.version.cuda}")

# Auto-select config
MODE = "dual" if num_gpus >= 2 else "single"
print(f"  Mode:       {MODE} T4")
print(f"{'='*60}\n")

# ============================================================
# Cell 3: Import Verification
# ============================================================
import zynthe
print(f"zynthe v{zynthe.__version__}")

from zynthe import (
    DistillationToolkit, Evaluator, EvaluationReport,
    ModelComparator, StudentInference, PipelineBuilder,
    PreflightAnalyzer, PTQRunner, apply_ptq,
    load_models, save_model, save_checkpoint, CheckpointMetadata,
    AdapterRegistry, list_presets, get_preset,
)
from zynthe.evaluation.visualizer import (
    plot_training_curves, plot_distillation_gap,
    plot_evaluation_dashboard, plot_teacher_student_comparison,
)

print("✅ All imports verified")

# ============================================================
# Cell 4: Load Models
# ============================================================
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
)

TEACHER_NAME = "bert-base-uncased"
STUDENT_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
device = torch.device("cuda:0")

print(f"\n📥 Loading teacher: {TEACHER_NAME}")
teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_NAME, num_labels=NUM_LABELS
).to(device)

print(f"📥 Loading student: {STUDENT_NAME}")
student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_NAME, num_labels=NUM_LABELS
).to(device)

tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
s_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"\n📊 Teacher: {t_params:.1f}M params")
print(f"📊 Student: {s_params:.1f}M params")
print(f"📊 Compression: {t_params/s_params:.2f}x")

# ============================================================
# Cell 5: Prepare Data (SST-2)
# ============================================================
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("glue", "sst2")
print(f"\n📦 SST-2: {len(dataset['train'])} train, {len(dataset['validation'])} val")

def tokenize_fn(batch):
    return tokenizer(
        batch["sentence"], padding="max_length",
        truncation=True, max_length=128, return_tensors="pt",
    )

def collate_fn(batch):
    sentences = [x["sentence"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    enc = tokenizer(
        sentences, padding="max_length",
        truncation=True, max_length=128, return_tensors="pt",
    )
    # Return a plain dict of tensors (NOT BatchEncoding) so the
    # distiller's training_step isinstance(batch, dict) check works.
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }

# Auto-detect mode if not already set (in case cells run independently)
if 'MODE' not in dir():
    MODE = "dual" if torch.cuda.device_count() >= 2 else "single"

# Use subset for smoke test
TRAIN_SIZE = 2000 if MODE == "single" else 4000
VAL_SIZE = 500

train_subset = dataset["train"].select(range(min(TRAIN_SIZE, len(dataset["train"]))))
val_subset = dataset["validation"].select(range(min(VAL_SIZE, len(dataset["validation"]))))

BATCH_SIZE = 32 if MODE == "single" else 64

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"📊 Using {len(train_subset)} train / {len(val_subset)} val samples")
print(f"📊 Batch size: {BATCH_SIZE}")

# ============================================================
# Cell 6: Preflight Analysis
# ============================================================
print("\n🔍 Running preflight...")
analyzer = PreflightAnalyzer(
    teacher_model=teacher, student_model=student,
    config={
        "model": {"name": TEACHER_NAME, "student_name": STUDENT_NAME, "type": "sequence_classification"},
        "data": {"dataset_id": "glue/sst2", "train_path": "hf://glue/sst2"},
        "train": {"batch_size": BATCH_SIZE, "epochs": 2, "lr": 5e-5},
        "distillation": {"method": "kd_hinton"},
    },
)
validation = analyzer.validate_config()
print(f"  Config valid: {validation['is_valid']}")
print(f"  Warnings: {len(validation['warnings'])}")

# ============================================================
# Cell 7: Adapter Detection
# ============================================================
registry = AdapterRegistry()
teacher_adapter = registry.detect(teacher)
student_adapter = registry.detect(student)
print(f"\n🔌 Teacher adapter: {teacher_adapter}")
print(f"🔌 Student adapter: {student_adapter}")

# ============================================================
# Cell 8: DistillationToolkit — Plan & Preview
# ============================================================
if 'device' not in dir():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if 'MODE' not in dir():
    MODE = "dual" if torch.cuda.device_count() >= 2 else "single"

toolkit = DistillationToolkit(teacher, student, device=str(device))

if MODE == "dual":
    plan = toolkit.build_plan(goal="full")
else:
    plan = toolkit.build_plan(goal="balanced")

toolkit.preview(plan)

# ============================================================
# Cell 9: Training (actual distillation!)
# ============================================================
OUTPUT_DIR = "/kaggle/working/zynthe_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n🚀 Starting distillation ({MODE} T4)...")
print(f"   Output: {OUTPUT_DIR}")

# For actual training we use the quick preset to keep time short
quick_plan = toolkit.build_plan(goal="quick")
report = toolkit.run(
    plan=quick_plan,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir=OUTPUT_DIR,
)
print(f"\n✅ Distillation complete!")
print(f"   Best model: {report.get('best_model_path', 'N/A')}")

# ============================================================
# Cell 10: Evaluate Student
# ============================================================
print("\n📊 Evaluating student model...")
eval_results = toolkit.evaluate(
    val_loader, tokenizer=tokenizer,
    loss_fn=torch.nn.CrossEntropyLoss(),
)
print(f"  Loss: {eval_results.get('loss', 'N/A')}")
print(f"  Accuracy: {eval_results.get('accuracy', 'N/A')}")
print(f"  F1: {eval_results.get('f1', 'N/A')}")

# ============================================================
# Cell 11: Teacher vs Student Comparison
# ============================================================
print("\n🔬 Comparing teacher vs student...")
comparison = toolkit.compare(
    val_loader, tokenizer=tokenizer,
    save_dir=f"{OUTPUT_DIR}/comparison",
)
print(f"  Compression: {comparison['compression_ratio']:.2f}x")
print(f"  Accuracy gap: {comparison['accuracy_gap']:.4f}")
print(f"  F1 gap: {comparison['f1_gap']:.4f}")

# ============================================================
# Cell 12: Visualization
# ============================================================
print("\n📈 Generating visualizations...")

# Synthetic training curves (from report if available, otherwise dummy)
train_losses = report.get("train_losses", [2.0, 1.5, 1.2, 1.0, 0.8])
val_losses = report.get("val_losses", [2.1, 1.6, 1.3, 1.1, 0.9])
metrics_history = report.get("metrics_history", {
    "accuracy": [0.5, 0.65, 0.72, 0.78, 0.82],
    "f1": [0.48, 0.62, 0.70, 0.76, 0.80],
})

plot_training_curves(train_losses, val_losses, metrics_history,
                     f"{OUTPUT_DIR}/training_curves.png")

teacher_m = comparison.get("teacher", {})
student_m = comparison.get("student", {})
if teacher_m and student_m:
    plot_distillation_gap(
        {k: teacher_m[k] for k in ["accuracy", "f1", "precision", "recall"] if k in teacher_m},
        {k: student_m[k] for k in ["accuracy", "f1", "precision", "recall"] if k in student_m},
        f"{OUTPUT_DIR}/distillation_gap.png",
    )

print(f"  Plots saved to {OUTPUT_DIR}/")

# ============================================================
# Cell 13: Inference Demo
# ============================================================
print("\n🔮 Inference demo...")
infer = StudentInference(student, tokenizer, device=device)
test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible film, waste of time.",
    "An okay experience, nothing special.",
    "Best performance I've ever seen in cinema!",
]
results = infer.predict(test_texts)
for r in results:
    sentiment = "positive" if r["label_id"] == 1 else "negative"
    print(f"  [{sentiment:>8} {r['prob']:.2%}] {r['text'][:50]}")

# ============================================================
# Cell 14: Post-Training Quantization
# ============================================================
print("\n⚡ Applying PTQ...")
student_cpu = student.cpu()
quantized = apply_ptq(student_cpu, device="cpu", mode="dynamic")

def count_bytes(m):
    return sum(t.numel() * t.element_size() for t in m.state_dict().values() if torch.is_tensor(t))

orig_mb = count_bytes(student_cpu) / 1e6
quant_mb = count_bytes(quantized) / 1e6
print(f"  Original:  {orig_mb:.1f} MB")
print(f"  Quantized: {quant_mb:.1f} MB")
print(f"  Reduction: {(1 - quant_mb/orig_mb)*100:.1f}%")

# Move student back to GPU for any further use
student.to(device)

# ============================================================
# Cell 15: Save Everything
# ============================================================
print("\n💾 Saving artifacts...")

# Evaluation report
eval_report = EvaluationReport(
    loss=eval_results.get("loss"),
    metrics=eval_results.get("metrics", {}),
    modality="text",
    model_name=STUDENT_NAME,
    task_type="classification",
    metadata={"gpu": gpu_name, "mode": MODE, "num_gpus": num_gpus},
)
eval_report.save_json(f"{OUTPUT_DIR}/evaluation_report.json")
eval_report.save_markdown(f"{OUTPUT_DIR}/evaluation_report.md")

# Save model
save_model(student, f"{OUTPUT_DIR}/student_model", tokenizer=tokenizer)
print(f"  Model saved to {OUTPUT_DIR}/student_model/")

# Save checkpoint
optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
meta = CheckpointMetadata(
    epoch=2, metrics=eval_results.get("metrics", {}),
)
save_checkpoint(student, optimizer, path=f"{OUTPUT_DIR}/checkpoint.pt", metadata=meta)
print(f"  Checkpoint saved")

# ============================================================
# Cell 16: Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  KAGGLE T4 TEST COMPLETE ({MODE.upper()})")
print(f"{'='*60}")
print(f"  GPU:           {num_gpus}x {gpu_name}")
print(f"  Teacher:       {TEACHER_NAME} ({t_params:.1f}M)")
print(f"  Student:       {STUDENT_NAME} ({s_params:.1f}M)")
print(f"  Compression:   {comparison['compression_ratio']:.2f}x")
print(f"  Accuracy gap:  {comparison['accuracy_gap']:.4f}")
print(f"  Quantized:     {orig_mb:.1f}MB → {quant_mb:.1f}MB ({(1-quant_mb/orig_mb)*100:.0f}% smaller)")
print(f"  Output:        {OUTPUT_DIR}/")
print(f"{'='*60}")

# List all output artifacts
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.isdir(path):
        print(f"  📁 {f}/")
    else:
        size = os.path.getsize(path)
        unit = "KB" if size < 1e6 else "MB"
        val = size / 1024 if size < 1e6 else size / 1e6
        print(f"  📄 {f} ({val:.1f} {unit})")
