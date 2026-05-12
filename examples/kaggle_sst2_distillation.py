# =============================================================================
# Zynthe Distillation Test — SST-2 Sentiment (Kaggle T4)
#
# Teacher: textattack/bert-base-uncased-SST-2  (110M, ~93% acc)
# Student: google/bert_uncased_L-4_H-256_A-4   (11M,  ~50% untrained)
#
# Copy each section into a separate Kaggle notebook cell.
# =============================================================================

# ── Cell 1: Install ──────────────────────────────────────────────────────────
# !pip install -q zynthe datasets

# ── Cell 2: Imports & Setup ──────────────────────────────────────────────────
import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Cell 2b: ASCII log sanitizer (avoid Jupyter Unicode errors) ─────────────
class _AsciiLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = record.msg.encode("ascii", "ignore").decode("ascii")
        if record.args:
            record.args = tuple(
                arg.encode("ascii", "ignore").decode("ascii") if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True

_root_logger = logging.getLogger()
_root_logger.addFilter(_AsciiLogFilter())
for _handler in _root_logger.handlers:
    _handler.addFilter(_AsciiLogFilter())


# ── Cell 3: Load Models ─────────────────────────────────────────────────────
TEACHER_NAME = "textattack/bert-base-uncased-SST-2"
STUDENT_NAME = "google/bert_uncased_L-4_H-256_A-4"

tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

# Teacher: already fine-tuned on SST-2 (binary sentiment)
teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_NAME, num_labels=2
).to(DEVICE)
teacher.eval()
teacher.config.output_hidden_states = True

# Student: tiny BERT (4 layers, 256 hidden) — NOT fine-tuned
student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_NAME, num_labels=2
).to(DEVICE)
student.train()
student.config.output_hidden_states = True

t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
s_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"Teacher: {t_params:.1f}M params")
print(f"Student: {s_params:.1f}M params")
print(f"Compression: {t_params / s_params:.1f}x")

# ── Cell 4: Load SST-2 Dataset ──────────────────────────────────────────────
dataset = load_dataset("glue", "sst2")

def tokenize_fn(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

# Tokenize
train_ds = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
val_ds = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])

# Set format for PyTorch
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")

# ── Cell 5: Quick teacher accuracy check ─────────────────────────────────────
teacher.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(DEVICE)
        logits = teacher(**inputs).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
teacher_acc = 100 * correct / total
print(f"Teacher accuracy on SST-2 val: {teacher_acc:.2f}%")

# ── Cell 5b: Baseline student accuracy (pre-distill) ─────────────────────────
student.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(DEVICE)
        logits = student(**inputs).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
student_baseline_acc = 100 * correct / total
print(f"Student baseline accuracy: {student_baseline_acc:.2f}%")

# ── Cell 6: Run Zynthe Distillation ──────────────────────────────────────────
from zynthe import DistillationToolkit

toolkit = DistillationToolkit(teacher, student, device=str(DEVICE))
plan = toolkit.build_plan(
    goal="balanced",
    overrides={
        "quality_gate": {
            "stop_on_regression": True,
            "max_accuracy_drop": 0.5,
            "min_stage_accuracy": 50.0,
        },
        "distillation": {
            "stages": [
                {
                    "name": "Stage 1 - Logit Alignment",
                    "type": "kd",
                    "epochs": 3,
                    "config": {
                        "grad_clip": {
                            "type": "agc",
                            "clip_factor": 0.01,
                            "eps": 1e-3,
                            "exclude_bias_and_norm": True,
                        },
                        "auto_lr": {
                            "enabled": True,
                            "metric": "combined",
                            "primary": "val_loss",
                            "secondary": "train_loss",
                            "primary_weight": 0.7,
                            "reduce_patience": 1,
                            "reduce_factor": 0.5,
                            "increase_patience": 1,
                            "increase_factor": 1.1,
                            "plateau_threshold": 1e-4,
                            "increase_threshold": 5e-4,
                            "min_lr": 1e-6,
                            "max_lr": 2e-3,
                            "cooldown": 1,
                        },
                    },
                }
                {
                    "name": "Stage 2 - Similarity Transfer",
                    "type": "similarity",
                    "epochs": 2,
                    "depends_on": [1],
                    "config": {
                        "grad_clip": {
                            "type": "agc",
                            "clip_factor": 0.01,
                            "eps": 1e-3,
                            "exclude_bias_and_norm": True,
                        },
                        "auto_lr": {
                            "enabled": True,
                            "metric": "combined",
                            "primary": "val_loss",
                            "secondary": "train_loss",
                            "primary_weight": 0.7,
                            "reduce_patience": 1,
                            "reduce_factor": 0.5,
                            "increase_patience": 1,
                            "increase_factor": 1.1,
                            "plateau_threshold": 1e-4,
                            "increase_threshold": 5e-4,
                            "min_lr": 1e-6,
                            "max_lr": 5e-4,
                            "cooldown": 1,
                        },
                        "similarity_transfer": {
                            "similarity_metric": "cosine",
                            "weight": 0.2,
                            "kd_weight": 0.1,
                            "progressive": False,
                            "auto_layers": "last",
                            "auto_layer_count": 1,
                            "normalize": True,
                            "weight_schedule": {
                                "type": "linear",
                                "start": 0.05,
                                "end": 0.2,
                            },
                        },
                    },
                }
            ]
        }
    },
)
toolkit.preview(plan)

OUTPUT_DIR = "/kaggle/working/zynthe_sst2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

report = toolkit.run(
    plan=plan,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir=OUTPUT_DIR,
)

print("\n" + "=" * 60)
print("DISTILLATION COMPLETE")
print("=" * 60)
print(f"Best model path: {report.get('best_model_path', 'N/A')}")

# ── Cell 7: Final accuracy comparison ────────────────────────────────────────
student.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(DEVICE)
        logits = student(**inputs).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

student_acc = 100 * correct / total
print(f"\nFinal Student accuracy: {student_acc:.2f}%")
print(f"Teacher accuracy:       {teacher_acc:.2f}%")
print(f"Student baseline:       {student_baseline_acc:.2f}%")
print(f"Compression ratio:      {t_params / s_params:.1f}x")
print(f"Student params:         {s_params:.1f}M")
