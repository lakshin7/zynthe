# Kaggle SST-2 Distillation Tutorial

This guide walks you through distilling a large BERT model (110M parameters) into a tiny student model (11M parameters) on the SST-2 sentiment analysis dataset. We'll use the exact configuration from `examples/kaggle_sst2_distillation.py`.

## 1. Environment Setup

If you're running this on Kaggle, make sure you have a GPU (like the T4) enabled. Install `zynthe` and `datasets`:

```python
!pip install -q zynthe datasets
```

Then, initialize your device:

```python
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
```

## 2. Load Models & Tokenizer

We will distill `textattack/bert-base-uncased-SST-2` (Teacher) into `google/bert_uncased_L-4_H-256_A-4` (Student).

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TEACHER_NAME = "textattack/bert-base-uncased-SST-2"
STUDENT_NAME = "google/bert_uncased_L-4_H-256_A-4"

tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_NAME, num_labels=2
).to(DEVICE)
teacher.eval()
teacher.config.output_hidden_states = True

student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_NAME, num_labels=2
).to(DEVICE)
student.train()
student.config.output_hidden_states = True
```

## 3. Prepare the DataLoader

Load the GLUE SST-2 dataset using HuggingFace `datasets` and prepare PyTorch DataLoaders.

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("glue", "sst2")

def tokenize_fn(examples):
    return tokenizer(
        examples["sentence"], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )

train_ds = dataset["train"].map(tokenize_fn, batched=True)
val_ds = dataset["validation"].map(tokenize_fn, batched=True)

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
```

## 4. Run Zynthe Distillation

Here's the core of the toolkit. We build a `DistillationToolkit` and create a `balanced` training plan that performs **Logit Alignment** followed by **Similarity Transfer**. 

```python
from zynthe import DistillationToolkit

toolkit = DistillationToolkit(teacher, student, device=str(DEVICE))

# Use the preset "balanced" plan, with Kaggle-specific overrides
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
                        "grad_clip": {"type": "agc", "clip_factor": 0.01},
                        "auto_lr": {
                            "enabled": True,
                            "metric": "combined",
                            "primary": "val_loss",
                            "secondary": "train_loss",
                        },
                    },
                },
                {
                    "name": "Stage 2 - Similarity Transfer",
                    "type": "similarity",
                    "epochs": 2,
                    "depends_on": [1],
                    "config": {
                        "grad_clip": {"type": "agc", "clip_factor": 0.01},
                        "similarity_transfer": {
                            "similarity_metric": "cosine",
                            "weight": 0.2,
                            "auto_layers": "last",
                            "auto_layer_count": 1,
                        },
                    },
                }
            ]
        }
    },
)

# Run the training
report = toolkit.run(
    plan=plan,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="/kaggle/working/zynthe_sst2_output",
)

print(f"Distillation complete! Best model at: {report.get('best_model_path')}")
```

Zynthe will automatically track the learning rate, optimize gradients via AGC (Automatic Gradient Clipping), and evaluate the model using the built-in quality gates.
