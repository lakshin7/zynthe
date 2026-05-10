# Zynthé Usage Guide

> Complete reference for distilling models across every supported modality.

---

## Table of Contents

- [Two APIs: Simple vs Advanced](#two-apis-simple-vs-advanced)
- [Text / NLP Models](#1-text--nlp-models)
- [Code Models](#2-code-models)
- [Vision Models (ViT)](#3-vision-models-vit)
- [Multimodal / CLIP Models](#4-multimodal--clip-models)
- [Causal LM / GPT Models](#5-causal-lm--gpt-models)
- [Goal Reference Table](#goal-reference-table)
- [Post-Distillation: Evaluate & Quantize](#post-distillation-evaluate--quantize)
- [Configuration Overrides](#configuration-overrides)

---

## Two APIs: Simple vs Advanced

Zynthé offers two ways to run distillation. Pick whichever fits your workflow.

### Simple API — `Distiller`

Best for quick experiments. Three lines to distill any model:

```python
from zynthe import Distiller

distiller = Distiller(teacher, student, goal="balanced")
distiller.fit(train_loader, val_loader, epochs=3, output_dir="./output")
```

### Advanced API — `DistillationToolkit`

Best when you need to inspect the plan, override stages, or chain evaluation:

```python
from zynthe import DistillationToolkit

toolkit = DistillationToolkit(teacher, student)
plan = toolkit.build_plan(goal="balanced")
toolkit.preview(plan)                         # inspect before running
report = toolkit.run(plan, train_loader, val_loader, output_dir="./output")
metrics = toolkit.evaluate(val_loader)        # post-training eval
comparison = toolkit.compare(val_loader)      # teacher vs student side-by-side
```

---

## 1. Text / NLP Models

**Use case:** BERT, RoBERTa, ALBERT, DeBERTa, XLNet — any encoder for classification, NER, or sentence similarity.

**Goal:** `"text"`, `"nlp"`, `"bert"`, or `"balanced"`

```python
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from zynthe import Distiller

# --- Models ---
TEACHER = "bert-base-uncased"            # 110M params
STUDENT = "distilbert-base-uncased"      # 66M params
NUM_LABELS = 2

teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=NUM_LABELS)
student = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained(TEACHER)

# --- Data ---
dataset = load_dataset("glue", "sst2")

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=64)

# --- Distill ---
distiller = Distiller(teacher, student, goal="text")
distiller.fit(train_loader, val_loader, epochs=5, output_dir="./sst2_distill")
```

### Common Teacher → Student Pairs

| Teacher | Student | Compression |
|---------|---------|-------------|
| `bert-base-uncased` (110M) | `distilbert-base-uncased` (66M) | 1.7x |
| `roberta-base` (125M) | `distilroberta-base` (82M) | 1.5x |
| `bert-large-uncased` (340M) | `bert-base-uncased` (110M) | 3.1x |
| `deberta-v3-base` (184M) | `distilbert-base-uncased` (66M) | 2.8x |

---

## 2. Code Models

**Use case:** CodeBERT, GraphCodeBERT, CodeT5 — vulnerability detection, clone detection, code classification.

**Goal:** `"code"` or `"text"` (code models use the same balanced distillation pipeline as text models)

```python
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from zynthe import Distiller

# --- Models ---
TEACHER = "microsoft/codebert-base"       # 125M params
STUDENT = "distilbert-base-uncased"       # 66M params
NUM_LABELS = 2  # e.g., vulnerable / not-vulnerable

teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=NUM_LABELS)
student = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained(TEACHER)

# --- Data ---
# Replace with your code dataset (e.g., CodeXGLUE defect detection)
dataset = load_dataset("glue", "mrpc")  # placeholder

def tokenize(batch):
    return tokenizer(
        batch["sentence1"], batch["sentence2"],
        padding="max_length", truncation=True, max_length=256,
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=32)

# --- Distill ---
distiller = Distiller(teacher, student, goal="code")
distiller.fit(train_loader, val_loader, epochs=3, output_dir="./code_distill")
```

### Common Teacher → Student Pairs

| Teacher | Student | Use Case |
|---------|---------|----------|
| `microsoft/codebert-base` (125M) | `distilbert-base-uncased` (66M) | Code classification |
| `microsoft/graphcodebert-base` (125M) | `distilbert-base-uncased` (66M) | Clone detection |
| `Salesforce/codet5-base` (220M) | `Salesforce/codet5-small` (60M) | Code summarization |

---

## 3. Vision Models (ViT)

**Use case:** ViT, DeiT, Swin Transformer — image classification, feature extraction.

**Goal:** `"vision"` or `"transformer"`

The `vision_transformer` preset uses attention transfer stages designed specifically for vision transformers, aligning patch-level attention patterns between teacher and student.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor
from zynthe import Distiller

# --- Models ---
TEACHER = "google/vit-base-patch16-224"    # 86M params
STUDENT = "google/vit-small-patch16-224"   # 22M params (or use vit-tiny)
NUM_LABELS = 10

teacher = AutoModelForImageClassification.from_pretrained(TEACHER, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
student = AutoModelForImageClassification.from_pretrained(STUDENT, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
processor = AutoImageProcessor.from_pretrained(TEACHER)

# --- Data ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": images, "labels": labels}

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

# --- Distill ---
distiller = Distiller(teacher, student, goal="vision")
distiller.fit(train_loader, val_loader, epochs=5, output_dir="./vit_distill")
```

### What the `vision_transformer` Preset Does

| Stage | Method | Epochs | Purpose |
|-------|--------|--------|---------|
| 1 | KD Warmup | 2 | Align output logits with high temperature |
| 2 | Attention Transfer | 3 | Match self-attention and relational attention maps |
| 3 | Feature Polishing | 3 | Align intermediate hidden representations |

### Common Teacher → Student Pairs

| Teacher | Student | Compression |
|---------|---------|-------------|
| `vit-base-patch16-224` (86M) | `vit-small-patch16-224` (22M) | 3.9x |
| `vit-large-patch16-224` (304M) | `vit-base-patch16-224` (86M) | 3.5x |
| `swin-base` (88M) | `swin-tiny` (28M) | 3.1x |
| `deit-base` (86M) | `deit-small` (22M) | 3.9x |

---

## 4. Multimodal / CLIP Models

**Use case:** CLIP, BLIP, SigLIP — vision-language embedding, zero-shot classification, image-text retrieval.

**Goal:** `"multimodal"`, `"clip"`, `"vlm"`, or `"vision_language"`

The `multimodal` preset skips logit-based KD (which doesn't apply to contrastive embedding models) and instead focuses on embedding alignment and relational similarity transfer.

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from zynthe import Distiller

# --- Models ---
TEACHER = "openai/clip-vit-base-patch32"     # 151M params
STUDENT = "openai/clip-vit-base-patch32"     # Use a pruned/smaller CLIP variant in production

teacher = CLIPModel.from_pretrained(TEACHER)
student = CLIPModel.from_pretrained(STUDENT)
processor = CLIPProcessor.from_pretrained(TEACHER)

# --- Data ---
# For CLIP, your dataset needs both images and text captions.
# Here's a minimal example using a custom Dataset:

class ImageTextDataset(Dataset):
    """Replace with your actual image-text pair dataset (e.g., COCO Captions, CC3M)."""
    def __init__(self, processor, num_samples=1000):
        self.processor = processor
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Replace with real image loading and captions
        from PIL import Image
        image = Image.new("RGB", (224, 224), color=(idx % 256, 128, 64))
        text = f"A sample caption for image {idx}"
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}

train_dataset = ImageTextDataset(processor, num_samples=5000)
val_dataset = ImageTextDataset(processor, num_samples=500)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# --- Distill ---
distiller = Distiller(teacher, student, goal="clip")
distiller.fit(train_loader, val_loader, epochs=5, output_dir="./clip_distill")
```

### What the `multimodal` Preset Does

| Stage | Method | Epochs | Purpose |
|-------|--------|--------|---------|
| 1 | Embedding Alignment | 3 | Align image/text embeddings using cosine + L2 |
| 2 | Relational Similarity | 3 | Preserve inter-sample relationships in embedding space |
| 3 | Attention Alignment | 2 | Transfer cross-modal attention patterns |

---

## 5. Causal LM / GPT Models

**Use case:** GPT-2, DistilGPT-2, LLaMA, Mistral — text generation, language modeling.

**Goal:** `"causal_lm"`, `"gpt"`, or `"gpt_smoke"`

```python
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from zynthe import Distiller

# --- Models ---
TEACHER = "gpt2"              # 124M params
STUDENT = "distilgpt2"        # 82M params

teacher = AutoModelForCausalLM.from_pretrained(TEACHER)
student = AutoModelForCausalLM.from_pretrained(STUDENT)
tokenizer = AutoTokenizer.from_pretrained(TEACHER)
tokenizer.pad_token = tokenizer.eos_token

# --- Data ---
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True,
        max_length=512, return_tensors="pt",
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# For causal LM, labels = input_ids (the model predicts the next token)
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }

train_loader = DataLoader(dataset["train"], batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset["validation"], batch_size=4, collate_fn=collate_fn)

# --- Distill ---
distiller = Distiller(teacher, student, goal="causal_lm")
distiller.fit(train_loader, val_loader, epochs=3, output_dir="./gpt_distill")
```

### What the `causal_lm` Preset Does

| Stage | Method | Epochs | Purpose |
|-------|--------|--------|---------|
| 1 | Token-level KD | 1 | Align next-token predictions with teacher |
| 2 | Similarity Structure | 1 | Preserve relational structure across hidden states |
| 3 | Attention Transfer | 1 | Match autoregressive attention patterns |
| 4 | Feature Polishing | 1 | Final hidden representation alignment |

### Common Teacher → Student Pairs

| Teacher | Student | Compression |
|---------|---------|-------------|
| `gpt2` (124M) | `distilgpt2` (82M) | 1.5x |
| `gpt2-medium` (355M) | `gpt2` (124M) | 2.9x |
| `gpt2-large` (774M) | `gpt2-medium` (355M) | 2.2x |

---

## Goal Reference Table

Complete mapping of goal strings to presets:

| Goal String | Preset | Best For |
|---|---|---|
| `"quick"`, `"speed"`, `"baseline"` | `quick_start` | Fast smoke tests, 2-epoch baseline |
| `"balanced"`, `"default"` | `balanced` | General-purpose, recommended default |
| `"text"`, `"nlp"`, `"bert"` | `balanced` | BERT, RoBERTa, DeBERTa, ALBERT |
| `"code"` | `balanced` | CodeBERT, GraphCodeBERT, CodeT5 |
| `"vision"`, `"transformer"` | `vision_transformer` | ViT, DeiT, Swin Transformer |
| `"multimodal"`, `"clip"`, `"vlm"` | `multimodal` | CLIP, BLIP, SigLIP |
| `"causal_lm"`, `"gpt"` | `causal_lm_smoke` | GPT-2, LLaMA, Mistral |
| `"all"`, `"full"`, `"complete"` | `all_distillers_t4` | All 4 distillers, T4-safe |
| `"smoke"` | `classification_smoke` | Fast 4-distiller classification test |
| `"compression"`, `"aggressive"` | `compression_max` | Maximum model shrinking |

---

## Post-Distillation: Evaluate & Quantize

After distillation, evaluate the student and optionally quantize for deployment:

```python
from zynthe import Distiller
from zynthe.core.quant import apply_ptq

# --- Distill ---
distiller = Distiller(teacher, student, goal="text")
distiller.fit(train_loader, val_loader, epochs=5, output_dir="./output")

# --- Evaluate ---
metrics = distiller.evaluate(val_loader)
print(f"Student Accuracy: {metrics['accuracy']:.2%}")

# --- Compare teacher vs student ---
comparison = distiller.compare(val_loader, save_dir="./comparison")

# --- Quantize for deployment (6x smaller) ---
quantized_student = apply_ptq(student, mode="dynamic")
```

---

## Configuration Overrides

Need to customize epochs, batch size, or learning rate beyond the preset defaults? Use `overrides`:

```python
from zynthe import DistillationToolkit

toolkit = DistillationToolkit(teacher, student)
plan = toolkit.build_plan(
    goal="balanced",
    overrides={
        "training": {"epochs": 20},
        "distillation": {
            "loss_schedule": {"alpha": 0.9, "beta": 0.3, "gamma": 0.1},
        },
    },
)
toolkit.preview(plan)  # verify your overrides
report = toolkit.run(plan, train_loader, val_loader)
```

Or with the `Distiller` API, just override epochs directly:

```python
from zynthe import Distiller

distiller = Distiller(teacher, student, goal="balanced")
distiller.fit(train_loader, val_loader, epochs=20)  # overrides preset epochs
```

---

## Hardware Notes

- **GPU auto-detection:** Zynthé automatically detects CUDA, MPS (Apple Silicon), or falls back to CPU.
- **Force a device:** Pass `device="cuda:1"` to `DistillationToolkit()` or `Distiller()`.
- **Mixed precision:** Presets like `all_distillers_t4` enable `mixed_precision: True` by default for T4/P100 GPUs.
- **Multi-GPU:** Wrap your models in `torch.nn.DataParallel` before passing them to Zynthé.

```python
import torch
teacher = torch.nn.DataParallel(teacher)
student = torch.nn.DataParallel(student)
distiller = Distiller(teacher, student, goal="balanced", device="cuda:0")
```
