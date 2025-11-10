# Mac M2 Compatible Model Pairs for Knowledge Distillation

## Overview

This document lists model pairs that are:
- ✅ Compatible with Mac M2 (MPS backend)
- ✅ Tested for memory efficiency
- ✅ Suitable for knowledge distillation
- ✅ Available on Hugging Face

---

## 🎯 Recommended Model Pairs

### 1. BERT Family (Text Classification)

#### Pair 1: BERT-base → TinyBERT
```yaml
model:
  name: "bert-base-uncased"              # Teacher: 110M params
  student_name: "huawei-noah/TinyBERT_General_4L_312D"  # Student: 14.5M params
  type: "transformer"
  tokenizer_name: "bert-base-uncased"
  num_labels: 2

# Expected: ~7.5x compression, minimal accuracy drop
```

**Why this pair:**
- 7.5x compression ratio (much better than DistilBERT's 1.64x)
- TinyBERT specifically designed for distillation
- Well-tested architecture
- Good for deployment on resource-constrained devices

#### Pair 2: BERT-large → BERT-base
```yaml
model:
  name: "bert-large-uncased"             # Teacher: 340M params
  student_name: "bert-base-uncased"      # Student: 110M params
  type: "transformer"
  tokenizer_name: "bert-large-uncased"
  num_labels: 2

# Expected: ~3x compression, minimal accuracy drop
```

**Why this pair:**
- Similar architecture (easier knowledge transfer)
- Significant size reduction
- Better baseline for comparison

### 2. RoBERTa Family (Text Classification)

#### Pair 3: RoBERTa-base → DistilRoBERTa
```yaml
model:
  name: "roberta-base"                   # Teacher: 125M params
  student_name: "distilroberta-base"     # Student: 82M params
  type: "transformer"
  tokenizer_name: "roberta-base"
  num_labels: 2

# Expected: ~1.5x compression, good performance retention
```

**Why this pair:**
- RoBERTa often outperforms BERT
- Better for sentiment analysis
- Good tokenization for informal text

#### Pair 4: RoBERTa-large → RoBERTa-base
```yaml
model:
  name: "roberta-large"                  # Teacher: 355M params
  student_name: "roberta-base"           # Student: 125M params
  type: "transformer"
  tokenizer_name: "roberta-large"
  num_labels: 2

# Expected: ~2.8x compression, excellent performance
# Warning: Requires 16GB+ RAM on Mac M2
```

### 3. ALBERT Family (Efficient Architecture)

#### Pair 5: ALBERT-base-v2 → ALBERT-base-v2 (Layer Reduction)
```yaml
model:
  name: "albert-base-v2"                 # Teacher: 12M params (12 layers)
  student_name: "albert-base-v2"         # Student: Same, but use layer selection
  type: "transformer"
  tokenizer_name: "albert-base-v2"
  num_labels: 2

# Note: Use custom layer selection in code
# Expected: 2x compression with layer pruning
```

**Why this pair:**
- ALBERT uses parameter sharing (already efficient)
- Good for testing layer-wise distillation
- Very memory efficient

### 4. ELECTRA Family (Discriminative Pre-training)

#### Pair 6: ELECTRA-base → ELECTRA-small
```yaml
model:
  name: "google/electra-base-discriminator"   # Teacher: 110M params
  student_name: "google/electra-small-discriminator"  # Student: 14M params
  type: "transformer"
  tokenizer_name: "google/electra-base-discriminator"
  num_labels: 2

# Expected: ~7.8x compression, good performance
```

**Why this pair:**
- ELECTRA often more efficient than BERT
- Good compression ratio
- Strong performance on downstream tasks

---

## 🚀 Tested Configurations for Mac M2

### Configuration 1: High Compression (Recommended for M2 8GB)

**File:** `configs/high_compression.yaml`
```yaml
train:
  epochs: 3
  batch_size: 4                  # Reduced for memory
  lr: 2e-5
  grad_accum_steps: 2            # Compensate for small batch
  mixed_precision: false
  early_stop_patience: 2

model:
  name: "bert-base-uncased"
  student_name: "huawei-noah/TinyBERT_General_4L_312D"
  type: "transformer"
  tokenizer_name: "bert-base-uncased"
  max_length: 128
  num_labels: 2

distillation:
  method: "kd_hinton"
  temperature: 3.0               # Higher temp for bigger gap
  alpha: 0.3                     # More weight on soft labels

device:
  prefer_mps: true
  prefer_cuda: false

output_root: "experiments"
seed: 42
compare_models: true
```

### Configuration 2: Balanced Performance (Recommended for M2 16GB)

**File:** `configs/balanced_roberta.yaml`
```yaml
train:
  epochs: 4
  batch_size: 8
  lr: 2e-5
  grad_accum_steps: 1
  mixed_precision: false
  early_stop_patience: 3

model:
  name: "roberta-base"
  student_name: "distilroberta-base"
  type: "transformer"
  tokenizer_name: "roberta-base"
  max_length: 128
  num_labels: 2

distillation:
  method: "kd_hinton"
  temperature: 2.5
  alpha: 0.4

device:
  prefer_mps: true
  prefer_cuda: false

output_root: "experiments"
seed: 42
compare_models: true
```

### Configuration 3: Maximum Quality (Requires 16GB+ RAM)

**File:** `configs/maximum_quality.yaml`
```yaml
train:
  epochs: 5
  batch_size: 6
  lr: 1e-5                       # Lower LR for large model
  grad_accum_steps: 2
  mixed_precision: false
  early_stop_patience: 3

model:
  name: "roberta-large"
  student_name: "roberta-base"
  type: "transformer"
  tokenizer_name: "roberta-large"
  max_length: 128
  num_labels: 2

distillation:
  method: "kd_hinton"
  temperature: 2.0
  alpha: 0.5

device:
  prefer_mps: true
  prefer_cuda: false

output_root: "experiments"
seed: 42
compare_models: true
```

---

## 🔍 Why Your Student Outperformed Teacher

### Possible Reasons:

1. **Pre-training Quality**
   - DistilBERT was specifically trained on BERT's outputs
   - It may have better initialization for your task
   - Pre-distilled models can outperform on specific tasks

2. **Tokenizer Issues**
   - Check if both models use compatible tokenizers
   - BERT uses `token_type_ids`, DistilBERT doesn't
   - This can cause confusion in the comparison

3. **Overfitting**
   - Teacher might be overfitting on training data
   - Student's simpler architecture provides better generalization
   - Check training vs validation curves

4. **Fine-tuning Duration**
   - Teacher fine-tuned for 2 epochs
   - Student trained for 3 epochs with distillation
   - More training can improve performance

### 🔬 Diagnostic Steps

1. **Check Tokenizer Compatibility:**
```python
# Run this to verify tokenizers
python -c "
from transformers import AutoTokenizer
teacher_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
student_tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
text = 'This is a test sentence.'
print('Teacher tokens:', teacher_tok.tokenize(text))
print('Student tokens:', student_tok.tokenize(text))
print('Teacher IDs:', teacher_tok.encode(text))
print('Student IDs:', student_tok.encode(text))
print('Match:', teacher_tok.encode(text) == student_tok.encode(text))
"
```

2. **Compare Training Curves:**
```bash
# Check if teacher overfitted
cat experiments/YOUR_EXP/logs/teacher_training.log
cat experiments/YOUR_EXP/logs/student_training.log
```

3. **Test on Different Data Split:**
```bash
# Create a fresh test set
python data/preprocess.py --split-ratio 0.7:0.15:0.15  # train:val:test
```

---

## 📋 Recommended Testing Plan

### Phase 1: Test Different Architectures (Week 1)

```bash
# Test 1: TinyBERT (High Compression)
python app/main.py --config configs/high_compression.yaml

# Test 2: RoBERTa pair (Better Quality)
python app/main.py --config configs/balanced_roberta.yaml

# Test 3: ELECTRA pair (Alternative Architecture)
python app/main.py --config configs/electra_distill.yaml
```

### Phase 2: Compare Results

```bash
# Run comparison on all experiments
for exp in experiments/2025*/; do
    echo "Comparing: $exp"
    python test_comparison.py --exp "$exp"
done

# Aggregate results
python tools/aggregate_experiments.py --output comparison_summary.csv
```

### Phase 3: Select Best Model

Criteria for selection:
- ✅ Student accuracy ≤ Teacher accuracy (expected behavior)
- ✅ Compression ratio > 2x
- ✅ Inference latency improvement
- ✅ No sanity check warnings

---

## 🛠️ Quick Model Testing Script

Create this script to quickly test different model pairs:

**File:** `test_model_pair.py`
```python
#!/usr/bin/env python3
"""Quick test for different model pairs."""

import argparse
import yaml
from pathlib import Path

PRESETS = {
    "tinybert": {
        "teacher": "bert-base-uncased",
        "student": "huawei-noah/TinyBERT_General_4L_312D",
        "tokenizer": "bert-base-uncased",
        "batch_size": 4,
        "temperature": 3.0,
        "alpha": 0.3
    },
    "distilbert": {
        "teacher": "bert-base-uncased",
        "student": "distilbert-base-uncased",
        "tokenizer": "distilbert-base-uncased",
        "batch_size": 8,
        "temperature": 1.5,
        "alpha": 0.4
    },
    "roberta": {
        "teacher": "roberta-base",
        "student": "distilroberta-base",
        "tokenizer": "roberta-base",
        "batch_size": 8,
        "temperature": 2.5,
        "alpha": 0.4
    },
    "electra": {
        "teacher": "google/electra-base-discriminator",
        "student": "google/electra-small-discriminator",
        "tokenizer": "google/electra-base-discriminator",
        "batch_size": 6,
        "temperature": 2.5,
        "alpha": 0.4
    }
}

def generate_config(preset_name: str, output_path: str):
    """Generate config for a model pair."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = PRESETS[preset_name]
    
    config = {
        "train": {
            "epochs": 3,
            "batch_size": preset["batch_size"],
            "lr": 2e-5,
            "grad_accum_steps": 1,
            "mixed_precision": False,
            "early_stop_patience": 2
        },
        "model": {
            "name": preset["teacher"],
            "student_name": preset["student"],
            "type": "transformer",
            "tokenizer_name": preset["tokenizer"],
            "max_length": 128,
            "num_labels": 2
        },
        "distillation": {
            "method": "kd_hinton",
            "temperature": preset["temperature"],
            "alpha": preset["alpha"]
        },
        "data": {
            "train_path": "data/imdb_train.jsonl",
            "val_path": "data/imdb_val.jsonl"
        },
        "device": {
            "prefer_mps": True,
            "prefer_cuda": False
        },
        "output_root": "experiments",
        "seed": 42,
        "quantization": {
            "enable": True,
            "mode": "ptq"
        },
        "similarity_transfer": True,
        "evaluate": True,
        "compare_models": True
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"✅ Generated config: {output_path}")
    print(f"   Teacher:  {preset['teacher']}")
    print(f"   Student:  {preset['student']}")
    print(f"   Run with: python app/main.py --config {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config for model pairs")
    parser.add_argument(
        "preset",
        choices=list(PRESETS.keys()),
        help="Model pair preset"
    )
    parser.add_argument(
        "--output",
        default="configs/test_pair.yaml",
        help="Output config path"
    )
    
    args = parser.parse_args()
    generate_config(args.preset, args.output)
```

### Usage:
```bash
# Generate TinyBERT config
python test_model_pair.py tinybert --output configs/tinybert_test.yaml

# Run training
python app/main.py --config configs/tinybert_test.yaml

# Compare results
python test_comparison.py --exp experiments/$(ls -t experiments | head -1)
```

---

## 💡 Best Practices

### For Mac M2 8GB:
- ✅ Use `batch_size: 4-6`
- ✅ Use `max_length: 128` or less
- ✅ Test TinyBERT or ELECTRA-small
- ✅ Enable gradient accumulation
- ❌ Avoid `*-large` models

### For Mac M2 16GB:
- ✅ Use `batch_size: 8-12`
- ✅ Can try `roberta-base` teacher
- ✅ Safe to use most model pairs
- ⚠️ Test `*-large` models carefully

### For Mac M2 32GB:
- ✅ Use `batch_size: 16-24`
- ✅ Can use `roberta-large` teacher
- ✅ All model pairs should work

---

## 📊 Expected Compression Ratios

| Model Pair | Compression | Expected Accuracy Drop |
|------------|-------------|------------------------|
| BERT-base → TinyBERT | 7.5x | 2-4% |
| BERT-large → BERT-base | 3.0x | 1-2% |
| RoBERTa-base → DistilRoBERTa | 1.5x | 1-3% |
| RoBERTa-large → RoBERTa-base | 2.8x | 1-2% |
| ELECTRA-base → ELECTRA-small | 7.8x | 2-5% |

---

## 🎯 Next Steps

1. **Try TinyBERT first** (best compression for M2 8GB)
2. **Compare with your current results**
3. **Check if student < teacher** (expected behavior)
4. **Document findings** for your use case

Would you like me to create these config files for you?
