# Quick Reference: Data Quality Features

## 🚀 Quick Start

### Test All Features
```bash
python test_new_features.py
```

### Train with Auto-Validation
```bash
python app/main.py --config configs/default.yaml
# Validation runs automatically - no changes needed!
```

---

## 📚 Common Tasks

### 1. Download HuggingFace Dataset

```python
from pathlib import Path
from core.utils.hf_dataset_loader import prepare_hf_dataset

# Download and prepare in one line
paths = prepare_hf_dataset(
    'glue/sst2',              # Dataset ID
    output_dir=Path('data/sst2'),
    max_samples=2000          # Optional: limit size
)

print(paths)
# {
#   'train': Path('data/sst2/train.jsonl'),
#   'val': Path('data/sst2/validation.jsonl'),
#   'test': Path('data/sst2/test.jsonl')
# }
```

### 2. List Available Datasets

```python
from core.utils.hf_dataset_loader import HuggingFaceDatasetLoader

# See all built-in datasets
catalog = HuggingFaceDatasetLoader.list_available_datasets()

# Sentiment: imdb, sst2, sentiment140, yelp_polarity
# NLI: mnli, snli
# QA: squad, squad_v2
# Summarization: cnn_dailymail, xsum
```

### 3. Manually Validate Data

```python
from core.utils.data_validator import DataValidator
import json

# Load data
train_data = [json.loads(line) for line in open('data/train.jsonl')]
val_data = [json.loads(line) for line in open('data/val.jsonl')]

# Validate
results = DataValidator.validate_dataset_split(
    train_data, val_data,
    text_key='text',
    label_key='label'
)

if results['validation_passed']:
    print("✓ Safe to train!")
else:
    print(f"Issues: {results['errors']}")
```

### 4. Analyze Training Health

```python
from core.utils.data_validator import OverfitUnderfitDetector

# Your training curves
train_losses = [0.6, 0.5, 0.4, 0.35, 0.32, 0.30]
val_losses = [0.65, 0.55, 0.47, 0.42, 0.40, 0.39]

# Analyze
analysis = OverfitUnderfitDetector.analyze_training_curves(
    train_losses, val_losses
)

print(f"Status: {analysis['status']}")
print(f"Confidence: {analysis['confidence']}")
if analysis['recommendations']:
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
```

---

## 📊 What Gets Generated

After training, you'll find in `experiments/<exp_id>/`:

### `data_validation_report.json`
```json
{
  "validation_passed": true,
  "errors": [],
  "warnings": [],
  "leakage": {
    "exact_overlap_count": 0,
    "has_exact_leakage": false
  },
  "train_balance": {
    "num_classes": 2,
    "imbalance_ratio": 1.05
  }
}
```

### `training_health.json`
```json
{
  "status": "healthy",
  "confidence": 0.45,
  "train_loss": 0.1815,
  "val_loss": 0.1726,
  "loss_gap": -0.0089,
  "recommendations": []
}
```

---

## 🎯 Built-in Datasets

### Sentiment Analysis
- `imdb` - Movie reviews (binary)
- `glue/sst2` - Stanford Sentiment (binary)
- `sentiment140` - Twitter (3-class)
- `yelp_polarity` - Yelp reviews (binary)

### Natural Language Inference
- `glue/mnli` - Multi-Genre NLI
- `snli` - Stanford NLI

### Question Answering
- `squad` - SQuAD v1.1
- `squad_v2` - SQuAD v2.0

### Summarization
- `cnn_dailymail` - News summarization
- `xsum` - Extreme summarization

---

## ⚠️ What to Watch For

### Pre-Training Validation

If you see this:
```
❌ DATA LEAKAGE DETECTED!
   1998 samples (100.00% of validation) appear in both train and val!

Data validation failed. Continue anyway? (yes/no):
```

**Action**: Type `no` and fix the data split.

### Post-Training Health

If you see this:
```
Status: ❌ OVERFITTING
Confidence: 85.0%

Recommendations:
  1. Reduce model complexity or use regularization
  2. Add dropout or increase dropout rate
  3. Use early stopping
```

**Action**: Follow the recommendations or stop training earlier.

---

## 🔧 Configuration

### Adjust Sample Limit for Validation

In `training/trainer.py` (line ~465):
```python
max_check_samples = 1000  # Increase if you want more thorough checking
```

### Custom Overfitting Thresholds

The detector uses these defaults:
- Overfitting: loss_gap > 20%
- Severe overfitting: loss_gap > 50%
- Underfitting: both losses > 0.5

Thresholds are in `core/utils/data_validator.py` line ~260+

---

## 📖 Full Documentation

- **Complete Guide**: `NEW_FEATURES_GUIDE.md`
- **Implementation Report**: `DATA_QUALITY_FEATURES_COMPLETE.md`
- **Test Suite**: `test_new_features.py`

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'datasets'"

```bash
pip install datasets
```

### HuggingFace download fails

```bash
# Login if dataset requires authentication
huggingface-cli login

# Or use offline mode if cached
export HF_DATASETS_OFFLINE=1
```

### Validation too slow

Decrease the sample limit in `trainer.py` or skip validation for quick tests (not recommended for production).

---

## ✅ Benefits

| Feature | Before | After |
|---------|--------|-------|
| Data Leakage | ❌ Undetected | ✅ Auto-detected before training |
| Dataset Prep | ❌ Manual downloads | ✅ One-line from HuggingFace |
| Overfitting | ❌ No warnings | ✅ Real-time detection |
| Class Imbalance | ❌ Discovered late | ✅ Warned upfront |

---

## 🎉 Summary

**3 Major Features**:
1. ✅ Automatic data leakage detection
2. ✅ HuggingFace dataset integration (15+ datasets)
3. ✅ Overfitting/underfitting detection

**Status**: Production ready, all tests passing ✅

**Next**: Run `python app/main.py` to see it in action!
