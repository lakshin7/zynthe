# New Features: Data Validation, HuggingFace Integration & Training Health Monitoring

## 🎯 Overview

We've integrated three major enhancements to prevent data leakage, simplify dataset management, and detect overfitting/underfitting:

### ✨ **Feature 1: Automatic Data Leakage Detection**
- **What**: Checks train/val overlap before every training run
- **Why**: Prevents fake "perfect" results from leaked data
- **When**: Runs automatically at the start of `trainer.fit()`

### 🤗 **Feature 2: HuggingFace Dataset Integration**
- **What**: Direct loading from HuggingFace Hub
- **Why**: No manual downloads, automatic train/val splits
- **When**: Use in preflight or standalone

### 📊 **Feature 3: Overfitting/Underfitting Detection**
- **What**: Analyzes training curves to detect issues
- **Why**: Get actionable recommendations during training
- **When**: Runs after every epoch, summary at end

---

## 📋 Quick Start

### Test the New Features

```bash
cd /Users/lakshins/Documents/Zynthe
python test_new_features.py
```

This runs:
1. Data leakage check on existing datasets
2. Overfitting detection scenarios
3. HuggingFace dataset download (SST-2 sample)

---

## 🔍 Feature 1: Data Leakage Detection

### Automatic Integration (No Code Changes Needed!)

The trainer now automatically validates data before training:

```python
# In trainer.py - runs automatically in fit()
from core.utils.data_validator import DataValidator

# Checks for:
# - Exact text overlap between train/val
# - Near-duplicates (first 100 chars)
# - Class imbalance issues
# - Dataset size warnings
```

### What Happens

**Before Training Starts:**
```
================================================================================
PRE-TRAINING DATA VALIDATION
================================================================================
Checking for data leakage between train and validation sets...
✓ No exact overlap detected between train and val
⚠️  Training set: Class imbalance detected. Ratio: 1.58:1
   Consider using class weights or oversampling

Validation Summary:
  Train: 1599 samples, 2 classes
  Val:   401 samples, 2 classes
  Leakage: ✓ None
  Status: ✓ PASSED
================================================================================
```

**If Leakage Detected:**
```
❌ DATA LEAKAGE DETECTED!
   1998 samples (100.00% of validation) appear in both train and val!

   Example overlapping texts:
   [1] I really liked this Summerslam due to...
   [2] Functioning as a sort of midpoint...

================================================================================
DATA VALIDATION FAILED!
================================================================================
  ❌ DATA LEAKAGE: 1998 samples overlap!

Please fix data issues before training.
See report: experiments/.../data_validation_report.json

Data validation failed. Continue anyway? (yes/no):
```

### Manual Usage

```python
from core.utils.data_validator import DataValidator

# Validate your split
train_data = [{'text': '...', 'label': 0}, ...]
val_data = [{'text': '...', 'label': 1}, ...]

results = DataValidator.validate_dataset_split(
    train_data,
    val_data,
    text_key='text',
    label_key='label'
)

if results['validation_passed']:
    print("✓ Safe to train!")
else:
    print(f"Errors: {results['errors']}")
    print(f"Warnings: {results['warnings']}")

# Save report
from pathlib import Path
DataValidator.save_validation_report(
    results,
    Path('validation_report.json')
)
```

### Output Files

- `experiments/<exp_id>/data_validation_report.json`: Full validation details
- Includes:
  - Overlap count and samples
  - Class distribution per split
  - Imbalance ratios
  - Recommendations

---

## 🤗 Feature 2: HuggingFace Dataset Integration

### Built-in Dataset Catalog

```python
from core.utils.hf_dataset_loader import HuggingFaceDatasetLoader

# List all available datasets
catalog = HuggingFaceDatasetLoader.list_available_datasets()
print(catalog)
# {
#   'sentiment': ['imdb', 'sst2', 'sentiment140', 'yelp_polarity'],
#   'nli': ['mnli', 'snli'],
#   'qa': ['squad', 'squad_v2'],
#   'summarization': ['cnn_dailymail', 'xsum']
# }

# Get info about a dataset
info = HuggingFaceDatasetLoader.get_dataset_info('imdb')
print(info)
# {
#   'path': 'imdb',
#   'text_col': 'text',
#   'label_col': 'label',
#   'num_classes': 2,
#   'task': 'binary_classification'
# }
```

### Quick Download & Prepare

```python
from pathlib import Path
from core.utils.hf_dataset_loader import prepare_hf_dataset

# One command to download, split, and save
paths = prepare_hf_dataset(
    dataset_id='imdb',           # or 'glue/sst2', 'sentiment140', etc.
    output_dir=Path('data/imdb'),
    max_samples=1000,            # Optional: limit size
    split_ratio=(0.8, 0.1, 0.1)  # train/val/test
)

print(paths)
# {
#   'train': Path('data/imdb/train.jsonl'),
#   'val': Path('data/imdb/val.jsonl'),
#   'test': Path('data/imdb/test.jsonl')
# }

# Now use in your config:
config['data']['train_path'] = str(paths['train'])
config['data']['val_path'] = str(paths['val'])
```

### Supported Datasets

#### Sentiment Analysis
- `imdb`: Movie reviews (binary)
- `sst2` (glue/sst2): Stanford Sentiment Treebank
- `sentiment140`: Twitter sentiment (3-class)
- `yelp_polarity`: Yelp reviews (binary)

#### Natural Language Inference
- `mnli` (glue/mnli): Multi-Genre NLI
- `snli`: Stanford NLI

#### Question Answering
- `squad`: SQuAD v1.1
- `squad_v2`: SQuAD v2.0

#### Summarization
- `cnn_dailymail`: News summarization
- `xsum`: Extreme summarization

### Advanced Usage

```python
loader = HuggingFaceDatasetLoader()

# Load raw dataset
dataset = loader.load_from_hub(
    'glue',
    dataset_name='sst2',
    split='train',
    max_samples=500
)

# Convert to JSONL
from pathlib import Path
loader.convert_to_jsonl(
    dataset,
    output_path=Path('data/sst2_train.jsonl'),
    text_col='sentence',
    label_col='label'
)

# For multi-column text (e.g., NLI)
dataset = loader.load_from_hub('snli', split='train')
loader.convert_to_jsonl(
    dataset,
    output_path=Path('data/snli_train.jsonl'),
    combine_cols=['premise', 'hypothesis'],
    separator=' [SEP] ',
    label_col='label'
)
```

---

## 📊 Feature 3: Overfitting/Underfitting Detection

### Automatic Analysis

At the end of training, you'll see:

```
================================================================================
TRAINING HEALTH SUMMARY
================================================================================

Status: ✓ HEALTHY
Confidence: 45.0%

Latest Metrics:
  Train Loss: 0.1815
  Val Loss:   0.1726
  Gap:        -0.0089 (-4.9%)
  Trend:      decreasing

Recommendations:
  (None - training is healthy)

================================================================================
```

### Overfitting Detected

```
================================================================================
TRAINING HEALTH SUMMARY
================================================================================

Status: ❌ OVERFITTING
Confidence: 85.0%

Latest Metrics:
  Train Loss: 0.0400
  Val Loss:   0.6500
  Gap:        0.6100 (1525.0%)
  Trend:      increasing

Recommendations:
  1. Reduce model complexity or use regularization
  2. Add dropout or increase dropout rate
  3. Use early stopping
  4. Increase training data or use data augmentation
  5. Reduce training epochs

================================================================================
```

### Underfitting Detected

```
================================================================================
TRAINING HEALTH SUMMARY
================================================================================

Status: ❌ UNDERFITTING
Confidence: 70.0%

Latest Metrics:
  Train Loss: 0.7000
  Val Loss:   0.7200
  Gap:        0.0200 (2.9%)
  Trend:      decreasing

Recommendations:
  1. Increase model capacity
  2. Train for more epochs
  3. Reduce regularization
  4. Check if learning rate is too low
  5. Verify data quality and preprocessing

================================================================================
```

### Manual Usage

```python
from core.utils.data_validator import OverfitUnderfitDetector

# Analyze your training curves
train_losses = [0.6, 0.5, 0.4, 0.35, 0.32, 0.30]
val_losses = [0.65, 0.55, 0.47, 0.42, 0.40, 0.39]

analysis = OverfitUnderfitDetector.analyze_training_curves(
    train_losses,
    val_losses
)

print(analysis)
# {
#   'status': 'healthy',
#   'confidence': 0.45,
#   'train_loss': 0.30,
#   'val_loss': 0.39,
#   'loss_gap': 0.09,
#   'loss_gap_pct': 30.0,
#   'val_loss_trend': 'decreasing',
#   'recommendations': []
# }
```

### Output Files

- `experiments/<exp_id>/training_health.json`: Full health analysis
- Includes:
  - Status (healthy/overfitting/underfitting/suspicious)
  - Confidence score
  - Loss gaps and trends
  - Specific recommendations

---

## 🎮 Usage Examples

### Example 1: Train with Auto-Validation

```python
# No changes needed! Just run normally:
python app/main.py --config configs/default.yaml

# Validation runs automatically:
# 1. Checks for data leakage before training
# 2. Monitors overfitting during training
# 3. Generates health report at end
```

### Example 2: Download HuggingFace Dataset

```python
from pathlib import Path
from core.utils.hf_dataset_loader import prepare_hf_dataset

# Download SST-2 for sentiment analysis
paths = prepare_hf_dataset(
    'glue/sst2',
    output_dir=Path('data/sst2'),
    max_samples=2000
)

# Update your config
config['data']['train_path'] = str(paths['train'])
config['data']['val_path'] = str(paths['val'])

# Train as normal
python app/main.py --config your_config.yaml
```

### Example 3: Validate Existing Data

```python
from core.utils.data_validator import DataValidator
from pathlib import Path
import json

# Load your data
train_data = []
with open('data/train.jsonl') as f:
    for line in f:
        train_data.append(json.loads(line))

val_data = []
with open('data/val.jsonl') as f:
    for line in f:
        val_data.append(json.loads(line))

# Validate
results = DataValidator.validate_dataset_split(
    train_data, val_data,
    text_key='text',
    label_key='label'
)

# Check results
if not results['validation_passed']:
    print("Issues found:")
    for error in results['errors']:
        print(f"  - {error}")
```

---

## 🔧 Configuration

### Disable Auto-Validation (Not Recommended)

If you want to skip validation:

```python
# In your training code, modify trainer.py temporarily
# (Not recommended - validation is fast and catches critical issues)
```

### Adjust Validation Thresholds

```python
from core.utils.data_validator import OverfitUnderfitDetector

# Custom thresholds
analysis = OverfitUnderfitDetector.analyze_training_curves(
    train_losses,
    val_losses,
    # Default thresholds are in the code:
    # - Overfitting: loss_gap > 20%
    # - Severe overfitting: loss_gap > 50%
    # - Underfitting: both losses > 0.5
)
```

---

## 📁 Output Files

After training with the new features, you'll find:

```
experiments/<experiment_id>/
├── data_validation_report.json  # Pre-training validation
├── training_health.json          # Post-training health analysis
├── metrics.json                  # Standard metrics
├── training_curves.png           # Loss curves
└── ...
```

### `data_validation_report.json`

```json
{
  "validation_passed": true,
  "errors": [],
  "warnings": [
    "Training set: Class imbalance detected. Ratio: 1.58:1"
  ],
  "leakage": {
    "train_size": 1599,
    "val_size": 401,
    "exact_overlap_count": 0,
    "has_exact_leakage": false
  },
  "train_balance": {
    "num_classes": 2,
    "imbalance_ratio": 1.05,
    "is_balanced": true
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
  "loss_gap_pct": -4.9,
  "val_loss_trend": "decreasing",
  "recommendations": []
}
```

---

## ✅ Benefits

### Before (Old System)
- ❌ No detection of data leakage → fake results
- ❌ Manual dataset downloads → time-consuming
- ❌ No overfitting warnings → wasted training time
- ❌ Metrics look perfect but model fails in production

### After (New System)
- ✅ Auto-detection of data leakage → reliable results
- ✅ One-command dataset preparation → save hours
- ✅ Real-time overfitting detection → stop early
- ✅ Actionable recommendations → improve model quality

---

## 🐛 Troubleshooting

### "Data validation failed. Continue anyway?"

**What it means**: The system detected data leakage or severe quality issues.

**What to do**:
1. Type `no` to abort
2. Check `data_validation_report.json`
3. Fix the data issues (see earlier sections)
4. Re-run training

### HuggingFace download fails

**Common causes**:
- No internet connection
- HuggingFace Hub is down
- Dataset requires authentication

**Solutions**:
```bash
# Install HuggingFace datasets if not installed
pip install datasets

# Login if dataset requires authentication
huggingface-cli login

# Use cached version if available
export HF_DATASETS_OFFLINE=1
```

### Validation too slow

**Issue**: Checking large datasets takes time.

**Solution**: Validation only checks first 1000 samples by default (configurable):

```python
# In trainer.py, line ~465:
max_check_samples = 1000  # Increase if needed
```

---

## 📚 API Reference

### DataValidator

```python
from core.utils.data_validator import DataValidator

DataValidator.validate_dataset_split(
    train_data: List[Dict],
    val_data: List[Dict],
    text_key: str = 'text',
    label_key: str = 'label'
) -> Dict[str, Any]
```

### HuggingFaceDatasetLoader

```python
from core.utils.hf_dataset_loader import prepare_hf_dataset

prepare_hf_dataset(
    dataset_id: str,
    output_dir: Path,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_samples: Optional[int] = None,
    text_col: str = 'text',
    label_col: str = 'label'
) -> Dict[str, Path]
```

### OverfitUnderfitDetector

```python
from core.utils.data_validator import OverfitUnderfitDetector

OverfitUnderfitDetector.analyze_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "accuracy"
) -> Dict[str, Any]
```

---

## 🎉 Summary

You now have three powerful tools:

1. **Data Leakage Detection**: Never waste time on fake results again
2. **HuggingFace Integration**: Download and prepare datasets in seconds
3. **Training Health Monitoring**: Know when to stop, when to continue

All features work automatically with minimal code changes. Just run your training as normal and benefit from the enhanced validation and monitoring!

**Test it now:**
```bash
python test_new_features.py
```
