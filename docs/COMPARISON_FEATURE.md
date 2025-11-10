# Teacher vs Student Model Comparison Feature

## Overview

The Knowledge Distillation Toolkit now automatically compares teacher and student models after training, generating comprehensive visualizations, metrics, and reports. This feature is integrated into the main training pipeline and can also be run independently.

---

## Features

### 🎯 What Gets Compared

1. **Performance Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - Per-class metrics
   - Confusion matrices
   - Loss values

2. **Model Statistics**
   - Parameter counts
   - Compression ratio
   - Model file sizes
   - Memory footprint

3. **Efficiency Metrics**
   - Inference latency
   - Throughput
   - FLOPs estimation (if thop is installed)
   - Speedup calculations

4. **Comprehensive Visualizations**
   - Side-by-side metrics comparison
   - Confusion matrix heatmaps
   - Per-class performance charts
   - Efficiency scatter plots
   - Comparison tables

---

## Automatic Comparison (During Training)

The comparison runs automatically at the end of training when `compare_models: true` in the config.

### Configuration

Add to your YAML config:

```yaml
# Evaluation & Comparison
evaluate: true           # Run final evaluation after training
compare_models: true     # Compare teacher vs student (generates visualizations)
```

### Example: Run Full Training with Comparison

```bash
python app/main.py --config configs/default.yaml
```

**What happens:**
1. ✅ Teacher fine-tuning
2. ✅ Knowledge distillation
3. ✅ Student evaluation
4. ✅ Quantization (optional)
5. ✅ **Teacher-Student Comparison** (NEW!)

---

## Manual Comparison (Existing Experiments)

You can run comparison on already-trained models without retraining.

### Using the Test Script

```bash
# Run comparison on specific experiment
python test_comparison.py --exp experiments/20251018T100839Z_9b3dfc41

# Or use the latest experiment
python test_comparison.py --exp experiments/$(ls -t experiments | head -1)
```

### Using the Examples Script

```bash
# Full comparison with options
python examples/compare_teacher_student.py \
    --exp experiments/20251018T100839Z_9b3dfc41 \
    --tokenizer-mode separate \
    --batch-size 8 \
    --max-length 128
```

**Tokenizer Modes:**
- `separate` (recommended): Use original tokenizer for each model
- `student`: Use student tokenizer for both (may skew results)
- `teacher`: Use teacher tokenizer for both (may skew results)

---

## Output Files

All comparison results are saved to: `{experiment_dir}/comparison/`

### 📊 Visualizations

| File | Description |
|------|-------------|
| `metrics_comparison.png` | Bar chart comparing all metrics |
| `visual_comparison.png` | Duplicate of metrics for compatibility |
| `confusion_matrices_comparison.png` | Side-by-side confusion matrices |
| `confusion_matrices/teacher_confusion_matrix.png` | Teacher confusion matrix only |
| `confusion_matrices/student_confusion_matrix.png` | Student confusion matrix only |
| `per_class_comparison.png` | Per-class precision, recall, F1 |
| `efficiency_comparison.png` | Accuracy vs parameters scatter |
| `comparison_table.png` | Detailed metrics table |

### 📄 Reports & Data

| File | Description |
|------|-------------|
| `COMPARISON_REPORT.md` | Markdown report with analysis |
| `comparison_results.json` | JSON with all metrics |
| `teacher_metrics.json` | Detailed teacher metrics |
| `student_metrics.json` | Detailed student metrics |
| `latency_results.csv` | Inference latency benchmarks |
| `compression_summary.txt` | Compression statistics |
| `final_report.pdf` | PDF with all visualizations |

---

## Example Results

### From Recent Run (20251018T100839Z_9b3dfc41)

```
📊 Results Summary:
   Teacher Accuracy:  0.9630 (96.30%)
   Student Accuracy:  0.9855 (98.55%)
   Accuracy Drop:     -0.0225 (student is BETTER!)
   Compression Ratio: 1.64x
   Teacher Params:    109,483,778
   Student Params:    66,955,010
```

**Verdict:** ✅ Excellent! Student model outperforms teacher while being 1.64x smaller.

### Interpretation

- **Negative Accuracy Drop**: Student performs better than teacher (excellent!)
- **1.64x Compression**: 38.8% parameter reduction
- **98.55% Accuracy**: High performance maintained

---

## Programmatic Usage

You can also use the comparison API in your own scripts:

```python
from evaluation.model_comparison import ModelComparator
from data.dataloaders import create_dataloaders
import torch

# Device selection
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Initialize comparator
comparator = ModelComparator(
    teacher_path="experiments/YOUR_EXP/teacher_model",
    student_path="experiments/YOUR_EXP/student_model",
    device=device,
    use_same_tokenizer=True
)

# Load data
config = {"data": {"train_path": "...", "val_path": "..."}, ...}
_, val_loader = create_dataloaders(config, comparator.tokenizer)

# Run comparison
teacher_results, student_results = comparator.compare_models(val_loader)

# Generate visualizations
comparator.visualize_comparison(
    teacher_results,
    student_results,
    save_dir="output/comparison",
    show_plots=False
)

# Save results
comparator.save_results(teacher_results, student_results, save_dir="output/comparison")

# Generate report
comparator.generate_report(teacher_results, student_results, save_dir="output/comparison")

# Export additional artifacts (latency, compression stats, PDF)
comparator.export_expected_artifacts(
    teacher_results, 
    student_results, 
    save_dir="output/comparison"
)
```

---

## Performance Metrics Explained

### Primary Metrics

- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Efficiency Metrics

- **Compression Ratio**: Teacher params / Student params
- **Parameter Reduction**: % of parameters saved
- **Latency**: Average inference time per sample
- **Speedup**: Teacher latency / Student latency

### Verdict Criteria

| Accuracy Drop | Verdict | Recommendation |
|---------------|---------|----------------|
| < 2% | ✅ Excellent | Deploy with confidence |
| 2-5% | ✔️ Good | Acceptable tradeoff |
| > 5% | ⚠️ Fair | Consider retraining |

---

## Troubleshooting

### Issue: Models not found

**Error:** `Teacher model not found at: ...`

**Solution:** Ensure training completed successfully and models were saved:
```python
# Check in your config:
train:
  save_best_only: true  # Enable model saving
```

### Issue: Low accuracy (around 50%)

**Warning:** `Accuracy is near chance level (50%)`

**Possible causes:**
- Tokenizer mismatch between training and evaluation
- Model not trained properly
- Wrong model checkpoint loaded

**Solution:** Use `tokenizer-mode: separate` for fair comparison

### Issue: Memory error

**Error:** `CUDA out of memory` or MPS allocation failure

**Solution:** Reduce batch size:
```bash
python test_comparison.py --exp YOUR_EXP --batch-size 4
```

---

## Configuration Options

### In YAML Config

```yaml
# Enable/disable comparison
compare_models: true

# Enable/disable final evaluation
evaluate: true

# Control what gets saved
train:
  save_best_only: true  # Only save best checkpoints

# Batch size for evaluation
train:
  batch_size: 8
```

### Command Line

```bash
# Override config settings
python app/main.py --config configs/default.yaml --override compare_models=false

# For manual comparison
python test_comparison.py \
    --exp experiments/YOUR_EXP \
    --batch-size 8 \
    --max-length 128
```

---

## Advanced Features

### 1. Latency Benchmarking

Measures actual inference time on sample texts:

```python
sample_texts = [
    "The movie was wonderful!",
    "This film was disappointing.",
    "Average storyline but great acting."
]

teacher_latency = comparator.measure_latency(comparator.teacher, sample_texts, runs=50)
student_latency = comparator.measure_latency(comparator.student, sample_texts, runs=50)
```

### 2. FLOPs Estimation

Requires `thop` package:

```bash
pip install thop
```

Then FLOPs will be automatically computed:

```python
teacher_flops = comparator.estimate_flops(comparator.teacher)
student_flops = comparator.estimate_flops(comparator.student)
```

### 3. Custom Metrics

Add your own metrics to the comparison:

```python
def custom_metric(results):
    # Your metric logic
    return value

teacher_custom = custom_metric(teacher_results)
student_custom = custom_metric(student_results)
```

---

## Best Practices

### ✅ Do:

1. **Use same tokenizer** for fair comparison (`use_same_tokenizer=True`)
2. **Run on validation set** (not training set)
3. **Check sanity warnings** in the report
4. **Compare multiple experiments** to find best configuration
5. **Save comparison results** for documentation

### ❌ Don't:

1. **Don't compare with different datasets** (unfair)
2. **Don't ignore accuracy warnings** (50% = chance level)
3. **Don't use training set** for evaluation (overfitting)
4. **Don't skip visualization review** (catch issues visually)

---

## Integration with CI/CD

You can integrate comparison into your CI/CD pipeline:

```bash
#!/bin/bash
# train_and_compare.sh

# Train models
python app/main.py --config configs/production.yaml

# Get latest experiment
LATEST_EXP=$(ls -t experiments | head -1)

# Run comparison
python test_comparison.py --exp "experiments/$LATEST_EXP"

# Check if accuracy drop is acceptable
python scripts/check_accuracy_threshold.py "experiments/$LATEST_EXP/comparison"
```

---

## Related Files

- **Main integration**: `app/main.py` (lines 280-350)
- **Comparison module**: `evaluation/model_comparison.py`
- **Test script**: `test_comparison.py`
- **Example script**: `examples/compare_teacher_student.py`
- **Config option**: `configs/default.yaml` (compare_models flag)

---

## Future Enhancements

Planned features for future releases:

- [ ] Multi-model comparison (compare multiple students)
- [ ] Historical comparison (track improvements across experiments)
- [ ] Interactive HTML reports
- [ ] Automatic deployment decision based on thresholds
- [ ] Integration with MLflow/Weights & Biases
- [ ] A/B testing support
- [ ] Production monitoring integration

---

## Summary

✅ **Automatic comparison** after training  
✅ **Comprehensive metrics** and visualizations  
✅ **Multiple output formats** (PNG, JSON, MD, PDF, CSV)  
✅ **Latency benchmarking** and efficiency analysis  
✅ **Programmatic API** for custom workflows  
✅ **Sanity checks** and warnings  
✅ **Production-ready** reports  

The teacher-student comparison feature provides everything you need to make informed decisions about model deployment!
