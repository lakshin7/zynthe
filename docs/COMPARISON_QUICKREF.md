# Teacher-Student Comparison: Quick Reference

## 🚀 Quick Start

### Run Comparison During Training

```bash
# Full training with automatic comparison
python app/main.py --config configs/default.yaml
```

Ensure your config has:
```yaml
compare_models: true  # Enable comparison
```

### Run Comparison on Existing Models

```bash
# On specific experiment
python test_comparison.py --exp experiments/20251018T100839Z_9b3dfc41

# On latest experiment
python test_comparison.py --exp experiments/$(ls -t experiments | head -1)
```

---

## 📁 Output Structure

```
experiments/YOUR_EXPERIMENT_ID/
├── teacher_model/              # Trained teacher
├── student_model/              # Trained student
└── comparison/                 # ← NEW! Comparison results
    ├── COMPARISON_REPORT.md    # Main report
    ├── comparison_results.json # All metrics
    ├── metrics_comparison.png  # Visual comparison
    ├── confusion_matrices_comparison.png
    ├── per_class_comparison.png
    ├── efficiency_comparison.png
    ├── comparison_table.png
    ├── final_report.pdf        # Combined PDF
    ├── latency_results.csv     # Speed benchmarks
    ├── compression_summary.txt # Compression stats
    ├── teacher_metrics.json
    └── student_metrics.json
```

---

## 📊 Understanding Results

### Example Output

```
📊 Results Summary:
   Teacher Accuracy:  0.9630
   Student Accuracy:  0.9855
   Accuracy Drop:     -0.0225  ← Negative = Student is better!
   Compression Ratio: 1.64x    ← 38.8% smaller
   Teacher Params:    109,483,778
   Student Params:    66,955,010
```

### Verdict Guide

| Accuracy Drop | Verdict | What it means |
|---------------|---------|---------------|
| < 0% (negative) | 🌟 Outstanding | Student outperforms teacher! |
| 0-2% | ✅ Excellent | Minimal performance loss |
| 2-5% | ✔️ Good | Acceptable tradeoff |
| > 5% | ⚠️ Fair | Consider retraining |

---

## 🔧 Configuration

### Enable/Disable Comparison

**In YAML config:**
```yaml
compare_models: true   # Enable (default)
# compare_models: false  # Disable
```

**Via command line:**
```bash
python app/main.py --config configs/default.yaml --override compare_models=false
```

### Adjust Settings

```yaml
train:
  batch_size: 8          # For evaluation
  
model:
  max_length: 128        # Sequence length
  
data:
  val_path: "data/imdb_val.jsonl"  # Validation data
```

---

## 🐛 Troubleshooting

### Problem: Models not found

**Error:** `Teacher/Student model not found`

**Fix:** Check training completed:
```bash
ls experiments/YOUR_EXP/teacher_model
ls experiments/YOUR_EXP/student_model
```

### Problem: Low accuracy (~50%)

**Warning:** `Accuracy near chance level`

**Possible causes:**
- Tokenizer mismatch
- Model not trained
- Wrong checkpoint

**Fix:** Use separate tokenizers:
```bash
python examples/compare_teacher_student.py \
    --exp YOUR_EXP \
    --tokenizer-mode separate
```

### Problem: Out of memory

**Error:** CUDA/MPS allocation failed

**Fix:** Reduce batch size:
```bash
python test_comparison.py --exp YOUR_EXP --batch-size 4
```

---

## 💡 Tips & Tricks

### 1. Compare Multiple Experiments

```bash
#!/bin/bash
for exp in experiments/*/; do
    python test_comparison.py --exp "$exp"
done
```

### 2. Find Best Model

```python
import json
from pathlib import Path

results = []
for exp_dir in Path("experiments").glob("*/comparison"):
    with open(exp_dir / "comparison_results.json") as f:
        data = json.load(f)
        results.append({
            'exp': exp_dir.parent.name,
            'student_acc': data['student']['accuracy'],
            'compression': data['comparison']['compression_ratio']
        })

# Sort by accuracy
best = sorted(results, key=lambda x: x['student_acc'], reverse=True)[0]
print(f"Best model: {best['exp']} with {best['student_acc']:.4f} accuracy")
```

### 3. Quick Visual Check

```bash
# Open comparison report
open experiments/YOUR_EXP/comparison/COMPARISON_REPORT.md

# View metrics chart
open experiments/YOUR_EXP/comparison/metrics_comparison.png

# Check PDF report
open experiments/YOUR_EXP/comparison/final_report.pdf
```

### 4. Extract Key Metrics

```bash
# Using jq to extract accuracy
jq '.student.accuracy' experiments/YOUR_EXP/comparison/comparison_results.json

# Extract compression ratio
jq '.comparison.compression_ratio' experiments/YOUR_EXP/comparison/comparison_results.json
```

---

## 📚 Files Generated

| Priority | File | Use Case |
|----------|------|----------|
| ⭐⭐⭐ | `COMPARISON_REPORT.md` | Quick review |
| ⭐⭐⭐ | `metrics_comparison.png` | Visual overview |
| ⭐⭐ | `comparison_results.json` | Programmatic access |
| ⭐⭐ | `final_report.pdf` | Share with team |
| ⭐ | `latency_results.csv` | Performance tuning |
| ⭐ | `compression_summary.txt` | Deployment planning |

---

## 🎯 Common Use Cases

### Use Case 1: Validate Training

**After training, check if distillation worked:**

```bash
python test_comparison.py --exp experiments/LATEST
```

**Look for:**
- ✅ Student accuracy > 95% of teacher
- ✅ Compression ratio > 1.5x
- ✅ No sanity warnings

### Use Case 2: Optimize for Deployment

**Compare multiple configurations:**

```bash
# Train with different configs
python app/main.py --config configs/config_a.yaml
python app/main.py --config configs/config_b.yaml
python app/main.py --config configs/config_c.yaml

# Review comparison results
cat experiments/*/comparison/COMPARISON_REPORT.md
```

**Choose based on:**
- Highest accuracy
- Best compression
- Lowest latency

### Use Case 3: Document Results

**Generate report for stakeholders:**

```bash
python test_comparison.py --exp experiments/PRODUCTION_MODEL
cp experiments/PRODUCTION_MODEL/comparison/final_report.pdf reports/model_v2.pdf
```

---

## 🔍 What to Look For

### ✅ Good Signs

- Student accuracy within 2% of teacher
- High compression ratio (> 1.5x)
- Similar confusion matrix patterns
- No sanity warnings

### ⚠️ Warning Signs

- Accuracy drop > 5%
- Accuracy around 50% (chance level)
- Different confusion matrix patterns
- Sanity warnings in report

### 🔴 Red Flags

- Student accuracy < 50%
- Compression ratio < 1.2x
- Completely different predictions
- Multiple sanity warnings

---

## 📖 Related Documentation

- **Full Guide**: [docs/COMPARISON_FEATURE.md](COMPARISON_FEATURE.md)
- **API Reference**: [evaluation/model_comparison.py](../evaluation/model_comparison.py)
- **Config Guide**: [docs/CONFIG_MANAGER_IMPROVEMENTS.md](CONFIG_MANAGER_IMPROVEMENTS.md)
- **Training Guide**: [docs/quickstart.md](quickstart.md)

---

## 🚀 Next Steps

After reviewing comparison results:

1. **If student is good** (< 2% drop):
   - ✅ Export for deployment
   - ✅ Run additional tests
   - ✅ Deploy to production

2. **If student is okay** (2-5% drop):
   - 🔄 Try different hyperparameters
   - 🔄 Increase training epochs
   - 🔄 Adjust temperature/alpha

3. **If student is poor** (> 5% drop):
   - ❌ Review training logs
   - ❌ Check for tokenizer issues
   - ❌ Consider larger student model

---

## 💪 Pro Tips

1. **Always review visualizations** - Numbers don't tell the full story
2. **Check per-class metrics** - Overall accuracy can hide class imbalances  
3. **Benchmark latency** - Real-world speed matters
4. **Save all experiments** - Compare historical performance
5. **Document decisions** - Use comparison reports in docs

---

## ⚡ One-Liners

```bash
# Latest experiment comparison
python test_comparison.py --exp "experiments/$(ls -t experiments | head -1)"

# Open latest report
open "experiments/$(ls -t experiments | head -1)/comparison/COMPARISON_REPORT.md"

# Check latest accuracy
jq '.student.accuracy' "experiments/$(ls -t experiments | head -1)/comparison/comparison_results.json"

# List all experiment accuracies
for f in experiments/*/comparison/comparison_results.json; do 
  echo "$(dirname $f): $(jq -r '.student.accuracy' $f)"; 
done
```

---

**Need help?** Check the [full documentation](COMPARISON_FEATURE.md) or open an issue on GitHub!
