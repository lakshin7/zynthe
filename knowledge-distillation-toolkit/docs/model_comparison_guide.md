# 📊 Teacher vs Student Model Comparison Guide

Complete guide for comparing distilled models with comprehensive analysis and visualizations.

## 🎯 Overview

After training a student model through knowledge distillation, it's essential to comprehensively evaluate its performance against the teacher model. This toolkit provides automated comparison tools that analyze:

- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Error Analysis**: Confusion matrices and error patterns
- **Efficiency Metrics**: Model size, compression ratio, parameter reduction
- **Visual Comparisons**: Side-by-side charts and detailed plots
- **Deployment Recommendations**: Data-driven guidance for production use

## 🚀 Quick Start

### Method 1: CLI Tool (Fastest)

```bash
python examples/compare_teacher_student.py
```

**What it does:**
1. Loads both teacher and student models from your latest experiment
2. Evaluates both on the validation dataset
3. Computes all metrics
4. Generates all visualizations
5. Creates a comprehensive report
6. Saves everything to `experiments/.../comparison/`

### Method 2: Interactive Notebook (Most Detailed)

```bash
jupyter notebook examples/Teacher_vs_Student_Comparison.ipynb
```

**What you get:**
- Step-by-step guided analysis
- Interactive visualizations
- Real-time metric computation
- Customizable analysis sections
- Export-ready results

### Method 3: Programmatic (Most Flexible)

```python
from evaluation.model_comparison import ModelComparator

# Initialize
comparator = ModelComparator(
    teacher_path="path/to/teacher",
    student_path="path/to/student",
    device="mps"
)

# Compare
teacher_results, student_results = comparator.compare_models(dataloader)

# Visualize
comparator.visualize_comparison(teacher_results, student_results, save_dir="output")
```

## 📋 Phase-by-Phase Breakdown

### Phase 1: Setup

**Goal**: Verify model paths and tokenizer configuration

```python
# Verify paths
TEACHER_PATH = "experiments/20250926T052444Z_ba9f0508/teacher_model"
STUDENT_PATH = "experiments/20250926T052444Z_ba9f0508/student_model"

# Check required files
Required files in each directory:
- config.json
- pytorch_model.bin or model.safetensors
- tokenizer.json, vocab.txt (tokenizer files)
```

**Best Practice**: Use the same tokenizer for both models to ensure fair comparison.

### Phase 2: Load Models

**Goal**: Load both models correctly and gather statistics

```python
comparator = ModelComparator(
    teacher_path=TEACHER_PATH,
    student_path=STUDENT_PATH,
    device="mps",
    use_same_tokenizer=True  # Recommended!
)

print(f"Teacher: {comparator.teacher_params:,} parameters")
print(f"Student: {comparator.student_params:,} parameters")
print(f"Compression: {comparator.compression_ratio:.2f}x")
```

**What happens:**
- Models loaded with `from_pretrained()`
- Moved to specified device (mps/cuda/cpu)
- Set to evaluation mode
- Parameter counts calculated

### Phase 3: Dataset Preparation

**Goal**: Use identical dataset for fair comparison

```python
from data.dataloaders import get_imdb_dataloaders

train_loader, val_loader = get_imdb_dataloaders(
    train_path="data/imdb_train.jsonl",
    val_path="data/imdb_val.jsonl",
    tokenizer=comparator.tokenizer,  # Use shared tokenizer!
    batch_size=8,
    max_length=128
)
```

**Critical**: Both models MUST use the same:
- Dataset
- Tokenizer
- Max sequence length
- Batch processing order

### Phase 4: Evaluation Function

**Goal**: Run inference and collect predictions

```python
def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Collect
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return predictions, labels
```

**ModelComparator handles this automatically!**

### Phase 5: Metrics Computation

**Goal**: Calculate comprehensive metrics

```python
teacher_results, student_results = comparator.compare_models(val_loader)

# Access metrics
print(f"Teacher Accuracy: {teacher_results['accuracy']:.4f}")
print(f"Student Accuracy: {student_results['accuracy']:.4f}")
print(f"Teacher F1: {teacher_results['f1']:.4f}")
print(f"Student F1: {student_results['f1']:.4f}")
```

**Metrics included:**
- Accuracy (overall correct predictions)
- Precision (per-class and macro-averaged)
- Recall (per-class and macro-averaged)
- F1-Score (harmonic mean of precision/recall)
- Confusion Matrix (error breakdown)
- Loss (average evaluation loss)

### Phase 6: Visualization

**Goal**: Generate comparison charts

```python
comparator.visualize_comparison(
    teacher_results,
    student_results,
    save_dir="experiments/.../comparison"
)
```

**Generated visualizations:**

1. **metrics_comparison.png**
   - Side-by-side bar chart
   - Accuracy, Precision, Recall, F1
   - Teacher (blue) vs Student (purple)

2. **confusion_matrices_comparison.png**
   - Two heatmaps side by side
   - True/False Positives/Negatives
   - Identify systematic errors

3. **per_class_comparison.png**
   - Per-class precision, recall, F1
   - Identify which classes suffer most

4. **efficiency_comparison.png**
   - Scatter plot: Size vs Performance
   - Shows compression trade-off
   - Includes annotations

5. **comparison_table.png**
   - Comprehensive table image
   - All metrics formatted
   - Easy to include in presentations

### Phase 7: Reporting

**Goal**: Generate deployment recommendations

```python
comparator.generate_report(
    teacher_results,
    student_results,
    save_dir="experiments/.../comparison"
)
```

**Report includes:**
- Executive summary
- Model statistics
- Performance comparison table
- Embedded visualizations
- Deployment verdict and recommendations

**Sample verdict:**
```markdown
## 🎯 Conclusion

✅ **Excellent**: Student model maintains near-identical performance 
with significant size reduction.

The student model achieves **1.64x compression** with only **0.10%** 
accuracy difference, making it a viable candidate for deployment in 
resource-constrained environments.
```

## 📊 Understanding the Results

### Accuracy Analysis

**Excellent (< 1% drop)**
```
Teacher: 0.9800
Student: 0.9790
Drop: 0.10%
→ Deploy with confidence
```

**Good (1-3% drop)**
```
Teacher: 0.9800
Student: 0.9520
Drop: 2.80%
→ Deploy after A/B testing
```

**Fair (3-5% drop)**
```
Teacher: 0.9800
Student: 0.9350
Drop: 4.50%
→ Evaluate criticality of use case
```

**Poor (> 5% drop)**
```
Teacher: 0.9800
Student: 0.9100
Drop: 7.00%
→ Consider retraining with adjusted hyperparameters
```

### Compression Analysis

**Formula:**
```
Compression Ratio = Teacher Parameters / Student Parameters
Size Reduction = (1 - 1/Compression Ratio) × 100%
```

**Example:**
```
Teacher: 109,483,778 params (417.6 MB)
Student:  66,955,010 params (255.3 MB)
Compression: 1.64x
Reduction: 39.0%
Saved: 162.3 MB
```

### Confusion Matrix Analysis

**Ideal case (balanced errors):**
```
Teacher:  [[980, 44], [20, 956]]
Student:  [[975, 49], [25, 951]]
→ Similar error distribution
```

**Problem case (biased errors):**
```
Teacher:  [[980, 44], [20, 956]]
Student:  [[950, 74], [8, 968]]
→ Student has higher FP rate, lower FN rate
→ May indicate bias in distillation
```

## 🎯 Deployment Decision Matrix

| Accuracy Drop | Compression | Recommendation |
|--------------|-------------|----------------|
| < 1% | Any | ✅ Deploy immediately |
| 1-2% | > 2x | ✅ Strong candidate |
| 2-3% | > 3x | ✔️ Good trade-off |
| 3-5% | > 4x | ⚠️ Evaluate carefully |
| > 5% | Any | ❌ Retrain or adjust |

## 🔧 Troubleshooting

### Models not found
```bash
Error: Teacher model not found at: experiments/.../teacher_model
```

**Solution**: Ensure training completed successfully and saved both models.

```python
# Check trainer.py saves both models
student_save_dir = os.path.join(self.experiment_dir, 'student_model')
teacher_save_dir = os.path.join(self.experiment_dir, 'teacher_model')
```

### Tokenizer mismatch
```
Error: Input mismatch between teacher and student
```

**Solution**: Use `use_same_tokenizer=True` in ModelComparator.

### Device errors
```
Error: Input device mismatch (cpu vs mps)
```

**Solution**: Ensure consistent device usage:
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
comparator = ModelComparator(..., device=device)
```

### Memory issues
```
Error: Out of memory during evaluation
```

**Solution**: Reduce batch size:
```python
val_loader = DataLoader(dataset, batch_size=4)  # Reduced from 8
```

## 📈 Advanced Usage

### Custom Metrics

```python
from evaluation.model_comparison import ModelComparator

class CustomComparator(ModelComparator):
    def compute_custom_metrics(self, results):
        # Add inference time
        # Add memory footprint
        # Add custom domain metrics
        pass
```

### Multiple Dataset Comparison

```python
datasets = {
    'test_set': test_loader,
    'ood_set': ood_loader,
    'adversarial': adv_loader
}

for name, loader in datasets.items():
    teacher_res, student_res = comparator.compare_models(loader)
    comparator.save_results(teacher_res, student_res, f"comparison_{name}")
```

### Export to Different Formats

```python
# Save as JSON
with open('comparison.json', 'w') as f:
    json.dump({
        'teacher': teacher_results,
        'student': student_results
    }, f, indent=2)

# Export to CSV
import pandas as pd
df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1', 'Precision', 'Recall'],
    'Teacher': [teacher_results[m] for m in ['accuracy', 'f1', 'precision', 'recall']],
    'Student': [student_results[m] for m in ['accuracy', 'f1', 'precision', 'recall']]
})
df.to_csv('comparison.csv', index=False)
```

## 📚 Additional Resources

- [Design Documentation](../docs/design.md)
- [Mac M2 Optimization Guide](../docs/msme_playbook.md)
- [Evaluation Metrics Deep Dive](../docs/metrics.md)

## ✅ Checklist

- [ ] Verify teacher and student models exist
- [ ] Use same tokenizer for both models
- [ ] Load same dataset for evaluation
- [ ] Run comparison on validation/test set
- [ ] Generate all visualizations
- [ ] Review confusion matrices
- [ ] Check deployment recommendations
- [ ] Save results and report
- [ ] Document findings for team review

## 🎉 You're Ready!

You now have all the tools to comprehensively compare your distilled models and make informed deployment decisions. Happy distilling! 🚀
