# Teacher Training & Confusion Matrix Guide

## Overview

This guide covers two important features:
1. **Optional Teacher Training** - Fine-tune the teacher model before distillation
2. **Confusion Matrix Interpretation** - Understand the evaluation results

---

## 1. Teacher Training Option

### Why Train the Teacher?

By default, knowledge distillation uses a **pre-trained teacher** model to guide the student. However, you might want to train the teacher first in these scenarios:

✅ **When to train the teacher:**
- Teacher is not pre-trained on your specific task
- You want to adapt a general model to your domain
- You need the teacher to learn task-specific patterns
- You're using a custom or untrained teacher architecture

❌ **When NOT to train the teacher:**
- Teacher is already fine-tuned on your task (e.g., bert-base-uncased for sentiment)
- Using standard pre-trained models (most common case)
- Teacher is from HuggingFace and already performs well

### Configuration

Add these parameters to your config file:

```yaml
train:
  epochs: 2                    # Distillation epochs
  batch_size: 16
  lr: 3e-5
  
  # Teacher training (optional)
  train_teacher: true          # Set to true to enable
  teacher_epochs: 2            # Number of epochs for teacher
  teacher_lr: 2e-5             # Learning rate for teacher
```

### Workflow with Teacher Training

```
Phase 2.5: Fine-tuning Teacher Model
├── Train teacher for N epochs
├── Evaluate teacher on validation set
├── Save best teacher checkpoint
└── Use trained teacher for distillation

Phase 3-4: Distillation Training
├── Freeze teacher (inference mode)
├── Train student using teacher's soft logits
└── Student learns from trained teacher
```

### Example Configs

**Without teacher training (default):**
```yaml
train:
  train_teacher: false         # Use pre-trained teacher as-is
```

**With teacher training:**
```yaml
train:
  train_teacher: true          # Fine-tune teacher first
  teacher_epochs: 2            # Train for 2 epochs
  teacher_lr: 2e-5             # Conservative learning rate
```

### Testing

Test with teacher training enabled:

```bash
python app/main.py --config configs/with_teacher_training.yaml
```

Expected output:
```
======================================================================
PHASE 2.5: Fine-tuning Teacher Model
======================================================================

[INFO] Training teacher for 2 epochs before distillation...
[TEACHER] Epoch 1/2
[TEACHER] Epoch 1: Train Loss=0.3245, Val Loss=0.2891, Accuracy=0.8850, F1=0.8834
[TEACHER] Best model updated (val_loss=0.2891)
[TEACHER] Epoch 2/2
[TEACHER] Epoch 2: Train Loss=0.2567, Val Loss=0.2645, Accuracy=0.9100, F1=0.9089
[TEACHER] Best model updated (val_loss=0.2645)
[TEACHER] Restored best teacher model (val_loss=0.2645)
[INFO] Teacher training completed. Starting distillation...

======================================================================
PHASE 3-4: Distillation Training
======================================================================
...
```

---

## 2. Confusion Matrix Guide

### Understanding the Matrix

The confusion matrix shows how your model's predictions compare to true labels.

**Structure:**
- **Rows** = True/Actual labels
- **Columns** = Predicted labels
- **Diagonal** = Correct predictions (darker blue)
- **Off-diagonal** = Errors (lighter blue)

### Example Interpretation

Your recent test results:

```
Confusion Matrix:
              Predicted
              0       1
Actual  0  [ 999     25 ]  ← Class 0: 999 correct, 25 wrong
        1  [  76    900 ]  ← Class 1: 900 correct, 76 wrong
```

**Metrics derived:**
- **Class 0 (Negative)**: 
  - Precision = 999/(999+76) = 92.9%
  - Recall = 999/(999+25) = 97.6%
  
- **Class 1 (Positive)**:
  - Precision = 900/(25+900) = 97.3%
  - Recall = 900/(76+900) = 92.2%

- **Overall Accuracy** = (999+900)/2000 = **94.95%** ✨

### Reading the Visual

The enhanced confusion matrix now includes:

1. **Clear labels**: "True Label" vs "Predicted Label"
2. **Subtitle**: "(Rows=Actual, Columns=Predicted)"
3. **Overall accuracy**: Displayed at the bottom
4. **Higher DPI**: 150 DPI for better clarity
5. **Square cells**: Equal aspect ratio for readability

### Common Patterns

**Good model (like yours):**
```
[ 999    25 ]  ← Most predictions on diagonal
[  76   900 ]  ← Few errors off-diagonal
```

**Biased model (needs work):**
```
[ 950    74 ]  ← Predicts mostly one class
[   5    21 ]  ← Poor recall for minority class
```

**Random model (broken):**
```
[ 512   488 ]  ← ~50% accuracy
[ 503   497 ]  ← No learning
```

### Files Generated

After training, you'll find:

```
experiments/YOUR_EXPERIMENT_ID/
├── confusion_matrix.png         # Visual confusion matrix
├── metrics.json                 # Numeric metrics
├── training_curves.png          # Loss/accuracy over time
└── visualizations/
    └── model_comparison.png     # Teacher vs student comparison
```

---

## 3. Complete Workflow Example

### Scenario: Train teacher, distill to student, evaluate

**Config** (`configs/complete_workflow.yaml`):
```yaml
train:
  epochs: 3
  batch_size: 16
  lr: 3e-5
  
  train_teacher: true
  teacher_epochs: 2
  teacher_lr: 2e-5

model:
  name: "roberta-base"
  student_name: "distilroberta-base"

visualization:
  enable: true
  plot_confusion_matrix: true
```

**Run:**
```bash
python app/main.py --config configs/complete_workflow.yaml
```

**Expected phases:**
```
Phase 0: Environment Setup
Phase 1.1: Config Validation
Phase 1.2: Model Loading
Phase 2: Dataset Preparation
Phase 2.5: Fine-tuning Teacher    ← NEW! Teacher training
Phase 3-4: Distillation Training  ← Student learns from trained teacher
Phase 5: Evaluation
Phase 6: Quantization
Phase 9: Visualization            ← Enhanced confusion matrix
Phase 8: Final Report
```

---

## 4. Best Practices

### Teacher Training

1. **Use lower learning rate** for teacher (2e-5 vs 3e-5) to avoid catastrophic forgetting
2. **Fewer epochs** (1-3) - teacher is already pre-trained
3. **Monitor validation loss** - stop if overfitting
4. **Compare with baseline** - ensure trained teacher improves

### Confusion Matrix

1. **Check diagonal** - should be darkest (most correct)
2. **Look for class imbalance** - one class predicted more than others?
3. **Compare teacher vs student** - student should match or approach teacher
4. **Track across epochs** - confusion should decrease

### Performance Expectations

| Scenario | Teacher Acc | Student Acc | Compression |
|----------|-------------|-------------|-------------|
| Good distillation | 95% | 92-94% | 1.5-2x |
| Excellent distillation | 95% | 94-95% | 1.5-2x |
| Poor distillation | 95% | <85% | Check config |

---

## 5. Troubleshooting

### Issue: Teacher accuracy is low after training

**Cause:** 
- Learning rate too high
- Not enough epochs
- Data quality issues

**Fix:**
```yaml
train:
  train_teacher: true
  teacher_epochs: 3        # Increase epochs
  teacher_lr: 1e-5         # Lower learning rate
```

### Issue: Student worse than teacher by >5%

**Cause:**
- Student too small
- Temperature too high/low
- Alpha imbalance

**Fix:**
```yaml
distillation:
  temperature: 2.0         # Try 1.5-3.0
  alpha: 0.5               # Balance hard/soft (0.3-0.7)
```

### Issue: Confusion matrix looks random

**Cause:**
- Model not learning
- Data labels incorrect
- Training diverged

**Fix:**
1. Check data with `head data/imdb_train.jsonl`
2. Verify labels are 0/1 or correct format
3. Lower learning rate
4. Increase training epochs

---

## 6. Quick Reference

### Enable teacher training:
```yaml
train_teacher: true
```

### Disable teacher training (default):
```yaml
train_teacher: false
```

### View confusion matrix:
```bash
open experiments/LATEST_EXPERIMENT/confusion_matrix.png
```

### Compare models:
```bash
python examples/compare_teacher_student.py --experiment experiments/LATEST_EXPERIMENT
```

---

## Summary

✅ **Teacher training** is now optional - enable with `train_teacher: true`
✅ **Confusion matrix** is enhanced with clear labels and accuracy
✅ **Workflow phases** are clearly separated and labeled
✅ **Artifacts** are organized in experiment directories

Your distillation toolkit is now production-ready! 🎉
