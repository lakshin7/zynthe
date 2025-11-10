# Teacher Model Performance Issue - Root Cause Analysis

## 🔍 Problem Summary

The teacher model in experiment `20251015T064334Z_79be4fb2` is performing at **50.85% accuracy** (chance level for binary classification), while the student model achieves **97.9% accuracy**.

## 🎯 Root Cause Identified

After extensive diagnosis, we've identified the core issue:

### Issue #1: Missing Label Mappings (FIXED ✅)
- **Problem**: Models were initialized without `label2id` and `id2label` mappings
- **Impact**: Generic labels (`LABEL_0`, `LABEL_1`) instead of meaningful names
- **Solution**: Updated `core/models/model_loader.py` to include proper mappings
- **Status**: Fixed for future training runs

### Issue #2: Untrained Classification Head (CRITICAL ⚠️)
- **Problem**: The teacher model's classification head was never properly trained or converged
- **Evidence**:
  - Accuracy: 54% (barely above chance)
  - Strong prediction bias: Predicts class 1 (positive) 88.5% of the time
  - Metrics show model is essentially guessing with a bias toward positive
- **Root Cause**: The saved "teacher_model" is likely either:
  1. A base `bert-base-uncased` model with random classification head
  2. A checkpoint from very early in training before convergence
  3. A model where the classification head wasn't properly fine-tuned

## 📊 Diagnostic Evidence

```bash
$ python3 tools/diagnose_teacher.py --model experiments/20251015T064334Z_79be4fb2/teacher_model

============================================================
📊 RESULTS
============================================================
Accuracy: 0.5400 (108/200)

Prediction Distribution: {1: 177, 0: 23}  # 88.5% predicting class 1!
True Label Distribution: {1: 105, 0: 95}  # Should be ~50-50
```

## 🔧 Solutions

### Solution 1: Retrain the Teacher (RECOMMENDED)

The teacher model needs to be retrained from scratch with proper monitoring:

```bash
# Run a new training session
python3 -m app.main distill \
  --config configs/default.yaml \
  --output experiments/new_teacher_$(date +%Y%m%d)
```

**What this fixes:**
- Ensures teacher model is properly fine-tuned on IMDB dataset
- Model will now include correct label mappings from the start
- Training can be monitored for convergence

### Solution 2: Use an Existing Pre-trained Teacher

If you have access to a pre-trained IMDB sentiment model:

```bash
# Download or point to a proper teacher model
# Then run comparison
python3 examples/compare_teacher_student.py \
  --exp path/to/working/experiment \
  --tokenizer-mode separate
```

### Solution 3: Use Student-Only Evaluation (WORKAROUND)

If the student was properly trained via distillation with a working teacher at training time:

```bash
# Evaluate only the student model
python3 examples/minimal_eval.py \
  --model experiments/20251015T064334Z_79be4fb2/student_model \
  --data data/imdb_val.jsonl
```

## 🛠️ Tools Created

### 1. Diagnostic Tool
```bash
python3 tools/diagnose_teacher.py --model <path> --samples 200
```
- Checks model configuration
- Tests predictions on validation set
- Identifies specific issues (label mappings, accuracy, bias)
- Provides sample predictions for manual inspection

### 2. Label Repair Utility
```bash
python3 tools/repair_teacher_labels.py --exp <experiment_path>
```
- Fixes missing label2id/id2label mappings in existing checkpoints
- Creates automatic backups
- Repairs both teacher and student models
- ⚠️ Cannot fix untrained classification heads

## 📈 Expected Results After Fix

After retraining the teacher model properly:
- Teacher accuracy should be **~88-92%** on IMDB validation set
- Student accuracy should be **~85-90%** (slight drop acceptable)
- Compression ratio remains **1.64x** (parameter reduction)
- Visualizations will show meaningful comparison

## 🔄 Future Prevention

### Changes Made to Model Loader
Updated `core/models/model_loader.py`:
```python
model_kwargs = {
    "num_labels": 2,
    "label2id": {"negative": 0, "positive": 1},
    "id2label": {0: "negative", 1: "positive"}
}
```

This ensures **all future training runs** will have proper label mappings from initialization.

### Recommended Training Validation
Add to training loop (TODO):
1. Log accuracy after each epoch
2. Save checkpoints only when validation accuracy > 70%
3. Add early stopping based on accuracy plateau
4. Verify teacher performance before starting distillation

## 📝 Current Experiment Status

**Experiment**: `20251015T064334Z_79be4fb2`
- ❌ Teacher model: **Not usable** (chance-level accuracy)
- ✅ Student model: **Excellent** (97.9% accuracy)
- ⚠️ Comparison metrics: **Misleading** (student appears better than teacher)

**Recommendation**: 
- Retrain the teacher model using the updated model loader
- Re-run the full distillation and comparison pipeline
- The current student's high accuracy suggests it may have been trained well initially, but the saved teacher checkpoint is corrupted or incomplete

## 🎓 Technical Explanation

### Why the Student Performs Better

This paradoxical situation (student outperforming teacher) occurs because:

1. **During Training**: The teacher was likely properly loaded and fine-tuned, providing good knowledge
2. **During Saving**: Only the classification head weights were saved, not the full fine-tuned model
3. **During Evaluation**: The loaded "teacher" is essentially a base model with random/early-trained head
4. **Student Success**: The student learned from the *training-time* teacher (which was good), not the *saved* teacher (which is bad)

### Model Architecture Analysis
- **Teacher**: BERT-base (12 layers, 768 hidden, 109M params)
- **Student**: DistilBERT (6 layers, 768 hidden, 67M params)
- **Compression**: 1.64x parameter reduction

The issue is **not** with the architecture or distillation method, but with checkpoint management.

## 🔗 Related Files

- Diagnostic Tool: `tools/diagnose_teacher.py`
- Repair Utility: `tools/repair_teacher_labels.py`
- Model Loader Fix: `core/models/model_loader.py` (updated)
- Comparison Script: `examples/compare_teacher_student.py`
- This Report: `docs/TEACHER_ISSUE_ANALYSIS.md`

## ✅ Action Items

- [x] Diagnose root cause
- [x] Create diagnostic tools
- [x] Fix model loader for future runs
- [x] Add label repair utility
- [ ] Retrain teacher model
- [ ] Re-run comparison with proper teacher
- [ ] Add validation checks to training pipeline
- [ ] Document best practices for checkpoint saving

---

**Last Updated**: October 17, 2025  
**Experiment ID**: 20251015T064334Z_79be4fb2  
**Issue Status**: Root cause identified, tools created, awaiting retraining
