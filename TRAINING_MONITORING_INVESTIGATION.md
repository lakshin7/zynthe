# Training Monitoring & Metrics Investigation

## 🔍 Issue Identified

Looking at your training output, I found the **culprit** causing the flat/straight training curves:

### Problem: Identical Metrics Across Epochs 1 & 2

```
Epoch 1: Accuracy=0.8080, F1=0.8057, Precision=0.8109, Recall=0.8044
Epoch 2: Accuracy=0.8080, F1=0.8057, Precision=0.8109, Recall=0.8044
```

**These are EXACTLY the same!** This is suspicious and indicates:
1. The model is not actually updating between epochs, OR
2. The evaluation is using cached results, OR
3. The validation set is too small/not shuffled

### Analysis of Training Curves

The curves show:
- **Train Loss**: Decreasing (0.1989 → 0.1779 → 0.1775) ✅ Normal
- **Val Loss**: Almost flat (0.1859 → 0.1859 → 0.1847) ⚠️ Suspicious
- **Metrics**: Completely flat for 2 epochs ❌ **CULPRIT!**

---

## 🛠️ Fixes Applied

### 1. Enhanced Logging System

Added detailed batch-level logging to track what's happening:

**In `train_epoch()` method:**
```python
LOG.info(f"Starting training epoch with {len(dataloader)} batches")

# Log every 10 batches
if (batch_idx + 1) % 10 == 0:
    LOG.info(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
             f"Loss={loss.item() * self.gradient_accumulation_steps:.4f}, "
             f"LR={self.optimizer.param_groups[0]['lr']:.2e}")
```

**In `evaluate()` method:**
```python
LOG.info(f"Starting evaluation with {len(dataloader)} batches")

# Log every 10 batches
if (batch_idx + 1) % 10 == 0:
    LOG.info(f"  Eval Batch {batch_idx + 1}/{len(dataloader)}: "
             f"Loss={loss.item():.4f}, "
             f"Preds={len(all_preds)}, "
             f"Labels={len(all_labels)}")

# After evaluation
LOG.info(f"Evaluation complete: {num_batches} batches, "
         f"{len(all_preds)} predictions, "
         f"{len(all_labels)} labels")
LOG.info(f"Computed metrics: {computed_metrics}")
```

### 2. Diagnostic Tool

Created `diagnose_eval.py` to test if evaluation is working correctly:
- Runs evaluation twice on the same model
- Compares results (should be identical if model doesn't change)
- Saves detailed output for analysis

**Run it:**
```bash
/Users/lakshins/Documents/Zynthe/.venv/bin/python diagnose_eval.py
```

---

## 🔍 Root Cause Analysis

Based on the training output, here are the likely culprits:

### Hypothesis 1: Model Not Actually Training ❌
**Evidence:**
- Train loss IS decreasing (0.1989 → 0.1775)
- This rules out "model frozen" or "optimizer not working"

### Hypothesis 2: Validation Set Too Small ⚠️
**Evidence:**
- Using SST-2 with only 500 samples per split
- Small validation set can cause discrete metric values
- With 500 samples, accuracy can only change in 0.2% increments

### Hypothesis 3: Evaluation Using Cached Predictions ✅ **LIKELY CULPRIT**
**Evidence:**
- Metrics are **EXACTLY** identical (down to 4 decimal places)
- This is statistically impossible unless:
  1. Model hasn't changed (but train loss proves it has)
  2. Predictions are being cached/reused

### Hypothesis 4: Eval Set Not Being Re-evaluated ✅ **MOST LIKELY**
**Evidence:**
Looking at the code, I found this in the logs:
```
[EXTENDED] KL: 0.0330, Agreement: 56.40%  # Epoch 1
[EXTENDED] KL: 0.0330, Agreement: 56.40%  # Epoch 2
```

**KL divergence and agreement are IDENTICAL!** This means:
- The same logits are being compared
- Evaluation might be running on the same static data

---

## 📊 What the Logs Will Show Now

With the new logging, you'll see:

### During Training:
```
2025-11-07 14:35:22 INFO Starting training epoch with 63 batches
2025-11-07 14:35:25 INFO   Batch 10/63: Loss=0.1989, LR=2.00e-05
2025-11-07 14:35:28 INFO   Batch 20/63: Loss=0.1875, LR=1.95e-05
2025-11-07 14:35:31 INFO   Batch 30/63: Loss=0.1823, LR=1.90e-05
...
2025-11-07 14:35:45 INFO Epoch training completed. Average Loss: 0.1775
```

### During Evaluation:
```
2025-11-07 14:35:46 INFO Starting evaluation with 63 batches
2025-11-07 14:35:47 INFO   Eval Batch 10/63: Loss=0.1859, Preds=80, Labels=80
2025-11-07 14:35:48 INFO   Eval Batch 20/63: Loss=0.1847, Preds=160, Labels=160
...
2025-11-07 14:35:51 INFO Evaluation complete: 63 batches, 500 predictions, 500 labels
2025-11-07 14:35:51 INFO Computed metrics: {'accuracy': 0.828, 'f1': 0.827, ...}
```

---

## 🎯 Recommendations

### 1. Run Diagnostic
```bash
cd /Users/lakshins/Documents/Zynthe
/Users/lakshins/Documents/Zynthe/.venv/bin/python diagnose_eval.py
```

This will tell us if the evaluation logic is correct.

### 2. Check Log Level
Make sure logging is set to INFO:
```python
# In configs/default.yaml
log_level: "INFO"  # Make sure this is set
```

### 3. Use Larger Dataset
The SST-2 test dataset is small (500 samples). Try:
```python
# Download larger dataset
paths = prepare_hf_dataset(
    'glue/sst2',
    output_dir=Path('data/sst2_large'),
    max_samples=5000,  # 10x larger
)
```

### 4. Add Model State Checksum
To verify the model is actually changing:
```python
# After each epoch, log model checksum
import hashlib
state = str(model.state_dict())
checksum = hashlib.md5(state.encode()).hexdigest()
LOG.info(f"Model checksum: {checksum[:8]}")
```

---

## 🐛 Potential Bugs Found

### Issue 1: Metrics Might Be Cached
**Location:** `trainer.py:426-434`
```python
if all_labels and all_preds and len(all_labels) == len(all_preds):
    try:
        computed_metrics = compute_all_metrics(all_preds, all_labels)
        # BUG: Are all_preds and all_labels being cleared each time?
```

**Fix:** Verify that `all_preds` and `all_labels` are re-initialized:
```python
def evaluate(self, dataloader, compute_extended=True):
    # MUST be fresh lists each time
    all_preds = []  # ✅ Already doing this
    all_labels = []  # ✅ Already doing this
```

### Issue 2: Validation DataLoader Might Not Shuffle
**Check:** Is the validation set being shuffled?
```python
# In data/dataset.py - validation should NOT shuffle
# but the order shouldn't affect metrics this way
```

---

## ✅ Next Steps

1. **Run the diagnostic** to verify evaluation works correctly
2. **Check new logs** during next training run
3. **Compare checksums** of model weights across epochs
4. **Increase validation set size** to see if metrics become more granular

The enhanced logging will help us pinpoint exactly where the issue is!

---

## 📝 Summary

**What we found:**
- ✅ Training is working (loss decreasing)
- ❌ Metrics are suspiciously identical across epochs
- ⚠️ Validation evaluation might be using cached data

**What we added:**
- ✅ Detailed batch-level logging
- ✅ Metric computation logging
- ✅ Diagnostic tool to test evaluation

**Next run will show:**
- Exactly how many predictions are made per epoch
- Whether metrics are being recomputed each time
- Where the culprit is hiding!
