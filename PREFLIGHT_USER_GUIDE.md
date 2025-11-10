# 🚀 Preflight System - User Guide

## What is Preflight Validation?

Preflight validation checks **BEFORE** starting training to ensure:
- ✅ Models exist on HuggingFace
- ✅ Models are compatible with your device (Mac M2/MPS)
- ✅ Teacher and student models work well together
- ✅ Configuration is valid
- ✅ Data files are accessible

**Why is this important?**
- Saves time (no failed training after hours of waiting)
- Saves bandwidth (no downloading incompatible models)
- Provides clear guidance when something is wrong
- Flags missing safety rails (e.g., overfit guard or teacher warmup) before you hit "Start"

---

## 📋 Step-by-Step Usage

### Step 1: Start Zynthe
```bash
cd /Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit
./start-zynthe.sh
```

Wait for:
- ✅ Backend started on http://localhost:8765
- ✅ UI ready on http://localhost:5173

### Step 2: Create New Experiment
1. Open http://localhost:5173
2. Click "New Experiment" button
3. Fill in Steps 1-3:
   - **Step 1**: Experiment name and description
   - **Step 2**: Select dataset (or upload custom)
   - **Step 3**: Select teacher and student models

### Step 3: Use Debug Panel (Optional but Recommended)
In **Step 4 - Preflight Validation**, you'll see a debug panel at the top:

```
🔍 Preflight Debug Panel
[ Show Details ▼ ]
```

Click **"Show Details"** to expand, then click **"Test Connection"**:

**What it checks:**
- Backend Health (http://localhost:8765/health)
- Device Info (MPS/CUDA/CPU detection)
- HuggingFace Token Status
- Selected Models

**Expected Output:**
```
✓ Backend: healthy (MPS device available)
✓ Device: mps
✓ HF Token: Configured
✓ Teacher: bert-base-uncased
✓ Student: distilbert-base-uncased
```

### Step 4: Run Preflight Validation
Click the **"Run Preflight"** button.

**What happens:**
1. Shows loading spinner
2. Validates configuration (Phase 1)
3. Validates models on HuggingFace (Phase 2)
4. Shows results in 2-4 seconds

When the new overfit guard or teacher warmup settings are disabled, the results banner includes an explicit warning so you can fix the config before training.

---

## 🎨 Understanding the Results

### ✅ Success (Green)
```
✓ Preflight Validation Passed!

Teacher: bert-base-uncased (438.5 MB)
Student: distilbert-base-uncased (267.8 MB)
Compression: 61% (Good!)
Device: MPS (Compatible)

[Next Step →]
```

## 🛡️ Guard Rails Against Overfitting

- **Overfit Guard** — Enabled by default through `train.overfit_guard`. It watches the training/validation gap and, when confidence is high, automatically pauses (mode `early_stop`) to keep the best checkpoint clean. Preflight warns if you disable it.
- **Adaptive Mitigation** — `train.overfit_mitigation` stages proactive regularization (stronger augmentation, higher dropout, LR/weight decay tweaks) before the guard halts. Watch the console for `OVERFIT-MITIGATION` messages. Preflight flags when this safety net is off.
- **Teacher Warmup** — Set `train.train_teacher: true` to fine-tune the teacher for a few epochs before distillation. Preflight highlights when this is off so you can avoid distilling from an unfocused teacher.
- After a run, open `training_health.json` to see overfit guard events, mitigation steps that fired, and recommendations.

**Action**: Click "Next Step" to continue to training!

---

### ❌ Error (Red)
```
✗ Preflight Validation Failed

Teacher Model: fake-model-123
  ✗ Model not found on HuggingFace Hub

Suggested Alternatives:
  • bert-base-uncased (5M downloads)
  • roberta-base (2.5M downloads)
  • albert-base-v2 (1M downloads)

[Try Another Model]
```

**Action**: 
1. Click "Previous" to go back to Step 3
2. Select one of the suggested alternatives
3. Run preflight again

---

### ⚠️ Warning (Yellow)
```
⚠ Preflight Validation Passed with Warnings

Teacher: bert-large-uncased (1.3 GB)
  ⚠ Large model may take significant time to download
  ⚠ Requires at least 8GB RAM

Device: MPS (Compatible)

Estimated Download Time: 5-10 minutes
Estimated Training Time: 2-3 hours

[Proceed with Caution] [Choose Different Model]
```

**Action**: 
- If you have good internet and patience: Click "Proceed with Caution"
- If you want faster training: Click "Choose Different Model"

---

## 🔍 Common Issues & Solutions

### Issue 1: "Network connection error"
**Symptoms:**
```
✗ Could not connect to backend
```

**Solutions:**
1. Check if backend is running:
   ```bash
   curl http://localhost:8765/health
   ```
2. If no response, restart:
   ```bash
   ./stop-zynthe.sh
   ./start-zynthe.sh
   ```

---

### Issue 2: "HuggingFace token not configured"
**Symptoms:**
```
⚠ HF Token: Not configured
✗ Cannot access private/gated models
```

**Solutions:**
1. Get token from https://huggingface.co/settings/tokens
2. Add to `.env` file:
   ```bash
   echo "HF_TOKEN=hf_your_token_here" >> .env
   ```
3. Restart backend

---

### Issue 3: "Model not compatible with device"
**Symptoms:**
```
✗ Model requires CUDA but you have MPS
```

**Solutions:**
1. Check suggested alternatives (will show MPS-compatible models)
2. Look for models with "distilbert" or "albert" (usually smaller and compatible)
3. Filter by model size < 1GB

---

### Issue 4: "Model not found"
**Symptoms:**
```
✗ Model 'albert-tiny' not found on HuggingFace Hub
```

**Solutions:**
1. Use the **Model Browser** in Step 3 to search for models
2. Make sure to use full model ID: `albert/albert-base-v2` (not just `albert-base`)
3. Check spelling and capitalization

---

## 💡 Pro Tips

### Tip 1: Use the Model Browser
In Step 3, use the search bar to find models:
- Type "distilbert" → See all DistilBERT variants
- Filter by task: "text-classification"
- Sort by downloads (most popular first)

### Tip 2: Start with Recommended Pairs
These pairs are tested and work well:
```
Teacher                 → Student
-----------------         ------------------
bert-base-uncased       → distilbert-base-uncased
roberta-base            → distilroberta-base  
albert-base-v2          → albert/albert-tiny-v2
```

### Tip 3: Check Model Sizes
Aim for compression ratio between 40-70%:
- **Too little compression** (<30%): Not worth distilling
- **Too much compression** (>80%): May lose too much knowledge

### Tip 4: Use Debug Panel First
Always click "Test Connection" before running preflight:
- Confirms backend is responsive
- Shows your device capabilities
- Verifies HF token status

---

## 📊 Understanding the Validation Report

### Full Report Structure
```json
{
  "valid": true,                    // Can proceed?
  "can_proceed": true,              // Alternative check
  
  "teacher": {
    "id": "bert-base-uncased",      // Model ID
    "exists": true,                  // Found on HF?
    "device_compatible": true,       // Works with MPS?
    "size_mb": 438.5,               // Download size
    "errors": [],                    // Blocking issues
    "warnings": [],                  // Non-blocking issues
    "alternatives": []               // Suggested alternatives
  },
  
  "student": { /* ... same structure ... */ },
  
  "compression_ratio": 0.61,        // 61% size reduction
  
  "device_info": {
    "current_device": "mps",         // Your device
    "available_devices": ["mps", "cpu"]
  }
}
```

---

## 🎯 Quick Reference

| Status | Meaning | Action |
|--------|---------|--------|
| ✓ Green | All checks passed | Proceed to training |
| ⚠ Yellow | Warnings but can proceed | Review warnings, decide |
| ✗ Red | Blocking errors | Fix issues or choose alternatives |

| Check | What it does |
|-------|-------------|
| Model Exists | Queries HuggingFace Hub |
| Device Compatible | Checks MPS/CUDA requirements |
| Size Check | Estimates download time |
| Pair Compatible | Verifies teacher > student |
| Architecture | Confirms distillation support |

---

## 🚨 Emergency Troubleshooting

If preflight keeps failing:

1. **Check logs:**
   ```bash
   tail -f /Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit/ui/backend/api.log
   ```

2. **Test endpoints manually:**
   ```bash
   # Health
   curl http://localhost:8765/health
   
   # Device
   curl http://localhost:8765/api/device/info
   
   # Validation
   curl -X POST http://localhost:8765/api/models/validate \
     -H "Content-Type: application/json" \
     -d '{"teacher_model":"bert-base-uncased","student_model":"distilbert-base-uncased"}'
   ```

3. **Restart everything:**
   ```bash
   ./stop-zynthe.sh
   sleep 2
   ./start-zynthe.sh
   ```

4. **Check Python environment:**
   ```bash
   source .venv/bin/activate
   python -c "from core.preflight.model_validator import ModelValidator; print('OK')"
   ```

---

## 📞 Getting Help

If you're still stuck:
1. Check PREFLIGHT_SYSTEM_COMPLETE.md for technical details
2. Run test script: `bash test_preflight_integration.sh`
3. Check backend logs for error traces
4. Verify HF_TOKEN is set in .env file

---

**Happy Distilling!** 🎉
