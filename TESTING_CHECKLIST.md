# ✅ Zynthe E2E Testing Checklist

**Quick reference for testing all features**

---

## 🚀 Pre-Test Setup

- [ ] Backend running: `./start.sh` or manual start
- [ ] Frontend running: Browser at http://localhost:5173
- [ ] Test file ready: `test_dataset.jsonl` exists
- [ ] Browser DevTools open (F12) for console monitoring

---

## Test 1: Settings ⚙️ (2 min)

### Actions:
- [ ] Click gear icon (top-right)
- [ ] **Appearance tab:** Change theme to Dark
- [ ] **Appearance tab:** Adjust glass intensity slider
- [ ] **Training tab:** Set batch size = 8
- [ ] **Training tab:** Set epochs = 3
- [ ] **Training tab:** Set learning rate = 3e-5
- [ ] **Notifications tab:** Enable all checkboxes
- [ ] Click "Save Settings"
- [ ] See "Settings saved successfully!" alert
- [ ] Close modal, reopen, verify settings persisted

### Expected:
✅ Theme changes immediately
✅ All settings save to localStorage
✅ Settings persist after modal close/reopen

---

## Test 2: Dataset Upload 📁 (3 min)

### Actions:
- [ ] Click "New Training" button
- [ ] Fill Step 1: Name = "E2E Test", Description = "Test run"
- [ ] Click "Next" to Step 2 (Dataset)
- [ ] Drag `test_dataset.jsonl` to upload zone OR
- [ ] Click "Choose File" and select the file
- [ ] See success message with sample count
- [ ] Verify dataset auto-selected in dropdown

### Expected:
✅ File uploads successfully
✅ Success message shows "10 samples"
✅ Dataset appears and is selected

---

## Test 3: Start Training 🚀 (5 min)

### Actions:
- [ ] **Step 3 - Model:**
  - [ ] Teacher: `distilbert-base-uncased`
  - [ ] Student Architecture: `DistilBERT`
  - [ ] Hidden Size: `512`
  - [ ] Num Layers: `4`
  - [ ] Optimization: `Fast`
- [ ] **Step 4 - Config:**
  - [ ] Device: `Auto`
  - [ ] Batch Size: `8` (from settings)
  - [ ] Epochs: `3` (from settings)
  - [ ] Learning Rate: `3e-5` (from settings)
  - [ ] Log Level: `INFO`
  - [ ] Checkpoint: `1` epoch
- [ ] **Step 5 - Review:**
  - [ ] Verify all settings correct
  - [ ] Click "Start Training"
- [ ] Modal closes
- [ ] New experiment card appears

### Expected:
✅ All steps navigate smoothly
✅ Settings defaults pre-populate
✅ Training starts after clicking "Start Training"
✅ Experiment card shows "Running" status

---

## Test 4: Live Dashboard 📊 (10 min)

### Actions:
- [ ] Click "View Live" button (cyan, pulsing)
- [ ] **Overview tab:**
  - [ ] See current stage (Preflight → Distillation)
  - [ ] Progress bar updates
  - [ ] ETA displays
- [ ] **Metrics tab:**
  - [ ] Loss chart renders
  - [ ] Accuracy chart renders
  - [ ] Charts update every few seconds
- [ ] **Logs tab:**
  - [ ] Logs stream in real-time
  - [ ] Color coding works (red errors, green success, blue info)
  - [ ] Auto-scrolls to bottom

### Expected:
✅ Dashboard opens successfully
✅ Metrics update via WebSocket
✅ Logs stream continuously
✅ Charts render and update

---

## Test 5: Training Controls 🎮 (5 min)

### Actions:
- [ ] Click "Pause" button
- [ ] Button changes to "Resume"
- [ ] Metrics stop updating
- [ ] Wait 10 seconds
- [ ] Click "Resume" button
- [ ] Metrics resume updating
- [ ] *(Optional)* Click "Save Checkpoint"
- [ ] *(Optional)* Click "Stop" → Confirm

### Expected:
✅ Pause actually stops training
✅ Resume continues from where paused
✅ Checkpoint button works
✅ Stop terminates training

---

## Test 6: Completed Experiment 📋 (3 min)

### Actions:
- [ ] Wait for training to complete OR stop it
- [ ] Close Live Dashboard
- [ ] Find experiment card on Projects page
- [ ] Click card to open Project Details
- [ ] **Metrics tab:**
  - [ ] Final accuracy visible
  - [ ] Final loss visible
  - [ ] Other metrics (F1, precision, recall)
- [ ] **Logs tab:**
  - [ ] Full log history available
  - [ ] Can scroll through all logs
- [ ] **Config tab:**
  - [ ] All parameters displayed
  - [ ] Model paths shown

### Expected:
✅ All 5 stages show "Completed"
✅ Final metrics accurate
✅ Full logs accessible
✅ Config matches what was set

---

## Test 7: Model Comparison 📊 (5 min)

### Actions:
- [ ] Click "Compare Models" button (top of page)
- [ ] Wait for models to load
- [ ] Select 2-3 models from the grid
- [ ] See comparison table appear
- [ ] **Check table shows:**
  - [ ] Accuracy (best in green)
  - [ ] F1 Score
  - [ ] Precision
  - [ ] Recall
  - [ ] Loss (lowest in green)
  - [ ] Model Size (smallest in green)
  - [ ] Inference Time (fastest in green)
  - [ ] Parameters
- [ ] **Check visualizations:**
  - [ ] "Accuracy vs Size" bars
  - [ ] "Speed vs Accuracy" bars

### Expected:
✅ Models load from API
✅ Can select up to 3 models
✅ Comparison table renders correctly
✅ Color coding highlights best values
✅ Trade-off charts display

---

## Test 8: WebSocket Persistence 🔌 (2 min)

### Actions:
- [ ] Start a training (use previous steps)
- [ ] On Projects page, scroll down
- [ ] Wait for metrics to update (5-10 seconds)
- [ ] Verify scroll position maintained
- [ ] Open Project Details modal
- [ ] Keep modal open while training runs
- [ ] Verify modal doesn't close unexpectedly

### Expected:
✅ No scroll jumping during updates
✅ Smooth real-time updates
✅ Modals stay open during background updates
✅ Page remains responsive

---

## 🐛 Error Testing

### Test Invalid Uploads:
- [ ] Try uploading .txt file → Should show "Invalid format"
- [ ] Try uploading file > 100MB → Should show "File too large"
- [ ] Try uploading JSONL without 'text' field → Validation error

### Test Backend Offline:
- [ ] Stop backend (Ctrl+C)
- [ ] Try to start training → Should show error
- [ ] Restart backend
- [ ] Try again → Should work

---

## 📊 Final Verification

### Browser Console:
- [ ] No red errors in Console tab (F12)
- [ ] WebSocket connected successfully
- [ ] No 404 or 500 API errors

### Backend Terminal:
- [ ] No Python exceptions
- [ ] WebSocket connections logged
- [ ] Training process starts successfully

### Data Persistence:
- [ ] Settings saved to localStorage
- [ ] Experiments saved to `experiments/` folder
- [ ] Uploaded datasets in `data/` folder

---

## ✅ Test Complete!

**All boxes checked?** Congratulations! 🎉

Fill out: **`E2E_TEST_REPORT.md`** with your findings.

---

## 🐛 Found Issues?

Note in the report:
- What you were doing
- What happened
- What you expected
- Browser console errors
- Backend terminal output

---

**Happy Testing!** 🚀
