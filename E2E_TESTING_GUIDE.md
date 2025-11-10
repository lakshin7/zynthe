# End-to-End Workflow Testing Guide

## Prerequisites
1. Backend running: `cd ui/backend && python api.py` (Port 8765)
2. Frontend running: `cd ui && npm run dev` (Port 5173)

## Test Workflow

### 1. Settings Configuration ⚙️
**Objective:** Verify settings persist and apply correctly

**Steps:**
1. Click the **Settings** icon (gear) in top-right corner
2. Navigate to **Appearance** tab:
   - Change theme (Light/Dark/Auto)
   - Adjust glass intensity
   - Toggle accessibility options
3. Navigate to **Training** tab:
   - Set Device to "Apple Silicon (MPS)" or "Auto"
   - Change Batch Size to 8
   - Change Epochs to 5
   - Change Learning Rate to 3e-5
   - Update API Endpoint (should be http://localhost:8765)
4. Navigate to **Notifications** tab:
   - Enable all notification checkboxes
   - Enable sound notifications
5. Click **Save Settings**
6. Close modal and reopen to verify persistence

**Expected Results:**
- ✅ Settings persist across modal reopens
- ✅ Alert shows "Settings saved successfully!"
- ✅ Theme changes apply immediately
- ✅ Glass intensity adjusts visually

---

### 2. Dataset Upload 📁
**Objective:** Upload custom dataset and validate

**Test with Valid JSONL:**
1. Click **New Training** button
2. Navigate to **Dataset** step
3. Prepare a test file `test_dataset.jsonl`:
```jsonl
{"text": "This movie was amazing!", "label": 1}
{"text": "Terrible film, waste of time", "label": 0}
{"text": "Great acting and plot", "label": 1}
```
4. Drag and drop the file OR click "Choose File"
5. Verify success message shows
6. Check that dataset is auto-selected

**Test with Valid CSV:**
Create `test_dataset.csv`:
```csv
text,label
"Excellent product!",1
"Not worth the money",0
```
Repeat upload process

**Test Error Cases:**
1. Upload file > 100MB → Should show "File too large"
2. Upload `.txt` file → Should show "Invalid format"
3. Upload JSONL without 'text' field → Should show validation error

**Expected Results:**
- ✅ Valid files upload successfully
- ✅ Success message displays with sample count
- ✅ Dataset auto-selected in dropdown
- ✅ Invalid files show clear error messages

---

### 3. New Training Run 🚀
**Objective:** Start a full training pipeline

**Steps:**
1. Click **New Training** button
2. **Step 1 - Project Details:**
   - Name: "E2E Test Run"
   - Description: "End-to-end workflow test"
   - Tags: "test, e2e, demo"
3. **Step 2 - Dataset:**
   - Select uploaded dataset OR "IMDB Sample"
4. **Step 3 - Model Selection:**
   - Teacher: Select "distilbert-base-uncased"
   - **Student Model Configuration:**
     - Architecture: "DistilBERT"
     - Hidden Size: 512
     - Num Layers: 4
   - **Optimization Preset:** "Fast"
5. **Step 4 - Configuration:**
   - Device: Auto (should use settings default)
   - Batch Size: 8 (from settings)
   - Epochs: 5 (from settings)
   - Learning Rate: 3e-5 (from settings)
   - **Logging:**
     - Level: "INFO"
     - Save Checkpoints: Every 1 epoch
6. **Step 5 - Review:**
   - Verify all settings display correctly
   - Click **Start Training**

**Expected Results:**
- ✅ All steps navigate smoothly
- ✅ Settings defaults pre-populate fields
- ✅ Review shows accurate configuration
- ✅ Training starts successfully

---

### 4. Live Training Dashboard 📊
**Objective:** Monitor training in real-time

**Steps:**
1. After starting training, experiment card appears on Projects page
2. Click **View Live** button (cyan pulsing button)
3. Observe **Overview** tab:
   - Stage should show "Preflight" → "Distillation" progression
   - Progress bar updates
   - ETA estimates
4. Switch to **Metrics** tab:
   - Loss chart updates in real-time
   - Accuracy chart shows improvements
5. Switch to **Logs** tab:
   - Color-coded logs stream (errors in red, success in green)
   - Auto-scroll to bottom

**Testing Controls:**
1. Click **Pause** button
   - Button changes to "Resume"
   - Training pauses
2. Click **Resume** button
   - Training continues from where it stopped
3. Click **Save Checkpoint** button
   - Should trigger checkpoint save
4. Click **Stop** button
   - Confirm dialog appears
   - Training terminates

**Expected Results:**
- ✅ Metrics update every few seconds via WebSocket
- ✅ Logs stream continuously
- ✅ Charts render smoothly
- ✅ Pause/Resume works without data loss
- ✅ Stop terminates process immediately

---

### 5. View Completed Experiment 📋
**Objective:** Inspect completed training results

**Steps:**
1. Wait for training to complete OR stop manually
2. On Projects page, find the experiment card
3. Click the experiment card
4. **Metrics Tab:**
   - Verify final accuracy
   - Check distillation loss
   - View quantization metrics
5. **Logs Tab:**
   - Scroll through full training logs
   - Look for completion message
6. **Config Tab:**
   - Verify all configuration matches what was set
   - Check saved paths

**Expected Results:**
- ✅ All 5 stages show "Completed" status
- ✅ Final metrics displayed accurately
- ✅ Logs show full training history
- ✅ Config tab shows all parameters

---

### 6. Model Comparison 📊
**Objective:** Compare teacher vs student models

**Steps:**
1. Click **Compare Models** button on Projects page
2. Wait for models to load (should show completed experiments)
3. Select 2-3 models to compare:
   - Teacher model (if available)
   - Student model from E2E test
   - Quantized version (if available)
4. Observe comparison table:
   - Accuracy values (best highlighted in green)
   - Model sizes (smallest highlighted in green)
   - Inference times (fastest highlighted in green)
5. View trade-off visualizations:
   - Accuracy vs Size bars
   - Speed vs Accuracy bars

**Expected Results:**
- ✅ Models load from API
- ✅ Can select up to 3 models
- ✅ Comparison table shows all metrics
- ✅ Color coding indicates best values
- ✅ Trade-off charts render correctly
- ✅ Visualizations help understand compression benefits

---

### 7. WebSocket Persistence 🔌
**Objective:** Verify real-time updates don't interrupt UX

**Steps:**
1. Start a training run
2. On Projects page, scroll down
3. Wait for WebSocket update (metrics change)
4. Verify scroll position maintained
5. Open Project Details modal while training runs
6. Check that modal updates don't close unexpectedly

**Expected Results:**
- ✅ No scroll jumping during updates
- ✅ Smooth real-time data flow
- ✅ Modals stay open during background updates

---

### 8. Dataset Management 🗂️
**Objective:** Verify dataset lifecycle

**Steps:**
1. Go to Settings → Training tab
2. Note current dataset list
3. Upload new dataset via New Training modal
4. Backend endpoint test:
   ```bash
   curl http://localhost:8765/api/datasets
   ```
5. Verify new dataset appears in response
6. Delete custom dataset:
   ```bash
   curl -X DELETE http://localhost:8765/api/dataset/test_dataset
   ```

**Expected Results:**
- ✅ GET /api/datasets returns all datasets
- ✅ Uploaded datasets persist between sessions
- ✅ DELETE removes custom datasets only
- ✅ Built-in datasets cannot be deleted

---

## Performance Tests

### 9. Concurrent Training Runs
**Objective:** Test multiple training processes

**Steps:**
1. Start Training Run #1
2. Start Training Run #2 (different config)
3. Both should run simultaneously
4. Verify WebSocket broadcasts both correctly
5. Open "View Live" for each
6. Confirm no data cross-contamination

**Expected Results:**
- ✅ Multiple trainings run in parallel
- ✅ Separate logs and metrics
- ✅ No process interference

---

### 10. Error Handling 🚨
**Objective:** Verify graceful error handling

**Test Cases:**
1. **Backend Offline:**
   - Stop backend (`Ctrl+C` in api.py terminal)
   - Try to start training
   - Should show error message
   - Restart backend
   - Should reconnect automatically

2. **Invalid Config:**
   - Set batch size to 0
   - Try to start training
   - Should show validation error

3. **File Upload Errors:**
   - Upload corrupt JSONL
   - Should show parsing error

**Expected Results:**
- ✅ Clear error messages
- ✅ No crashes or blank screens
- ✅ Recovery after backend restart

---

## Checklist Summary

Use this checklist to track testing progress:

- [ ] Settings persist and apply correctly
- [ ] Dataset upload with JSONL works
- [ ] Dataset upload with CSV works
- [ ] Invalid file uploads show errors
- [ ] Training starts with correct config
- [ ] Settings defaults pre-populate
- [ ] Live dashboard shows real-time updates
- [ ] Pause/Resume controls work
- [ ] Stop button terminates training
- [ ] Completed experiments display correctly
- [ ] Model comparison loads and displays
- [ ] Trade-off visualizations render
- [ ] WebSocket updates don't interrupt scrolling
- [ ] Multiple trainings run concurrently
- [ ] Error handling works gracefully
- [ ] Backend restart reconnects automatically

---

## Known Issues & Limitations

1. **Checkpoint functionality** - Currently a stub, needs implementation
2. **Inference time metrics** - May be estimated, not actual measurements
3. **Browser notifications** - Require explicit permission
4. **Large file uploads** - May timeout on slow connections

---

## Success Criteria

**The E2E test passes if:**
1. All workflow steps complete without errors
2. Data persists correctly (settings, experiments, datasets)
3. Real-time features work (WebSocket, live updates)
4. UI remains responsive throughout
5. Error messages are clear and helpful
6. All modals open/close correctly
7. No console errors in browser DevTools

---

## Reporting Issues

If you encounter bugs during testing, note:
1. **Step where error occurred**
2. **Expected vs actual behavior**
3. **Browser console errors** (F12 → Console)
4. **Backend terminal output**
5. **Network tab** (F12 → Network) for API failures

Good luck with testing! 🚀
