# Zynthe E2E Test Report

**Date:** _____________
**Tester:** _____________
**Environment:** macOS / Browser: _____________

---

## Test Results Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Backend Startup | ⬜ Pass / ⬜ Fail | |
| Frontend Startup | ⬜ Pass / ⬜ Fail | |
| Settings Configuration | ⬜ Pass / ⬜ Fail | |
| Dataset Upload | ⬜ Pass / ⬜ Fail | |
| New Training Run | ⬜ Pass / ⬜ Fail | |
| Live Dashboard | ⬜ Pass / ⬜ Fail | |
| Training Controls | ⬜ Pass / ⬜ Fail | |
| Completed Experiment View | ⬜ Pass / ⬜ Fail | |
| Model Comparison | ⬜ Pass / ⬜ Fail | |
| WebSocket Updates | ⬜ Pass / ⬜ Fail | |

**Overall Status:** ⬜ PASS / ⬜ FAIL

---

## Detailed Test Results

### 1. Backend Startup ✅
**Command:** `./start.sh` or `cd ui/backend && python api.py`

- ⬜ Backend starts without errors
- ⬜ Listens on port 8765
- ⬜ Responds to `curl http://localhost:8765/`
- ⬜ API documentation accessible at http://localhost:8765/docs

**Errors/Issues:**
```
(Add any error messages here)
```

---

### 2. Frontend Startup 🌐
**Command:** `cd ui && npm run dev`

- ⬜ Frontend builds successfully
- ⬜ Runs on port 5173
- ⬜ Opens in browser automatically
- ⬜ UI loads without console errors

**Console Errors:**
```
(Check browser DevTools Console - F12)
```

---

### 3. Settings Configuration ⚙️
**Location:** Click gear icon in top-right

#### Appearance Tab:
- ⬜ Theme changes (Light/Dark/Auto) apply immediately
- ⬜ Glass intensity slider works
- ⬜ Reduce transparency checkbox works
- ⬜ Increase contrast checkbox works

#### Training Tab:
- ⬜ Device dropdown has options (Auto/CPU/GPU/MPS)
- ⬜ Batch size input accepts numbers
- ⬜ Epochs input accepts numbers
- ⬜ Learning rate input accepts decimals
- ⬜ API endpoint field is editable

#### Notifications Tab:
- ⬜ All checkboxes toggle correctly
- ⬜ Settings save successfully
- ⬜ Alert shows "Settings saved successfully!"

#### Persistence:
- ⬜ Close and reopen modal
- ⬜ Settings persist correctly

**Issues:**
```

```

---

### 4. Dataset Upload 📁
**Location:** New Training → Dataset Step

#### Valid JSONL Upload:
**File:** `test_dataset.jsonl`
- ⬜ Drag-and-drop zone highlights on drag
- ⬜ File picker opens on "Choose File" click
- ⬜ Upload completes successfully
- ⬜ Success message shows sample count
- ⬜ Dataset auto-selected in dropdown

#### Valid CSV Upload:
- ⬜ CSV file uploads successfully
- ⬜ Validation passes

#### Error Handling:
- ⬜ Large file (>100MB) shows error
- ⬜ Invalid format (.txt) shows error
- ⬜ Missing 'text' field shows validation error
- ⬜ Missing 'label' field shows validation error

**Issues:**
```

```

---

### 5. New Training Run 🚀
**Location:** Click "New Training" button

#### Step 1 - Project Details:
- ⬜ Name input works
- ⬜ Description textarea works
- ⬜ Tags input works
- ⬜ "Next" button advances to step 2

#### Step 2 - Dataset:
- ⬜ Dataset dropdown shows uploaded datasets
- ⬜ Built-in datasets appear (IMDB Sample, etc.)
- ⬜ File upload section works

#### Step 3 - Model Selection:
- ⬜ Teacher model dropdown populated
- ⬜ Student architecture selection works
- ⬜ Hidden size input accepts numbers
- ⬜ Num layers input accepts numbers
- ⬜ Optimization presets (Fast/Balanced/Quality) work

#### Step 4 - Configuration:
- ⬜ Settings defaults pre-populate
- ⬜ Device dropdown works
- ⬜ Batch size, epochs, learning rate editable
- ⬜ Log level dropdown works
- ⬜ Checkpoint frequency input works

#### Step 5 - Review:
- ⬜ All configuration displays correctly
- ⬜ "Start Training" button enabled
- ⬜ Clicking starts training

**Training Start:**
- ⬜ Modal closes after start
- ⬜ New experiment card appears on Projects page
- ⬜ Experiment shows "Running" status

**Issues:**
```

```

---

### 6. Live Training Dashboard 📊
**Location:** Click "View Live" on running experiment

#### Dashboard Opens:
- ⬜ Modal opens successfully
- ⬜ Title shows experiment name
- ⬜ 3 tabs visible (Overview, Metrics, Logs)

#### Overview Tab:
- ⬜ Stage indicator shows current stage
- ⬜ Progress bar updates
- ⬜ ETA displays and updates
- ⬜ Stage transitions (Preflight → Distillation → etc.)

#### Metrics Tab:
- ⬜ Loss chart renders
- ⬜ Accuracy chart renders
- ⬜ Charts update in real-time (via WebSocket)
- ⬜ Data points appear as training progresses

#### Logs Tab:
- ⬜ Logs stream in real-time
- ⬜ Color coding works:
  - Red for errors
  - Orange for warnings
  - Green for success
  - Blue for info
- ⬜ Auto-scrolls to bottom
- ⬜ Timestamps visible

**Issues:**
```

```

---

### 7. Training Controls 🎮
**Location:** Live Training Dashboard

#### Pause/Resume:
- ⬜ Click "Pause" button
- ⬜ Button changes to "Resume"
- ⬜ Training actually pauses (metrics stop updating)
- ⬜ Click "Resume" button
- ⬜ Training continues from where it stopped
- ⬜ Metrics resume updating

#### Stop:
- ⬜ Click "Stop" button
- ⬜ Confirmation dialog appears
- ⬜ Click "OK" to confirm
- ⬜ Training terminates
- ⬜ Dashboard closes or shows stopped state

#### Save Checkpoint:
- ⬜ Click "Save Checkpoint" button
- ⬜ Action completes (may show notification)

**Backend Verification:**
```bash
# Check if process actually paused/stopped
ps aux | grep "app/main.py"
```

**Issues:**
```

```

---

### 8. Completed Experiment View 📋
**Location:** Click experiment card after completion

#### Experiment Card:
- ⬜ All 5 stages show "Completed" status (green checkmarks)
- ⬜ Card shows completion time
- ⬜ No "View Live" button (training finished)

#### Details Modal - Metrics Tab:
- ⬜ Final accuracy displayed
- ⬜ Final loss displayed
- ⬜ F1 score, precision, recall visible
- ⬜ Model size information

#### Details Modal - Logs Tab:
- ⬜ Full training logs available
- ⬜ Scrollable to review entire history
- ⬜ Shows completion message

#### Details Modal - Config Tab:
- ⬜ All configuration parameters displayed
- ⬜ Paths to saved models shown
- ⬜ Hyperparameters match what was set

**Issues:**
```

```

---

### 9. Model Comparison 📊
**Location:** Click "Compare Models" button on Projects page

#### Modal Opens:
- ⬜ Modal opens successfully
- ⬜ Loading spinner appears initially
- ⬜ Models load from API

#### Model Selection:
- ⬜ Available models displayed in grid
- ⬜ Can select up to 3 models
- ⬜ Selected models highlighted
- ⬜ 4th selection attempt disabled

#### Comparison Table:
- ⬜ Table appears after selection
- ⬜ Shows all metrics (Accuracy, F1, Precision, Recall, Loss)
- ⬜ Shows model size (MB)
- ⬜ Shows inference time (ms)
- ⬜ Shows parameter count (M)

#### Color Coding:
- ⬜ Best values highlighted in green
- ⬜ Worst values highlighted in red
- ⬜ Middle values in default color

#### Trade-off Visualizations:
- ⬜ "Accuracy vs Size" chart renders
- ⬜ "Speed vs Accuracy" chart renders
- ⬜ Progress bars display correctly

**API Test:**
```bash
curl http://localhost:8765/api/models/compare
```

**Issues:**
```

```

---

### 10. WebSocket Real-Time Updates 🔌
**Location:** Projects page while training runs

#### Behavior:
- ⬜ Experiment cards update automatically
- ⬜ No manual page refresh needed
- ⬜ Scroll position maintained during updates
- ⬜ No "jumping" or layout shifts

#### WebSocket Connection:
**Browser DevTools → Network → WS:**
- ⬜ WebSocket connection established to `ws://localhost:8765/ws`
- ⬜ Messages received (training_log, training_metrics, training_update)
- ⬜ No disconnections or errors

**Issues:**
```

```

---

## Performance Observations

### Speed:
- Page load time: _______ seconds
- Training start delay: _______ seconds
- WebSocket update latency: _______ ms
- Modal open/close speed: _______

### Resource Usage:
- Backend CPU: _______ %
- Backend Memory: _______ MB
- Frontend Memory: _______ MB
- Browser tab responsive: ⬜ Yes / ⬜ No

---

## Browser Compatibility

Tested in:
- ⬜ Chrome/Chromium (Version: _______)
- ⬜ Safari (Version: _______)
- ⬜ Firefox (Version: _______)
- ⬜ Edge (Version: _______)

---

## Known Issues Found

### Critical (Blocking):
1. 

### Major (Important but not blocking):
1. 

### Minor (Cosmetic or edge cases):
1. 

---

## Suggestions/Improvements

1. 

---

## Conclusion

**Overall Assessment:**
⬜ Ready for production
⬜ Needs minor fixes
⬜ Needs major fixes
⬜ Not ready

**Recommendation:**


**Tester Signature:** _________________ **Date:** _____________
