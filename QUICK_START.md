# 🚀 Quick Start Guide - Zynthe E2E Testing

## Prerequisites

1. **Python 3.8+** installed
2. **Node.js 16+** installed
3. **Virtual environment** (already exists as `.venv`)

---

## Step 1: Start the Application

### Option A: Use the Start Script (Recommended)
```bash
# From the project root directory
./start.sh
```

This will:
- Activate virtual environment
- Install all dependencies
- Start backend on port 8765
- Start frontend on port 5173

### Option B: Manual Start

#### Terminal 1 - Backend:
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (first time only)
pip install -r ui/backend/requirements.txt

# Start backend
cd ui/backend
python api.py
```

#### Terminal 2 - Frontend:
```bash
# Install dependencies (first time only)
cd ui
npm install

# Start frontend
npm run dev
```

---

## Step 2: Open the Application

Once both services are running, open your browser to:
**http://localhost:5173**

You should see the Zynthe dashboard with the glass morphism UI.

---

## Step 3: Follow the E2E Test Flow

### ✅ **Test 1: Settings (2 mins)**
1. Click the **⚙️ Settings** icon (top-right)
2. Navigate through all 3 tabs:
   - **Appearance:** Change theme to Dark
   - **Training:** Set batch size to 8, epochs to 3
   - **Notifications:** Enable all checkboxes
3. Click **Save Settings**
4. Close and reopen to verify persistence

---

### ✅ **Test 2: Dataset Upload (3 mins)**
1. Click **➕ New Training** button
2. Fill Step 1 (Project Details):
   - Name: "E2E Test"
   - Description: "Testing the complete workflow"
   - Tags: "test"
3. Proceed to Step 2 (Dataset)
4. **Upload test_dataset.jsonl:**
   - Drag the file into the upload zone, OR
   - Click "Choose File" and select it
5. Verify success message appears
6. Note that the dataset is auto-selected

---

### ✅ **Test 3: Start Training (5 mins)**
Continue from the New Training modal:

1. **Step 3 - Model Selection:**
   - Teacher: `distilbert-base-uncased`
   - Student Architecture: `DistilBERT`
   - Hidden Size: `512`
   - Num Layers: `4`
   - Optimization: `Fast`

2. **Step 4 - Configuration:**
   - Device: `Auto` (should use your settings default)
   - Batch Size: `8` (from settings)
   - Epochs: `3` (from settings)
   - Learning Rate: `3e-5`
   - Log Level: `INFO`
   - Checkpoint: Every `1` epoch

3. **Step 5 - Review:**
   - Verify all settings are correct
   - Click **🚀 Start Training**

The modal will close and you'll see a new experiment card appear.

---

### ✅ **Test 4: Live Dashboard (10 mins)**
1. Find your running experiment card
2. Click the **📊 View Live** button (cyan, pulsing)
3. **Overview Tab:**
   - Watch the stage progress: Preflight → Distillation → Quantization
   - Observe progress bar updates
   - Note the ETA
4. **Metrics Tab:**
   - Loss chart should update every few seconds
   - Accuracy chart shows improvement
5. **Logs Tab:**
   - Color-coded logs streaming
   - Red for errors, green for success, blue for info
   - Auto-scrolls to bottom

---

### ✅ **Test 5: Training Controls (5 mins)**
While in the Live Dashboard:

1. Click **⏸️ Pause** button
   - Button changes to "▶️ Resume"
   - Metrics stop updating
2. Wait 10 seconds
3. Click **▶️ Resume** button
   - Training continues
   - Metrics resume
4. *(Optional)* Click **💾 Save Checkpoint**
5. *(Optional)* Click **⏹️ Stop** → Confirm

---

### ✅ **Test 6: View Completed Experiment (3 mins)**
After training completes (or you stop it):

1. Close the Live Dashboard
2. Find the experiment card on Projects page
3. Click the card to open **Project Details**
4. Check all 3 tabs:
   - **Metrics:** Final accuracy, loss, F1 score
   - **Logs:** Full training history
   - **Config:** All parameters used

---

### ✅ **Test 7: Model Comparison (5 mins)**
1. Click **📊 Compare Models** button (top of Projects page)
2. Wait for models to load
3. Select 2-3 completed experiments
4. Observe:
   - Comparison table with all metrics
   - Green highlights on best values
   - Red highlights on worst values
   - Trade-off charts:
     - Accuracy vs Size
     - Speed vs Accuracy

---

## Step 4: Fill Out the Test Report

Open and complete:
**`E2E_TEST_REPORT.md`**

Check off each test item and note any issues.

---

## Troubleshooting

### Backend won't start:
```bash
# Check if port 8765 is already in use
lsof -i :8765

# If yes, kill the process
kill -9 <PID>
```

### Frontend won't start:
```bash
# Check if port 5173 is already in use
lsof -i :5173

# If yes, kill the process
kill -9 <PID>
```

### Dependencies missing:
```bash
# Backend
source .venv/bin/activate
pip install fastapi uvicorn websockets pyyaml

# Frontend
cd ui
npm install
```

### Training doesn't start:
- Check browser console (F12) for errors
- Check backend terminal for errors
- Verify `experiments/` directory exists

### WebSocket not connecting:
- Check if backend is running
- Check browser console for WebSocket errors
- Try refreshing the page

---

## Quick Status Check

Run this to verify everything is working:
```bash
./test_e2e.sh
```

This will check:
- ✅ Backend running on port 8765
- ✅ Frontend running on port 5173
- ✅ API endpoints responding
- ✅ Number of experiments, models, datasets

---

## Expected Test Duration

| Test | Duration |
|------|----------|
| Settings | 2 mins |
| Dataset Upload | 3 mins |
| Start Training | 5 mins |
| Live Dashboard | 10 mins |
| Training Controls | 5 mins |
| Completed Experiment | 3 mins |
| Model Comparison | 5 mins |
| **Total** | **~35 mins** |

*Training duration will vary based on your hardware*

---

## Success Criteria

✅ **All tests pass if:**
1. All steps complete without errors
2. Data persists (settings, experiments)
3. Real-time updates work smoothly
4. UI remains responsive
5. No console errors
6. WebSocket connects successfully

---

## Next Steps After Testing

1. Fill out `E2E_TEST_REPORT.md`
2. Note any bugs or issues
3. Take screenshots of any errors
4. Review browser console logs
5. Check backend terminal output

---

## Need Help?

If you encounter issues:
1. Check `E2E_TESTING_GUIDE.md` for detailed steps
2. Check `FEATURE_SUMMARY.md` for feature documentation
3. Review browser DevTools Console (F12)
4. Check backend terminal output for errors

---

**Happy Testing! 🎉**
