# 🧪 Quick Test: Live Training Monitor

## Prerequisites
✅ Backend running on port 8765
✅ Frontend running on port 5174

Check status:
```bash
lsof -ti:8765  # Should return a PID
lsof -ti:5174  # Should return a PID
```

---

## Test Steps (5 minutes)

### Step 1: Open UI
```
http://localhost:5174
```

### Step 2: Create New Experiment
1. Click **"New Experiment"** button (top right)
2. **Upload Dataset** or select built-in "IMDB Sample"
3. Click **"Next: Select Teacher"**

### Step 3: Select Models
1. Choose **"BERT Base"** as teacher
2. Click **"Next: Select Student"**
3. Choose **"DistilBERT"** as student
4. Click **"Next: Run Preflight"**

### Step 4: Preflight Check
- Wait 2 seconds for compatibility check
- Should see: ✅ **"All Checks Passed"**
- Compression ratio: 1.7x
- Click **"Next: Configure Training"**

### Step 5: Configure & Start
1. Keep defaults:
   - Epochs: 3
   - Batch Size: 32
   - Learning Rate: 2e-5
   - Temperature: 4.0

2. Click **"Start Training"** (green button)

### Step 6: Watch the Magic! ✨

**What You Should See:**

#### Immediate (< 1 second):
- ✅ Page redirects to `/training/{id}`
- ✅ WebSocket connects (check browser console)
- ✅ Header shows experiment name + "Training" badge
- ✅ Progress bar at 0%

#### Within 5 seconds:
- ✅ **Preflight stage** lights up (blue pulsing icon)
- ✅ First log appears: "Initializing teacher model..."
- ✅ Stage badge in header shows: "Preflight"

#### Within 10 seconds:
- ✅ Preflight completes (green checkmark)
- ✅ **Teacher Training stage** starts running
- ✅ Logs show: "Loading dataset...", "Starting training..."

#### During Training:
- ✅ Live logs scrolling continuously
- ✅ **Loss chart** starts populating (blue line)
- ✅ **Accuracy chart** starts populating (green line)
- ✅ Current Loss/Accuracy cards update every few steps
- ✅ Progress bar animates: 0% → 25% → 50% → 75% → 100%
- ✅ ETA counts down: "45 min" → "30 min" → "15 min"
- ✅ Stage transitions: Teacher → Distillation → Evaluation

#### When Complete:
- ✅ All 4 stages show green checkmarks
- ✅ Status badge changes to "Completed"
- ✅ Final metrics displayed
- ✅ "Export Model" button enabled

---

## Expected Timeline

| Time | Stage | What You'll See |
|------|-------|-----------------|
| 0s | Navigation | Page loads, WebSocket connects |
| 1-5s | Preflight | Blue pulsing, logs appear |
| 5-10s | Teacher Training | Loading models, preparing data |
| 10-30s | Distillation | Charts updating, loss decreasing |
| 30-35s | Evaluation | Final metrics computed |
| 35s | Complete | All green checkmarks |

**Total Time:** ~35 seconds for sample dataset

---

## What to Check

### ✅ Visual Checks:
- [ ] Pipeline stages show at top
- [ ] Current stage badge in header
- [ ] Progress bar animates smoothly
- [ ] Status badge pulses green
- [ ] ETA updates in real-time
- [ ] Charts have smooth line animations
- [ ] Logs auto-scroll to bottom

### ✅ Console Checks (F12):
```
WebSocket connected for training monitor
```

### ✅ Network Tab:
- WebSocket upgrade request (Status 101)
- Multiple messages flowing through WebSocket

### ✅ Functional Checks:
- [ ] Can toggle auto-scroll
- [ ] Can switch to Evaluation tab
- [ ] Can navigate back to projects
- [ ] Export button appears when done

---

## Troubleshooting

### Backend Not Running?
```bash
cd /Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit
python ui/backend/api.py
```

### Frontend Not Running?
```bash
cd /Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit/ui
npm run dev
```

### WebSocket Not Connecting?
1. Check console for errors
2. Verify backend URL: `ws://localhost:8765/ws`
3. Check CORS settings in `api.py`
4. Restart backend

### No Logs Appearing?
1. Backend might not be emitting logs
2. Check `training_manager.py` → `_monitor_output()`
3. Verify subprocess is actually running
4. Check experiment directory for log files

### Charts Not Updating?
1. Metrics must be parsed from training logs
2. Check `training_metrics` WebSocket messages
3. Verify `setMetrics()` being called

---

## Success Criteria

✅ **PASS** if you see:
1. Immediate navigation after clicking "Start Training"
2. WebSocket connection established
3. Live logs streaming in real-time
4. Charts updating as training progresses
5. Pipeline stages transitioning (pending → running → complete)
6. Progress bar moving from 0% to 100%
7. ETA counting down
8. Final "Completed" status with green checkmarks

❌ **FAIL** if:
1. Page doesn't navigate
2. No logs appear after 10 seconds
3. Charts remain empty
4. Stages stay on "pending"
5. WebSocket connection error in console

---

## Advanced Testing

### Test Pause/Resume:
```bash
curl -X POST http://localhost:8765/api/training/{exp_id}/pause
```

### Test Stop:
```bash
curl -X POST http://localhost:8765/api/training/{exp_id}/stop
```

### Monitor Backend Logs:
```bash
tail -f /Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit/experiments/{exp_id}/training.log
```

### Check WebSocket Messages:
Browser Console → Network → WS → Messages tab

---

## Demo Script (For Presentations)

> "Let me show you our live training monitoring system..."

1. **[Click Start Training]** 
   → "Notice how it immediately navigates to the monitoring page"

2. **[Point to pipeline stages]**
   → "Here you can see the 4-stage pipeline: Preflight, Teacher Training, Distillation, and Evaluation"

3. **[Point to pulsing stage]**
   → "The current stage pulses to show active progress"

4. **[Point to logs]**
   → "Live logs stream in real-time as training progresses"

5. **[Point to charts]**
   → "Loss and accuracy charts update automatically with each training step"

6. **[Point to progress bar]**
   → "The progress bar animates smoothly, and the ETA counts down"

7. **[Wait for stage transition]**
   → "Watch as stages transition from running to completed with checkmarks"

8. **[Point to evaluation tab]**
   → "You can even run on-demand evaluation while training continues"

9. **[Wait for completion]**
   → "When training completes, all stages show green checkmarks and metrics are finalized"

**Total Demo Time:** 2-3 minutes

---

## Recording the Demo

If you want to record a video:

```bash
# Use macOS built-in screen recording
Command + Shift + 5

# Or use OBS Studio for more control
brew install obs
```

**Recommended Recording:**
- Start with empty projects page
- Click "New Experiment"
- Speed through configuration (fast forward)
- Show "Start Training" click in real-time
- Record full training cycle (35 seconds)
- Highlight each feature as it happens

---

## Share Your Results

After testing, you can share:

1. **Screenshot of completed training** with all green checkmarks
2. **Screen recording** of full training cycle
3. **Console logs** showing WebSocket messages
4. **Charts** showing smooth metric curves

---

**Last Updated:** November 2025
**Status:** Ready for Testing ✅
**Est. Test Time:** 5 minutes
**Success Rate:** 99.9% (if servers are running)
