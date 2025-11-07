# 🚀 Quick Start: Download Progress Feature

## What's New?

You now get **real-time download progress** when models are being fetched from HuggingFace! 

No more wondering if your system is frozen during model downloads. 📊

## How to Use

### 1. Start a Training Experiment

Just use the system normally:

```bash
# Start the system
./start-zynthe.sh

# Or directly
cd ui && npm run dev
```

### 2. Create an Experiment

In the UI:
1. Go to **Create Experiment**
2. Select your models (e.g., `bert-base-uncased` → `distilbert-base-uncased`)
3. Configure dataset
4. Click **Start Training**

### 3. Watch the Progress!

The **Training Monitor** page now shows:

```
📥 Downloading Teacher Model: 45% (120MB at 3.5MB/s)
📥 Downloading Student Model: Pending
📚 Loading Dataset: Pending
⚙️ Initializing: Pending
✅ Preflight Check: Pending
🚀 Training: Pending
📊 Evaluation: Pending
```

Each stage shows:
- ✅ Status badge (Pending/Running/Complete)
- 📊 Progress bar (0-100%)
- 💬 Real-time message
- ⚡ Download speed (for downloads)

## First Time vs. Later Runs

### First Time (Models Need Download)
```
📥 Teacher: Downloading... (2-5 minutes)
   └─ 120.8MB at 3.5MB/s
📥 Student: Downloading... (2-5 minutes)  
   └─ 85.2MB at 3.2MB/s
```

**Total wait**: ~5-10 minutes (but you see progress!)

### Later Runs (Models Cached)
```
📥 Teacher: ✨ Cached (2 seconds)
📥 Student: ✨ Cached (2 seconds)
```

**Total wait**: ~4 seconds! 🚀

## What You'll See

### Download in Progress
```
┌────────────────────────────────────────┐
│ 📥 Downloading Teacher Model           │
│ ████████████░░░░░░░░░░░░░░░░ 30%      │
│ bert-base-uncased: 85MB (3.8MB/s)      │
└────────────────────────────────────────┘
```

### Model Cached (Fast Load)
```
┌────────────────────────────────────────┐
│ 📥 Downloading Teacher Model           │
│ ████████████████████████████ 90%       │
│ ✨ Found in cache, loading...          │
└────────────────────────────────────────┘
```

### Complete
```
┌────────────────────────────────────────┐
│ 📥 Downloading Teacher Model    ✅     │
│ ████████████████████████████ 100%      │
│ bert-base-uncased loaded successfully  │
└────────────────────────────────────────┘
```

## Notification Settings

Click the 🔔 bell icon in the top-right to:

- ✅ Enable browser notifications
- 🔊 Enable/disable completion sound
- 🧪 Test the notification sound

When training completes, you'll get:
- Browser notification (even if in another tab)
- Optional sound alert (pleasant two-tone chime)

## Testing the Feature

### Quick Test Script

Want to see the download progress in action?

```bash
# Run the test script
python test_download_progress.py
```

This will:
1. Check which models are cached
2. Download a small model (`prajjwal1/bert-tiny` - only 17MB)
3. Show progress messages in real-time

### Clear Cache to Test Download

Want to see download progress for a cached model?

```bash
# Clear the cache for a specific model
rm -rf ~/.cache/huggingface/hub/models--bert-base-uncased

# Now run training - you'll see download progress!
```

## Troubleshooting

### "No progress showing"

**Check**:
1. Is the UI connected? (Look for 🟢 in top-right)
2. Is the backend running? (`http://localhost:8765`)
3. Check browser console for errors

**Fix**:
```bash
# Restart the system
./stop-zynthe.sh
./start-zynthe.sh
```

### "Download seems stuck"

**Possible causes**:
- Slow internet connection
- HuggingFace Hub is slow
- Model is very large

**Check**:
- Look at the speed (MB/s) - is it reasonable?
- Check your internet connection
- Large models (>1GB) take longer

**Normal speeds**:
- 🟢 Good: 3-5 MB/s
- 🟡 OK: 1-3 MB/s
- 🔴 Slow: < 1 MB/s

### "Models re-downloading every time"

**Check cache**:
```bash
# Check if models are cached
ls -lh ~/.cache/huggingface/hub/
```

**Fix**: Make sure you have write permissions to cache directory.

### "Progress percentage seems off"

This is **normal**! Here's why:

- We don't know exact model size until download completes
- Progress uses logarithmic curve for better UX
- Feels faster initially (10% → 50% is quick)
- Last 5% reserved for loading into memory

**This is by design** for better user experience!

## Where Download Files Go

Models are cached in:
```
~/.cache/huggingface/hub/
└── models--bert-base-uncased/
    ├── blobs/
    ├── refs/
    └── snapshots/
```

**Cache size**: 
- Small model: 50-100 MB
- Medium model: 200-500 MB
- Large model: 1-3 GB

**Clear cache** if running low on disk space:
```bash
# Clear specific model
rm -rf ~/.cache/huggingface/hub/models--<model-name>

# Clear all models (WARNING: will re-download everything)
rm -rf ~/.cache/huggingface/hub/
```

## FAQ

**Q: Why does it show "Downloading" even if cached?**

A: The stage is called "Downloading Teacher Model" but the message will say "Found in cache" if it's already downloaded. We kept the stage name generic.

**Q: Can I cancel a download?**

A: Yes, just stop the training experiment. The partial download will be cleaned up or resumed next time.

**Q: Will it resume interrupted downloads?**

A: Yes! HuggingFace's download library automatically resumes partial downloads.

**Q: How can I speed up downloads?**

A: 
- Use faster internet connection
- Download during off-peak hours
- Consider using HuggingFace CDN mirrors (if available in your region)

**Q: Why are the progress bars animated?**

A: Visual feedback to show the system is active and not frozen!

## Tips

### 💡 Tip 1: Warm Up Cache
Download models before heavy work session:

```bash
# Pre-download models
python test_download_progress.py
```

### 💡 Tip 2: Use Smaller Models for Testing
Test with tiny models first:
- `prajjwal1/bert-tiny` (17 MB)
- `google/mobilebert-uncased` (95 MB)
- `distilbert-base-uncased` (267 MB)

### 💡 Tip 3: Monitor Disk Space
Check cache size regularly:

```bash
du -sh ~/.cache/huggingface/hub/
```

### 💡 Tip 4: Keep UI Tab Open
The UI updates in real-time only when the tab is active or in background. Browser notifications work even when tab is hidden!

## What Happens Under the Hood

1. **Cache Check**: System checks if model is in `~/.cache/huggingface/`
2. **Download Start**: If not cached, starts download from HuggingFace Hub
3. **Monitor**: Background thread watches cache directory size
4. **Progress**: Calculates progress based on bytes downloaded
5. **Speed**: Calculates MB/s based on download time
6. **Complete**: Model loaded into memory, ready for training

**All automatic** - no configuration needed! 🎉

## Get Started Now!

```bash
# 1. Start the system
./start-zynthe.sh

# 2. Open browser
open http://localhost:5173

# 3. Create experiment

# 4. Watch the progress! 📊
```

---

**That's it!** Enjoy your new download progress feature! 🚀

If you have any questions or issues, check the full documentation:
- `DOWNLOAD_PROGRESS_COMPLETE.md` - Technical details
- `DOWNLOAD_PROGRESS_VISUAL_GUIDE.md` - Visual walkthrough
- `DOWNLOAD_PROGRESS_SUMMARY.md` - Complete feature summary
