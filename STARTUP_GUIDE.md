# Zynthe Desktop App - Startup Scripts

## Quick Reference

### 🚀 Production Startup (Recommended)

```bash
./start-zynthe.sh
```

**What it does:**
- ✅ Checks if virtual environment exists
- ✅ Checks if node_modules are installed
- ✅ Starts FastAPI backend on port 8765
- ✅ Waits for backend to be ready
- ✅ Starts Electron frontend on port 5173
- ✅ Shows status with colors and emojis
- ✅ Handles Ctrl+C gracefully (cleans up both processes)

**Perfect for:** First-time users, production, demo

---

### ⚡ Development Startup (Fast)

```bash
./dev.sh
```

**What it does:**
- 🏃 Skips all checks (assumes setup is done)
- 🏃 Starts backend immediately
- 🏃 Starts frontend immediately
- ✅ Handles Ctrl+C gracefully

**Perfect for:** Daily development, quick iterations

---

### 🔧 Manual Startup (Advanced)

If you need more control:

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
python ui/backend/api.py
```

**Terminal 2 - Frontend:**
```bash
cd ui
npm run start
```

**Perfect for:** Debugging, separate logs, custom ports

---

## First Time Setup

Before running any script:

1. **Install Python dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install Node dependencies:**
   ```bash
   cd ui
   npm install
   cd ..
   ```

3. **Run Zynthe:**
   ```bash
   ./start-zynthe.sh
   ```

---

## Troubleshooting

### Port 8765 already in use
```bash
# Kill existing backend
lsof -ti:8765 | xargs kill -9

# Then restart
./start-zynthe.sh
```

### Port 5173 already in use
```bash
# Kill existing frontend
lsof -ti:5173 | xargs kill -9

# Then restart
./start-zynthe.sh
```

### Clean restart
```bash
# Kill all processes
pkill -f "vite|electron|python ui/backend"

# Then start fresh
./start-zynthe.sh
```

### Backend won't start
```bash
# Check Python environment
source .venv/bin/activate
python --version  # Should be 3.9+

# Check dependencies
pip install -r requirements.txt

# Try manual start to see errors
python ui/backend/api.py
```

### Frontend won't start
```bash
# Reinstall dependencies
cd ui
rm -rf node_modules package-lock.json
npm install
cd ..

# Try manual start
cd ui
npm run start
```

---

## What's Running

When Zynthe is running, you have:

- **Backend API:** http://localhost:8765
  - Endpoints: `/api/models`, `/api/auto-student/generate`, `/api/training/start`
  - WebSocket: `ws://localhost:8765/ws`

- **Frontend UI:** http://localhost:5173
  - Electron window (main)
  - Dev tools (Cmd+Option+I on Mac)

- **Processes:**
  - Python backend (uvicorn)
  - Vite dev server
  - Electron app

---

## Keyboard Shortcuts (in app)

- **Cmd+Q** - Quit app
- **Cmd+Option+I** - Open dev tools
- **Cmd+K** - Search (coming soon)
- **Ctrl+C** (in terminal) - Stop all services

---

## Architecture

```
start-zynthe.sh
├─> Python Backend (FastAPI)
│   ├─> Uvicorn ASGI server
│   ├─> REST API endpoints
│   └─> WebSocket server
│
└─> Electron Frontend
    ├─> Vite dev server (React + TypeScript)
    ├─> Main process (Node.js)
    └─> Renderer process (Chromium + React)
```

---

## Files

- `start-zynthe.sh` - Production startup with checks
- `dev.sh` - Fast development startup
- `ui/backend/api.py` - FastAPI backend
- `ui/electron/main.js` - Electron main process
- `ui/src/App.tsx` - React frontend entry
