# Zynthe Packaging & Deployment Guide

## Problem: Python Environment Dependencies

### The Issue
When Zynthe is packaged as a standalone app, the training subprocess (`main.py`) needs access to PyTorch, Transformers, and other ML dependencies. The error you saw:

```
ModuleNotFoundError: No module named 'torch'
```

This happens because the subprocess uses a Python interpreter that doesn't have the required packages installed.

## Solutions Implemented

### 1. Smart Python Detection (training_manager.py)

The `TrainingManager` now intelligently finds the correct Python executable:

```python
# Priority order:
1. ZYNTHE_PYTHON environment variable (best for packaged apps)
2. .venv/bin/python (virtual environment)
3. Conda environment Python
4. System Python (fallback with warning)
```

**Key Features:**
- ✅ Environment variable support for packaged apps
- ✅ Automatic virtual environment detection
- ✅ Conda environment support
- ✅ Dependency verification before training starts
- ✅ Clear error messages if dependencies are missing

### 2. Enhanced Startup Script (start-zynthe.sh)

The startup script now:
- ✅ Automatically detects `.venv` or conda environments
- ✅ Creates virtual environment if none exists
- ✅ Installs dependencies automatically
- ✅ Verifies PyTorch and Transformers are available
- ✅ Sets `ZYNTHE_PYTHON` environment variable
- ✅ Uses correct Python for both API and training

### 3. Dependency Verification

Before starting any training, the system:
```python
# Verify Python has required packages
verify_cmd = [python_exe, "-c", "import torch; import transformers; print('OK')"]
result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)
```

If verification fails:
- ❌ Training doesn't start
- 📢 Error broadcast via WebSocket
- 💡 Clear error message shown to user

## For Development (Current Setup)

### Quick Start
```bash
cd /Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit
./start-zynthe.sh
```

The script will:
1. Detect/create virtual environment
2. Install dependencies if needed
3. Verify packages
4. Start backend with correct Python
5. Start frontend

### Manual Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export ZYNTHE_PYTHON="$(pwd)/.venv/bin/python"

# Start backend
python ui/backend/api.py

# In another terminal, start frontend
cd ui
npm run electron:dev
```

## For Production (Packaged App)

### Option 1: Bundle Python Environment (Recommended)

When packaging with Electron, include a Python environment:

**1. Create a standalone Python environment:**
```bash
# Use pyinstaller or similar
pip install pyinstaller

# Create standalone distribution
pyinstaller app/main.py \
    --onefile \
    --name zynthe-training \
    --hidden-import torch \
    --hidden-import transformers \
    --hidden-import yaml
```

**2. In Electron builder config (package.json):**
```json
{
  "build": {
    "extraResources": [
      {
        "from": "dist/zynthe-training",
        "to": "python/zynthe-training"
      },
      {
        "from": ".venv",
        "to": "python/venv"
      }
    ]
  }
}
```

**3. Update training_manager.py for packaged app:**
```python
# In start() method
if getattr(sys, 'frozen', False):
    # Running in packaged app
    app_path = Path(sys._MEIPASS)
    python_exe = app_path / "python" / "venv" / "bin" / "python"
else:
    # Running in development
    # ... existing logic
```

### Option 2: Use System Python with Installer

**During app installation:**
```bash
# Install script (install.sh)
#!/bin/bash

APP_DIR="/Applications/Zynthe.app"
PYTHON_ENV="$APP_DIR/Contents/Resources/python"

# Create Python environment
python3 -m venv "$PYTHON_ENV"
source "$PYTHON_ENV/bin/activate"

# Install dependencies
pip install torch transformers pyyaml

# Set environment in app's plist
defaults write com.zynthe.app ZynthePython "$PYTHON_ENV/bin/python"
```

**In Electron main process:**
```javascript
// main.js
const pythonPath = process.env.ZYNTHE_PYTHON || 
                   path.join(app.getPath('userData'), 'python', 'bin', 'python');

process.env.ZYNTHE_PYTHON = pythonPath;
```

### Option 3: Docker Container (Best for Server Deployment)

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set Python path
ENV ZYNTHE_PYTHON=/usr/local/bin/python

# Expose ports
EXPOSE 8765 5173

# Start services
CMD ["bash", "start-zynthe.sh"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  zynthe:
    build: .
    ports:
      - "8765:8765"
      - "5173:5173"
    volumes:
      - ./data:/app/data
      - ./experiments:/app/experiments
    environment:
      - ZYNTHE_PYTHON=/usr/local/bin/python
```

## Testing the Fix

### Test 1: Development Environment
```bash
# Clean start
rm -rf .venv
./start-zynthe.sh

# Should see:
# ✓ Using virtual environment: .venv
# ✓ All required packages available
# ✓ Backend started
```

### Test 2: Training with Correct Python
```bash
# Start app
./start-zynthe.sh

# In browser, start a training job
# Check logs - should NOT see "ModuleNotFoundError"

# Verify correct Python is used
ps aux | grep python
# Should show .venv/bin/python running main.py
```

### Test 3: Missing Dependencies
```bash
# Test error handling
export ZYNTHE_PYTHON="/usr/bin/python3"  # System Python without packages
./start-zynthe.sh

# Try to start training
# Should see clear error message about missing dependencies
```

### Test 4: Environment Variable Override
```bash
# Create custom environment
python3 -m venv /tmp/zynthe-test
source /tmp/zynthe-test/bin/activate
pip install torch transformers

# Set override
export ZYNTHE_PYTHON="/tmp/zynthe-test/bin/python"
./start-zynthe.sh

# Training should use /tmp/zynthe-test Python
```

## Best Practices

### ✅ DO

1. **Always use virtual environments in development**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Set ZYNTHE_PYTHON for production**
   ```bash
   export ZYNTHE_PYTHON="/path/to/python/with/dependencies"
   ```

3. **Include dependency verification in CI/CD**
   ```bash
   python -c "import torch; import transformers" || exit 1
   ```

4. **Bundle Python environment in packaged apps**
   - Include .venv in app bundle
   - Or use PyInstaller to create standalone binary

5. **Document Python requirements clearly**
   - Python >= 3.8
   - PyTorch >= 2.0
   - Transformers >= 4.30

### ❌ DON'T

1. **Don't rely on system Python**
   - System Python may not have ML packages
   - Different systems have different Python versions

2. **Don't use `sys.executable` blindly**
   - API might run with different Python than training needs
   - Always verify dependencies

3. **Don't skip dependency verification**
   - Check packages before starting training
   - Fail fast with clear error messages

4. **Don't hardcode Python paths**
   - Use environment variables
   - Support multiple installation methods

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Cause:** Training subprocess using wrong Python

**Fix:**
```bash
# Check which Python is being used
ps aux | grep "python.*main.py"

# Verify Python has packages
/path/to/that/python -c "import torch"

# Set correct Python
export ZYNTHE_PYTHON="/path/to/correct/python"
```

### Issue: "Backend started but training fails"

**Cause:** API and training using different Python environments

**Fix:**
```bash
# Use start script (it sets ZYNTHE_PYTHON automatically)
./start-zynthe.sh

# OR manually ensure same Python
PYTHON=$(which python)
export ZYNTHE_PYTHON="$PYTHON"
python ui/backend/api.py
```

### Issue: "Dependencies not found in packaged app"

**Cause:** Python environment not included in app bundle

**Fix:**
1. Include .venv in extraResources (Electron Builder)
2. Or use PyInstaller to bundle Python + packages
3. Or run post-install script to create environment

### Issue: "Training starts but imports fail"

**Cause:** Some packages missing from verification

**Fix:**
```python
# In training_manager.py, add to verify_cmd:
verify_cmd = [
    python_exe, "-c",
    "import torch; import transformers; import yaml; import numpy; print('OK')"
]
```

## Platform-Specific Notes

### macOS
- Use `.venv/bin/python` (not Scripts)
- May need to sign Python binary for Gatekeeper
- Include Python framework in app bundle

### Windows
- Use `.venv\Scripts\python.exe`
- Watch out for path separators (use Path objects)
- May need to install Visual C++ redistributables

### Linux
- Usually has Python pre-installed
- Still use virtual environment
- May need to install python3-venv package

## Summary

✅ **Fixed:** Python environment detection in training_manager.py
✅ **Fixed:** Start script now uses virtual environment
✅ **Added:** Dependency verification before training
✅ **Added:** ZYNTHE_PYTHON environment variable support
✅ **Added:** Automatic venv creation and setup
✅ **Added:** Clear error messages for missing dependencies

🎯 **Result:** Training will always use Python with correct dependencies, whether in development or packaged app.

## Next Steps for Production

1. **Choose packaging strategy:** Bundle Python or post-install?
2. **Update Electron builder config:** Include Python environment
3. **Add installer script:** Create venv during installation
4. **Test on clean system:** Verify everything works without manual setup
5. **Document for users:** How to verify installation

---

**Status:** Development environment fixed ✅  
**Next:** Production packaging configuration ⏳
