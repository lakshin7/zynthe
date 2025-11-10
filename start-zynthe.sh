#!/bin/bash

echo "🚀 Starting Zynthe Knowledge Distillation Toolkit..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill any existing processes on these ports
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
lsof -ti:8765 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

# Detect and activate Python environment
echo ""
echo -e "${BLUE}Setting up Python environment...${NC}"
cd /Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit

# Check for virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_CMD=".venv/bin/python"
    export ZYNTHE_PYTHON="$(pwd)/.venv/bin/python"
    echo -e "${GREEN}✓ Using virtual environment: .venv${NC}"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    PYTHON_CMD="python"
    export ZYNTHE_PYTHON="$(which python)"
    echo -e "${GREEN}✓ Using conda environment: $CONDA_DEFAULT_ENV${NC}"
else
    # Try to create venv if it doesn't exist
    echo -e "${YELLOW}⚠️  No virtual environment found. Creating one...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    PYTHON_CMD=".venv/bin/python"
    export ZYNTHE_PYTHON="$(pwd)/.venv/bin/python"
    
    # Install dependencies
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Verify dependencies
echo -e "${BLUE}Verifying Python dependencies...${NC}"
if $PYTHON_CMD -c "import torch; import transformers; print('✓ All required packages available')" 2>/dev/null; then
    echo -e "${GREEN}✓ All dependencies verified${NC}"
else
    echo -e "${YELLOW}⚠️  Missing dependencies. Installing...${NC}"
    pip install -r requirements.txt
fi

# Start Python backend
echo ""
echo -e "${BLUE}Starting FastAPI backend on http://localhost:8765...${NC}"
$PYTHON_CMD ui/backend/api.py &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 3

# Check if backend started successfully
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start Electron + Vite frontend
echo ""
echo -e "${BLUE}Starting Electron app with Vite...${NC}"
cd ui
npm run electron:dev

# Cleanup when script exits
echo ""
echo -e "${YELLOW}Shutting down Zynthe...${NC}"
kill $BACKEND_PID 2>/dev/null
echo -e "${GREEN}✓ Zynthe stopped${NC}"
