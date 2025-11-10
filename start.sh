#!/bin/bash

# Zynthe Complete Startup Script
# This script starts both backend and frontend for testing

set -e

echo "🚀 Starting Zynthe Application"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo -e "${BLUE}🐍 Activating virtual environment...${NC}"
source .venv/bin/activate

# Install backend dependencies
echo -e "${BLUE}📦 Installing backend dependencies...${NC}"
pip install -q -r ui/backend/requirements.txt || {
    echo -e "${YELLOW}⚠️  Some packages might need to be installed. Trying again...${NC}"
    pip install fastapi uvicorn websockets pyyaml
}

# Check if Node modules exist
if [ ! -d "ui/node_modules" ]; then
    echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
    cd ui
    npm install
    cd ..
fi

echo ""
echo -e "${GREEN}✅ All dependencies installed${NC}"
echo ""
echo -e "${BLUE}🎬 Starting services...${NC}"
echo "================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down services...${NC}"
    jobs -p | xargs -r kill 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM

# Start backend
echo -e "${BLUE}🔧 Starting Backend (Port 8765)...${NC}"
cd ui/backend
python api.py &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
echo -e "${BLUE}⏳ Waiting for backend to start...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:8765/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is running!${NC}"
        break
    fi
    sleep 1
done

# Start frontend
echo -e "${BLUE}🌐 Starting Frontend (Port 5173)...${NC}"
cd ui
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}✨ Zynthe is starting up!${NC}"
echo "================================"
echo ""
echo -e "📡 Backend:  ${GREEN}http://localhost:8765${NC}"
echo -e "🌐 Frontend: ${GREEN}http://localhost:5173${NC}"
echo ""
echo -e "${YELLOW}📖 See E2E_TESTING_GUIDE.md for testing instructions${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for processes
wait
