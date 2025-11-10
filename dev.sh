#!/bin/bash

# Quick dev mode - assumes everything is already set up
# Just runs the apps without checks

cd "$(dirname "$0")"

echo "🚀 Starting Zynthe (dev mode)..."

# Cleanup function
cleanup() {
    echo -e "\n🛑 Stopping..."
    pkill -f "python ui/backend/api.py" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    pkill -f "electron" 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
source .venv/bin/activate
python ui/backend/api.py &

# Wait a bit
sleep 2

# Start frontend
cd ui
npm run start &

# Wait for both
wait
