#!/bin/bash

# Zynthe Backend Startup Script
# This script starts the FastAPI backend with full transparency

echo "🚀 Zynthe Knowledge Distillation Backend"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  FastAPI not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies ready"
echo ""

# Start the backend server
echo "🌐 Starting API server on http://localhost:8765"
echo "📡 WebSocket endpoint: ws://localhost:8765/ws"
echo "📖 API documentation: http://localhost:8765/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"
echo ""

python3 api.py
