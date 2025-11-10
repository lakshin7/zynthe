#!/bin/bash

# Stop all Zynthe processes

echo "🛑 Stopping Zynthe Desktop App..."

# Kill Python backend
pkill -f "python ui/backend/api.py" 2>/dev/null && echo "✅ Backend stopped" || echo "ℹ️  No backend process found"

# Kill Vite dev server
pkill -f "vite" 2>/dev/null && echo "✅ Vite stopped" || echo "ℹ️  No Vite process found"

# Kill Electron
pkill -f "electron" 2>/dev/null && echo "✅ Electron stopped" || echo "ℹ️  No Electron process found"

# Kill any uvicorn processes
pkill -f "uvicorn" 2>/dev/null && echo "✅ Uvicorn stopped" || echo "ℹ️  No Uvicorn process found"

echo ""
echo "✨ All Zynthe processes stopped"
