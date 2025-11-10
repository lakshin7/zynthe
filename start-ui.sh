#!/bin/bash

echo "🚀 Zynthe Desktop App - Quick Start"
echo "===================================="
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✓ Node.js found: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed"
    exit 1
fi

echo "✓ npm found: $(npm --version)"
echo ""

# Navigate to UI directory
cd "$(dirname "$0")/ui" || exit 1

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
    echo ""
fi

# Start the app
echo "🎨 Starting Zynthe Desktop App..."
echo ""
echo "This will:"
echo "  1. Start Python backend (port 8765)"
echo "  2. Start React dev server (port 5173)"
echo "  3. Launch Electron window"
echo ""
echo "Press Ctrl+C to stop"
echo ""

npm run start
