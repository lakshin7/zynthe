#!/bin/bash
# Quick test script for evaluation API endpoints

echo "🚀 Starting Evaluation API Test"
echo "================================"
echo ""

# Check if backend is running
if ! curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "⚠️  Backend is not running!"
    echo ""
    echo "Please start the backend first:"
    echo "  cd ui/backend"
    echo "  python api.py"
    echo ""
    exit 1
fi

echo "✅ Backend is running"
echo ""

# Run the test
echo "🧪 Running API endpoint tests..."
echo ""

python test_evaluation_api.py

echo ""
echo "================================"
echo "✅ Test complete!"
