#!/bin/bash
# Test the training pipeline integration

echo "🧪 Testing Training Pipeline Integration"
echo "========================================"
echo ""

# Check if backend is running
echo "1. Checking if backend is running..."
curl -s http://localhost:8765/ | python3 -m json.tool
if [ $? -eq 0 ]; then
    echo "✅ Backend is running"
else
    echo "❌ Backend is not running. Start it with: cd ui/backend && python api.py"
    exit 1
fi
echo ""

# Test creating a training run
echo "2. Creating a test training run..."
cat > /tmp/test_training_config.json << EOF
{
  "teacher_model": "bert-base-uncased",
  "student_model": "distilbert-base-uncased",
  "student_architecture": "distilbert",
  "student_hidden_size": 384,
  "student_num_layers": 6,
  "dataset": "imdb_sample",
  "epochs": 2,
  "batch_size": 8,
  "learning_rate": 0.001,
  "temperature": 3.0,
  "alpha": 0.5,
  "optimization_preset": "fast",
  "log_level": "info",
  "save_checkpoints": true,
  "checkpoint_frequency": 1
}
EOF

RESPONSE=$(curl -s -X POST http://localhost:8765/api/training/create \
  -H "Content-Type: application/json" \
  -d @/tmp/test_training_config.json)

echo "$RESPONSE" | python3 -m json.tool
EXP_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment_id'])" 2>/dev/null)

if [ -n "$EXP_ID" ]; then
    echo "✅ Training created with ID: $EXP_ID"
else
    echo "❌ Failed to create training"
    exit 1
fi
echo ""

# Wait a bit for training to start
echo "3. Waiting 5 seconds for training to start..."
sleep 5
echo ""

# Check experiment status
echo "4. Checking experiment status..."
curl -s "http://localhost:8765/api/experiments/$EXP_ID" | python3 -m json.tool | head -30
echo ""

# Test pause
echo "5. Testing pause..."
curl -s -X POST "http://localhost:8765/api/training/$EXP_ID/pause" | python3 -m json.tool
sleep 2
echo ""

# Test resume
echo "6. Testing resume..."
curl -s -X POST "http://localhost:8765/api/training/$EXP_ID/resume" | python3 -m json.tool
sleep 2
echo ""

# Test checkpoint
echo "7. Testing checkpoint save..."
curl -s -X POST "http://localhost:8765/api/training/$EXP_ID/checkpoint" | python3 -m json.tool
sleep 2
echo ""

# Test stop
echo "8. Testing stop..."
curl -s -X POST "http://localhost:8765/api/training/$EXP_ID/stop" | python3 -m json.tool
echo ""

echo "========================================"
echo "✅ All tests completed!"
echo ""
echo "Next steps:"
echo "  1. Open the Electron app"
echo "  2. Click 'New Training'"
echo "  3. Configure and start a training run"
echo "  4. Click 'View Live' to see real-time metrics"
echo "  5. Try the Pause/Resume/Stop buttons"
