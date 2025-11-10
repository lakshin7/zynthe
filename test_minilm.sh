#!/bin/bash
# Test script for different models with full workflow including visualization

cd "$(dirname "$0")"

echo "======================================================================"
echo "Testing Complete Workflow with RoBERTa Models"
echo "======================================================================"
echo ""
echo "Models:"
echo "  Teacher: roberta-base (125M params)"
echo "  Student: distilroberta-base (82M params)"
echo "  Compression: 1.5x"
echo ""
echo "Dataset: IMDB (1000 samples for quick test)"
echo "Epochs: 2"
echo ""
echo "Expected time: ~5-10 minutes on M2 Air"
echo ""
echo "======================================================================"
echo ""

python3 app/main.py --config configs/quick_test_minilm.yaml
