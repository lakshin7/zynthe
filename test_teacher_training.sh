#!/bin/bash
# Test script for teacher training and confusion matrix fixes

echo "======================================================================"
echo "Testing Teacher Training & Confusion Matrix Enhancements"
echo "======================================================================"
echo ""
echo "This test will:"
echo "  1. Train teacher model (2 epochs)"
echo "  2. Distill to student (2 epochs)"
echo "  3. Generate enhanced confusion matrix"
echo ""
echo "Dataset: IMDB (1000 samples)"
echo "Models: roberta-base → distilroberta-base"
echo "Expected time: ~8-12 minutes on M2 Air"
echo ""
echo "======================================================================"
echo ""

# Activate virtual environment if needed
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run training with teacher training enabled
python3 app/main.py --config configs/with_teacher_training.yaml

echo ""
echo "======================================================================"
echo "Test Complete!"
echo "======================================================================"
echo ""
echo "Check the following artifacts:"
echo "  • Confusion matrix: experiments/LATEST/confusion_matrix.png"
echo "  • Training curves: experiments/LATEST/training_curves.png"
echo "  • Model comparison: experiments/LATEST/visualizations/model_comparison.png"
echo ""
echo "The confusion matrix should now show:"
echo "  ✓ Clear 'True Label' and 'Predicted Label' axes"
echo "  ✓ Subtitle explaining row/column meaning"
echo "  ✓ Overall accuracy displayed"
echo "  ✓ Higher resolution (150 DPI)"
echo ""
