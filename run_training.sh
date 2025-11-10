#!/bin/bash

# Complete Training Pipeline with Teacher Fine-tuning
# Trains both teacher and student models with proper monitoring

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Knowledge Distillation Toolkit - Training Pipeline      ║"
echo "║   With Teacher Fine-tuning Fix                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python environment
echo "🔍 Checking Python environment..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $PYTHON_VERSION"

# Check if torch is available
if python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} found')" 2>/dev/null; then
    DEVICE=$(python3 -c "import torch; print('MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU')")
    echo "Device: $DEVICE"
else
    echo "❌ PyTorch not found in current Python environment"
    echo ""
    echo "Install with: pip3 install torch torchvision"
    exit 1
fi

# Check transformers
if ! python3 -c "import transformers; print(f'✅ Transformers {transformers.__version__} found')" 2>/dev/null; then
    echo "❌ Transformers not found"
    echo "Install with: pip3 install transformers"
    exit 1
fi

# Check other dependencies
echo ""
echo "Checking other dependencies..."
for module in "sklearn" "matplotlib" "seaborn" "tqdm" "yaml" "numpy"; do
    if python3 -c "import $module" 2>/dev/null; then
        echo "✅ $module found"
    else
        echo "⚠️  $module not found (may cause issues)"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🚀 Starting Training Pipeline"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configuration: configs/retrain_teacher.yaml"
echo "Phase 1: Teacher fine-tuning (2 epochs)"
echo "Phase 2: Knowledge distillation (3 epochs)"
echo ""

# Record start time
START_TIME=$(date +%s)

# Add current directory to PYTHONPATH to resolve 'core' imports
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run training
if python3 train_with_fix.py; then
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "✅ Training Completed Successfully!"
    echo "════════════════════════════════════════════════════════════"
    echo "Duration: ${MINUTES}m ${SECONDS}s"
    echo ""
    
    # Find the latest experiment
    LATEST_EXP=$(ls -td experiments/*/ 2>/dev/null | head -1)
    
    if [ -n "$LATEST_EXP" ]; then
        echo "📁 Experiment saved to: $LATEST_EXP"
        echo ""
        echo "Next steps:"
        echo "  1. Test teacher: python3 tools/diagnose_teacher.py --model ${LATEST_EXP}teacher_model --samples 200"
        echo "  2. Compare models: python3 examples/compare_teacher_student.py --exp ${LATEST_EXP%/} --tokenizer-mode separate"
        echo ""
    fi
    
    exit 0
else
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "❌ Training Failed"
    echo "════════════════════════════════════════════════════════════"
    echo "Check the error messages above for details"
    echo ""
    exit 1
fi
