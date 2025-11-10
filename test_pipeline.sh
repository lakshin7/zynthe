#!/bin/bash

# End-to-End Test Script
# Tests the complete pipeline: training, diagnosis, and comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Add current directory to PYTHONPATH to resolve 'core' imports
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Knowledge Distillation Toolkit - E2E Test               ║"
echo "║   Full Pipeline Validation                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
TEST_CONFIG="configs/retrain_teacher.yaml"
DIAGNOSTIC_SAMPLES=100
COMPARISON_BATCH=8

# Track status
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to print test results
test_result() {
    if [ $1 -eq 0 ]; then
        echo "✅ PASSED: $2"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "❌ FAILED: $2"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
}

# Test 1: Environment Check
echo "════════════════════════════════════════════════════════════"
echo "TEST 1: Environment Check"
echo "════════════════════════════════════════════════════════════"
echo ""

python3 -c "
import sys
import torch
import transformers
import sklearn
import matplotlib
import yaml

print(f'Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Device: MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU')
" 2>/dev/null

test_result $? "Environment dependencies"

# Test 2: Config File Validation
echo "════════════════════════════════════════════════════════════"
echo "TEST 2: Configuration Validation"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ -f "$TEST_CONFIG" ]; then
    echo "Config file exists: $TEST_CONFIG"
    
    # Check key settings
    python3 -c "
import yaml
with open('$TEST_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(f\"Teacher epochs: {cfg['train'].get('teacher_epochs', 'NOT SET')}\")
print(f\"Finetune teacher: {cfg['train'].get('finetune_teacher', 'NOT SET')}\")
print(f\"Distillation epochs: {cfg['train']['epochs']}\")
print(f\"Batch size: {cfg['train']['batch_size']}\")
"
    test_result $? "Configuration validation"
else
    echo "Config file not found: $TEST_CONFIG"
    test_result 1 "Configuration validation"
fi

# Test 3: Data Files Check
echo "════════════════════════════════════════════════════════════"
echo "TEST 3: Dataset Files"
echo "════════════════════════════════════════════════════════════"
echo ""

TRAIN_DATA="data/imdb_train.jsonl"
VAL_DATA="data/imdb_val.jsonl"

if [ -f "$TRAIN_DATA" ] && [ -f "$VAL_DATA" ]; then
    TRAIN_LINES=$(wc -l < "$TRAIN_DATA")
    VAL_LINES=$(wc -l < "$VAL_DATA")
    echo "Training samples: $TRAIN_LINES"
    echo "Validation samples: $VAL_LINES"
    test_result 0 "Dataset files"
else
    echo "Missing data files!"
    test_result 1 "Dataset files"
fi

# Test 4: Model Loader Fix
echo "════════════════════════════════════════════════════════════"
echo "TEST 4: Model Loader Fix (Label Mappings)"
echo "════════════════════════════════════════════════════════════"
echo ""

python3 -c "
import sys
sys.path.insert(0, '.')
from core.models.model_loader import load_models
from core.config.config_manager import ConfigManager

cfg = ConfigManager('$TEST_CONFIG')
teacher, student, tokenizer = load_models(cfg, device='cpu')

# Check teacher labels
assert hasattr(teacher.config, 'label2id'), 'Teacher missing label2id'
assert hasattr(teacher.config, 'id2label'), 'Teacher missing id2label'
assert teacher.config.label2id == {'negative': 0, 'positive': 1}, f'Wrong label2id: {teacher.config.label2id}'
assert teacher.config.id2label == {0: 'negative', 1: 'positive'}, f'Wrong id2label: {teacher.config.id2label}'

# Check student labels
assert hasattr(student.config, 'label2id'), 'Student missing label2id'
assert hasattr(student.config, 'id2label'), 'Student missing id2label'
assert student.config.label2id == {'negative': 0, 'positive': 1}, f'Wrong label2id: {student.config.label2id}'
assert student.config.id2label == {0: 'negative', 1: 'positive'}, f'Wrong id2label: {student.config.id2label}'

print('✅ Teacher label2id:', teacher.config.label2id)
print('✅ Teacher id2label:', teacher.config.id2label)
print('✅ Student label2id:', student.config.label2id)
print('✅ Student id2label:', student.config.id2label)
" 2>&1

test_result $? "Model loader label mappings"

# Test 5: Trainer Fix
echo "════════════════════════════════════════════════════════════"
echo "TEST 5: Trainer Fix (Teacher Fine-tuning)"
echo "════════════════════════════════════════════════════════════"
echo ""

python3 -c "
import sys
sys.path.insert(0, '.')
from training.trainer import Trainer
import inspect

# Check if finetune_teacher method exists
assert hasattr(Trainer, 'finetune_teacher'), 'Trainer missing finetune_teacher method'
print('✅ Trainer has finetune_teacher method')

# Check __init__ for teacher_optimizer
init_source = inspect.getsource(Trainer.__init__)
assert 'teacher_optimizer' in init_source, 'Trainer missing teacher_optimizer'
print('✅ Trainer has teacher_optimizer')

# Check fit method calls finetune_teacher
fit_source = inspect.getsource(Trainer.fit)
assert 'finetune_teacher' in fit_source, 'Trainer.fit does not call finetune_teacher'
print('✅ Trainer.fit calls finetune_teacher')

print('✅ All trainer fixes verified')
" 2>&1

test_result $? "Trainer teacher fine-tuning fix"

# Test 6: Quick Training Test (Optional - can be slow)
echo "════════════════════════════════════════════════════════════"
echo "TEST 6: Quick Training Test (OPTIONAL)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "This test runs actual training and may take 10-30 minutes."
echo "Skip? (y/n) [default: n]"
read -t 10 -r SKIP_TRAINING || SKIP_TRAINING="n"

if [[ "$SKIP_TRAINING" =~ ^[Yy]$ ]]; then
    echo "⏭️  Skipping training test"
    echo ""
else
    echo "Running full training pipeline..."
    echo ""
    
    if bash run_training.sh; then
        # Get latest experiment
        LATEST_EXP=$(ls -td experiments/*/ 2>/dev/null | head -1)
        
        if [ -n "$LATEST_EXP" ]; then
            echo ""
            echo "Training completed! Experiment: $LATEST_EXP"
            
            # Test 7: Teacher Diagnosis
            echo ""
            echo "════════════════════════════════════════════════════════════"
            echo "TEST 7: Teacher Model Diagnosis"
            echo "════════════════════════════════════════════════════════════"
            echo ""
            
            if python3 tools/diagnose_teacher.py --model "${LATEST_EXP}teacher_model" --samples $DIAGNOSTIC_SAMPLES 2>&1 | tee /tmp/diagnosis.txt; then
                # Check accuracy
                ACCURACY=$(grep "Accuracy:" /tmp/diagnosis.txt | grep -oE '[0-9]+\.[0-9]+' | head -1)
                echo ""
                echo "Teacher Accuracy: $ACCURACY"
                
                # Should be > 0.70 (70%)
                if (( $(echo "$ACCURACY > 0.70" | bc -l) )); then
                    echo "✅ Teacher accuracy is good (>70%)"
                    test_result 0 "Teacher diagnosis"
                else
                    echo "⚠️  Teacher accuracy below 70% - may need more training"
                    test_result 1 "Teacher diagnosis"
                fi
            else
                test_result 1 "Teacher diagnosis"
            fi
            
            # Test 8: Model Comparison
            echo ""
            echo "════════════════════════════════════════════════════════════"
            echo "TEST 8: Teacher vs Student Comparison"
            echo "════════════════════════════════════════════════════════════"
            echo ""
            
            if python3 examples/compare_teacher_student.py \
                --exp "${LATEST_EXP%/}" \
                --tokenizer-mode separate \
                --batch-size $COMPARISON_BATCH 2>&1 | tee /tmp/comparison.txt; then
                
                # Extract results
                TEACHER_ACC=$(grep "Teacher Accuracy:" /tmp/comparison.txt | grep -oE '[0-9]+\.[0-9]+' | head -1)
                STUDENT_ACC=$(grep "Student Accuracy:" /tmp/comparison.txt | grep -oE '[0-9]+\.[0-9]+' | head -1)
                
                echo ""
                echo "Comparison Results:"
                echo "  Teacher: $TEACHER_ACC"
                echo "  Student: $STUDENT_ACC"
                
                # Validate results
                if (( $(echo "$TEACHER_ACC > 0.70" | bc -l) )) && \
                   (( $(echo "$STUDENT_ACC > 0.70" | bc -l) )); then
                    echo "✅ Both models performing well"
                    test_result 0 "Model comparison"
                else
                    echo "⚠️  One or both models below 70% accuracy"
                    test_result 1 "Model comparison"
                fi
                
                # Check comparison artifacts
                COMP_DIR="${LATEST_EXP}comparison"
                EXPECTED_FILES=(
                    "metrics_comparison.png"
                    "confusion_matrices_comparison.png"
                    "efficiency_comparison.png"
                    "comparison_results.json"
                    "COMPARISON_REPORT.md"
                )
                
                ALL_FILES_EXIST=true
                for file in "${EXPECTED_FILES[@]}"; do
                    if [ ! -f "$COMP_DIR/$file" ]; then
                        echo "❌ Missing: $file"
                        ALL_FILES_EXIST=false
                    fi
                done
                
                if $ALL_FILES_EXIST; then
                    echo "✅ All comparison artifacts generated"
                    test_result 0 "Comparison artifacts"
                else
                    test_result 1 "Comparison artifacts"
                fi
            else
                test_result 1 "Model comparison"
            fi
        else
            echo "No experiment directory found"
            test_result 1 "Training test"
        fi
    else
        test_result 1 "Training test"
    fi
fi

# Final Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Test Summary                                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed. Check the output above."
    exit 1
fi
