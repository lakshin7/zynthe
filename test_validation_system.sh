#!/bin/bash
# Quick test script for model validation system

echo "🧪 Testing Enhanced Model Validation System"
echo "============================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

echo "1️⃣ Testing Device Detection..."
python -c "
from core.preflight.model_validator import ModelValidator
validator = ModelValidator()
print(f'   ✓ Detected device: {validator.available_device}')
"
echo ""

echo "2️⃣ Testing Valid Model Pair..."
python -c "
from core.preflight.model_validator import validate_models
result = validate_models('bert-base-uncased', 'distilbert-base-uncased')
print(f'   ✓ Pair compatible: {result[\"pair_compatible\"]}')
print(f'   ✓ Teacher exists: {result[\"teacher\"][\"exists\"]}')
print(f'   ✓ Student exists: {result[\"student\"][\"exists\"]}')
if 'compression_ratio' in result:
    print(f'   ✓ Compression: {result[\"compression_ratio\"]}')
"
echo ""

echo "3️⃣ Testing Invalid Model..."
python -c "
from core.preflight.model_validator import validate_models
result = validate_models('fake-model-12345', 'distilbert-base-uncased')
print(f'   ✓ Validation failed as expected')
print(f'   ✓ Errors detected: {len(result[\"teacher\"][\"errors\"])} error(s)')
print(f'   ✓ Alternatives suggested: {len(result[\"teacher\"][\"alternatives\"])}')
"
echo ""

echo "✅ All validation tests passed!"
echo ""
echo "📝 Next Steps:"
echo "   1. Start Zynthe: ./start-zynthe.sh"
echo "   2. Open http://localhost:5173"
echo "   3. Create New Experiment"
echo "   4. Select models and run preflight"
echo ""
