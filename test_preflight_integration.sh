#!/bin/bash
# Preflight Integration Test Script

echo "🧪 Testing Preflight Integration"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    local data=$4
    
    echo -n "Testing $name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$url" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$url" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed (HTTP $http_code)${NC}"
        return 1
    fi
}

# Test 1: Backend Health
echo "Test 1: Backend Health Check"
echo "----------------------------"
test_endpoint "Health endpoint" "http://localhost:8765/health"
if [ $? -eq 0 ]; then
    echo "Response:"
    curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null || echo "  (Could not parse JSON)"
fi
echo ""

# Test 2: Device Info
echo "Test 2: Device Information"
echo "-------------------------"
test_endpoint "Device info" "http://localhost:8765/api/device/info"
if [ $? -eq 0 ]; then
    echo "Response:"
    curl -s http://localhost:8765/api/device/info | python3 -m json.tool 2>/dev/null || echo "  (Could not parse JSON)"
fi
echo ""

# Test 3: HF Token Status
echo "Test 3: HuggingFace Token Status"
echo "--------------------------------"
test_endpoint "HF token status" "http://localhost:8765/api/settings/hf-token"
if [ $? -eq 0 ]; then
    echo "Response:"
    curl -s http://localhost:8765/api/settings/hf-token | python3 -m json.tool 2>/dev/null || echo "  (Could not parse JSON)"
fi
echo ""

# Test 4: Valid Model Pair
echo "Test 4: Validate BERT → DistilBERT (should pass)"
echo "------------------------------------------------"
test_endpoint "Model validation (valid)" \
    "http://localhost:8765/api/models/validate" \
    "POST" \
    '{"teacher_model":"bert-base-uncased","student_model":"distilbert-base-uncased"}'

if [ $? -eq 0 ]; then
    echo "Response summary:"
    response=$(curl -s -X POST http://localhost:8765/api/models/validate \
        -H "Content-Type: application/json" \
        -d '{"teacher_model":"bert-base-uncased","student_model":"distilbert-base-uncased"}')
    
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Valid: {data.get('valid', False)}\")
    print(f\"  Can proceed: {data.get('can_proceed', False)}\")
    print(f\"  Compression: {data.get('compression_ratio', 'N/A')}\")
    if data.get('device_info'):
        print(f\"  Device: {data['device_info'].get('current_device', 'unknown')}\")
except:
    pass
" 2>/dev/null
fi
echo ""

# Test 5: Invalid Model
echo "Test 5: Validate Invalid Model (should fail gracefully)"
echo "-------------------------------------------------------"
test_endpoint "Model validation (invalid)" \
    "http://localhost:8765/api/models/validate" \
    "POST" \
    '{"teacher_model":"non-existent-model-12345","student_model":"distilbert-base-uncased"}'

if [ $? -eq 0 ]; then
    echo "Response summary:"
    response=$(curl -s -X POST http://localhost:8765/api/models/validate \
        -H "Content-Type: application/json" \
        -d '{"teacher_model":"non-existent-model-12345","student_model":"distilbert-base-uncased"}')
    
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Valid: {data.get('valid', False)}\")
    teacher_errors = data.get('teacher', {}).get('errors', [])
    if teacher_errors:
        print(f\"  Teacher errors: {len(teacher_errors)}\")
        for err in teacher_errors[:2]:
            print(f\"    - {err}\")
    alternatives = data.get('teacher', {}).get('alternatives', [])
    if alternatives:
        print(f\"  Alternatives suggested: {len(alternatives)}\")
except:
    pass
" 2>/dev/null
fi
echo ""

# Test 6: Datasets endpoint
echo "Test 6: Built-in Datasets"
echo "------------------------"
test_endpoint "Datasets list" "http://localhost:8765/api/datasets"
if [ $? -eq 0 ]; then
    echo "Response summary:"
    curl -s http://localhost:8765/api/datasets | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Total datasets: {len(data)}\")
    built_in = [d for d in data if d.get('type') == 'built-in']
    print(f\"  Built-in datasets: {len(built_in)}\")
except:
    pass
" 2>/dev/null
fi
echo ""

# Summary
echo "================================"
echo "✅ Preflight integration tests complete!"
echo ""
echo "📝 Next Steps:"
echo "   1. All tests passing? Great! Open http://localhost:5173"
echo "   2. Go to New Experiment → Step 4 (Preflight)"
echo "   3. Click 'Test Connection' in the debug panel"
echo "   4. Select models and run preflight validation"
echo ""
echo "🐛 If tests failed:"
echo "   • Make sure backend is running: ./start-zynthe.sh"
echo "   • Check backend logs for errors"
echo "   • Verify HF_TOKEN is set in .env file"
echo ""
