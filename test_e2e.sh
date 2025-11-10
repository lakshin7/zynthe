#!/bin/bash

# Zynthe E2E Testing Script
# This script performs end-to-end testing of the Zynthe UI

set -e

echo "🚀 Zynthe E2E Testing Script"
echo "=============================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if backend is running
echo -e "${BLUE}📡 Checking backend status...${NC}"
if curl -s http://localhost:8765/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is running on port 8765${NC}"
else
    echo -e "${RED}❌ Backend is not running${NC}"
    echo -e "${YELLOW}💡 Start it with: cd ui/backend && python api.py${NC}"
    echo ""
fi

# Check if frontend is running
echo -e "${BLUE}🌐 Checking frontend status...${NC}"
if curl -s http://localhost:5173/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend is running on port 5173${NC}"
else
    echo -e "${RED}❌ Frontend is not running${NC}"
    echo -e "${YELLOW}💡 Start it with: cd ui && npm run dev${NC}"
    echo ""
fi

echo ""
echo -e "${BLUE}🧪 Available API Endpoints:${NC}"
echo "================================"

# Test GET /api/experiments
echo -ne "GET /api/experiments ... "
if curl -s http://localhost:8765/api/experiments > /dev/null 2>&1; then
    EXPERIMENTS=$(curl -s http://localhost:8765/api/experiments | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    echo -e "${GREEN}✅ ($EXPERIMENTS experiments)${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test GET /api/models
echo -ne "GET /api/models ... "
if curl -s http://localhost:8765/api/models > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:8765/api/models | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    echo -e "${GREEN}✅ ($MODELS models)${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test GET /api/datasets
echo -ne "GET /api/datasets ... "
if curl -s http://localhost:8765/api/datasets > /dev/null 2>&1; then
    DATASETS=$(curl -s http://localhost:8765/api/datasets | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    echo -e "${GREEN}✅ ($DATASETS datasets)${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test GET /api/models/compare
echo -ne "GET /api/models/compare ... "
if curl -s http://localhost:8765/api/models/compare > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo ""
echo -e "${BLUE}📊 E2E Test Checklist:${NC}"
echo "================================"
echo "1. ⚙️  Settings Configuration"
echo "   - Open Settings (gear icon)"
echo "   - Change theme, adjust training defaults"
echo "   - Click Save and verify persistence"
echo ""
echo "2. 📁 Dataset Upload"
echo "   - Create test file: test_data.jsonl"
echo "   - Upload via drag-and-drop or file picker"
echo "   - Verify success message"
echo ""
echo "3. 🚀 New Training Run"
echo "   - Click 'New Training' button"
echo "   - Fill all 5 steps (Project, Dataset, Model, Config, Review)"
echo "   - Click 'Start Training'"
echo ""
echo "4. 📊 Live Dashboard"
echo "   - Click 'View Live' on running experiment"
echo "   - Watch metrics update in real-time"
echo "   - Test Pause/Resume/Stop buttons"
echo ""
echo "5. 📋 View Completed Experiment"
echo "   - Click experiment card after completion"
echo "   - Check all 3 tabs: Metrics, Logs, Config"
echo ""
echo "6. 📊 Model Comparison"
echo "   - Click 'Compare Models' button"
echo "   - Select 2-3 models"
echo "   - View comparison table and charts"
echo ""
echo -e "${GREEN}✨ Ready to test! Open http://localhost:5173 in your browser${NC}"
echo ""
echo -e "${YELLOW}📖 For detailed testing steps, see: E2E_TESTING_GUIDE.md${NC}"
