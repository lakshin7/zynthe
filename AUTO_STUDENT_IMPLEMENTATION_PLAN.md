# Auto-Student Module Implementation Plan

## 📋 Current Structure vs Target Structure

### ✅ **ALREADY EXISTS** (Your Current Toolkit)
```
knowledge-distillation-toolkit/
├── app/
│   ├── __init__.py                    ✅ EXISTS
│   └── main.py                         ✅ EXISTS
├── core/
│   ├── config/                         ✅ EXISTS
│   ├── distillers/                     ✅ EXISTS
│   │   └── multi_stage_distiller.py    ✅ EXISTS
│   ├── preflight/                      ✅ EXISTS
│   ├── quant/                          ✅ EXISTS
│   └── utils/                          ✅ EXISTS
├── configs/                            ✅ EXISTS (17 configs)
├── data/                               ✅ EXISTS
├── training/                           ✅ EXISTS
│   ├── trainer.py                      ✅ EXISTS
│   ├── optimizer.py                    ✅ EXISTS
│   └── scheduler.py                    ✅ EXISTS
├── evaluation/                         ✅ EXISTS
│   ├── evaluator.py                    ✅ EXISTS
│   └── metrics_extended.py             ✅ EXISTS (DEI/CAS)
├── examples/                           ✅ EXISTS
├── docs/                               ✅ EXISTS (30+ docs)
├── tests/                              ✅ EXISTS (as test_*.py files)
└── tools/                              ✅ EXISTS
```

---

## ❌ **MISSING COMPONENTS** (To Be Added)

### 1. **core/auto_student/** (NEW MODULE - Priority: HIGH)
```
core/auto_student/
├── __init__.py                         ❌ MISSING
├── auto_student_builder.py             ❌ MISSING - Main builder class
├── heuristics.py                       ❌ MISSING - Rules for student sizing
├── cost_model.py                       ❌ MISSING - Performance/size estimation
├── search_engine.py                    ❌ MISSING - Architecture search
├── validator.py                        ❌ MISSING - Student validation
├── __main__.py                         ❌ MISSING - CLI entry point
└── templates/                          ❌ MISSING
    ├── base_transformer.json           ❌ MISSING
    └── student_template.yaml           ❌ MISSING
```

**Purpose:** Automatically generate optimal student architectures from teacher models

---

### 2. **cli/** (NEW MODULE - Priority: HIGH)
```
cli/
├── __init__.py                         ❌ MISSING
├── zynthe_cli.py                       ❌ MISSING - Main CLI entrypoint
└── commands/                           ❌ MISSING
    ├── __init__.py                     ❌ MISSING
    ├── auto_student.py                 ❌ MISSING - Auto-student command
    ├── distill.py                      ❌ MISSING - Distillation command
    ├── quantize.py                     ❌ MISSING - Quantization command
    └── evaluate.py                     ❌ MISSING - Evaluation command
```

**Purpose:** Professional CLI interface (like `torchrun`, `transformers-cli`)

**Note:** Currently using `app/main.py` directly - CLI would be a wrapper/extension

---

### 3. **ui/** (NEW MODULE - Priority: LOW - Optional)
```
ui/
├── __init__.py                         ❌ MISSING
├── dashboard.py                        ❌ MISSING - Web dashboard
└── assets/                             ❌ MISSING
    ├── styles.css                      ❌ MISSING
    └── app.js                          ❌ MISSING
```

**Purpose:** Optional web UI for visualizing automation

**Status:** LOW PRIORITY - Command line is sufficient initially

---

### 4. **Core Extensions** (Priority: MEDIUM)

#### a. core/preflight/teacher_analyzer.py
```python
❌ MISSING - Extract teacher architecture details
```

#### b. core/distillers/orchestrator.py
```python
❌ MISSING - Bridge between auto_student and distillers
```

#### c. core/utils/orchestrator_bridge.py
```python
❌ MISSING - Cross-module glue
```

#### d. core/quant/integrate_auto_quant.py
```python
❌ MISSING - Auto quantization hooks (optional)
```

---

### 5. **Configuration Files** (Priority: HIGH)
```
configs/
├── auto_student.yaml                   ❌ MISSING - Auto-student config
├── teacher_specs.yaml                  ❌ MISSING - Teacher specifications
└── constraints.yaml                    ❌ MISSING - Resource constraints
```

---

### 6. **Data Directories** (Priority: MEDIUM)
```
data/
├── synthetic/                          ❌ MISSING
│   ├── train.jsonl                     ❌ MISSING
│   └── val.jsonl                       ❌ MISSING
└── generated_students/                 ❌ MISSING
```

**Purpose:** Store generated student configurations

---

### 7. **Scripts** (Priority: LOW)
```
scripts/
├── auto_student_pipeline.sh            ❌ MISSING
└── distill_and_quantize.sh             ❌ MISSING
```

**Status:** Can use existing `run_training.sh` initially

---

### 8. **Documentation** (Priority: HIGH)
```
docs/
├── AUTO_STUDENT_GUIDE.md               ❌ MISSING
└── DISTILLATION_AUTOMATION.md          ❌ MISSING
```

---

## 📊 Priority Summary

### 🔴 **HIGH PRIORITY** (Essential for Auto-Student)
1. **core/auto_student/** - Complete module (8 files)
2. **configs/** - 3 new config files
3. **Documentation** - 2 guide files

**Lines of Code:** ~2,000-3,000 LOC
**Time Estimate:** 6-8 hours

---

### 🟡 **MEDIUM PRIORITY** (Enhanced functionality)
4. **core/preflight/teacher_analyzer.py** (~200 LOC)
5. **core/distillers/orchestrator.py** (~300 LOC)
6. **core/utils/orchestrator_bridge.py** (~150 LOC)
7. **data/generated_students/** directory structure

**Lines of Code:** ~650 LOC
**Time Estimate:** 2-3 hours

---

### 🟢 **LOW PRIORITY** (Nice to have)
8. **cli/** - Professional CLI wrapper (can use app/main.py initially)
9. **ui/** - Web dashboard (optional, not needed initially)
10. **scripts/** - Automation scripts (can write manually)

**Lines of Code:** ~1,000-1,500 LOC
**Time Estimate:** 4-6 hours

---

## 🎯 Recommended Implementation Order

### **Phase 1: Core Auto-Student (Day 1)**
1. Create `core/auto_student/__init__.py`
2. Create `core/auto_student/auto_student_builder.py` (main class)
3. Create `core/auto_student/heuristics.py` (sizing rules)
4. Create `core/auto_student/validator.py` (validation)
5. Create `core/auto_student/templates/` (base templates)

**Deliverable:** Can programmatically generate student architectures

---

### **Phase 2: Search & Optimization (Day 1-2)**
6. Create `core/auto_student/search_engine.py` (architecture search)
7. Create `core/auto_student/cost_model.py` (performance estimation)
8. Create `core/auto_student/__main__.py` (CLI entry)

**Deliverable:** Can search for optimal student architectures

---

### **Phase 3: Integration (Day 2)**
9. Create `configs/auto_student.yaml`
10. Create `configs/teacher_specs.yaml`
11. Create `configs/constraints.yaml`
12. Create `core/preflight/teacher_analyzer.py`
13. Create `data/generated_students/` directory

**Deliverable:** Integrated with existing toolkit

---

### **Phase 4: Orchestration (Day 3)**
14. Create `core/distillers/orchestrator.py`
15. Create `core/utils/orchestrator_bridge.py`
16. Extend `app/main.py` with auto-student command

**Deliverable:** Full end-to-end automation

---

### **Phase 5: Documentation (Day 3)**
17. Create `docs/AUTO_STUDENT_GUIDE.md`
18. Create `docs/DISTILLATION_AUTOMATION.md`
19. Update main `README.md`

**Deliverable:** Complete documentation

---

## 🔧 What You Already Have (Leverage These!)

### ✅ **Strong Foundation:**
1. **Config System** - `core/config/config_manager.py` ✅
2. **Model Loading** - `core/models/model_loader.py` ✅
3. **Distillation** - 5 distillation strategies ✅
4. **Training** - `training/trainer.py` with optimizer/scheduler ✅
5. **Evaluation** - Extended metrics (DEI/CAS) ✅
6. **Preflight** - Model inspection framework ✅

### ✅ **Can Reuse:**
- `core/preflight/model_inspector.py` → basis for `teacher_analyzer.py`
- `core/distillers/multi_stage_distiller.py` → integrate with orchestrator
- `training/trainer.py` → no changes needed
- `evaluation/metrics_extended.py` → use for validation

---

## 💡 Minimal Viable Product (MVP)

If you want to start with **minimal auto-student functionality**:

### **MVP Components (4-5 hours):**
1. `core/auto_student/auto_student_builder.py` (core logic)
2. `core/auto_student/heuristics.py` (simple rules)
3. `core/auto_student/validator.py` (basic validation)
4. `configs/auto_student.yaml` (config)
5. Extend `app/main.py` to use auto-student

**This gives you:** Automatic student generation based on teacher architecture

**Skip for MVP:**
- Advanced search algorithms
- Cost modeling
- CLI wrapper
- Web UI
- Fancy orchestration

---

## 📝 Example Usage (After Implementation)

### **Current Workflow (Manual):**
```yaml
# configs/default.yaml
model:
  name: "bert-base-uncased"              # Manual specification
  student_name: "distilbert-base-uncased" # Manual specification
```

```bash
python app/main.py --config configs/default.yaml
```

---

### **New Workflow (Automated):**
```yaml
# configs/auto_student.yaml
teacher:
  name: "bert-base-uncased"
  
auto_student:
  enabled: true
  target_compression: 0.5    # 50% of teacher size
  target_speedup: 2.0        # 2x faster
  constraints:
    max_memory_mb: 500
    min_accuracy_retention: 0.95
  search:
    strategy: "heuristic"     # or "grid", "random", "bayesian"
    num_candidates: 3
```

```bash
# Option 1: Via main.py
python app/main.py --config configs/auto_student.yaml

# Option 2: Via CLI (future)
zynthe-cli auto-student --teacher bert-base-uncased --compression 0.5

# Option 3: Standalone module
python -m core.auto_student --teacher bert-base-uncased --output student_config.yaml
```

**Output:**
```
[AUTO-STUDENT] Analyzing teacher: bert-base-uncased (110M params)
[AUTO-STUDENT] Target: 55M params, 2x speedup
[AUTO-STUDENT] Generating candidates...
  Candidate 1: 6 layers, 512 hidden, 8 heads → 52M params
  Candidate 2: 4 layers, 768 hidden, 12 heads → 48M params
  Candidate 3: 6 layers, 384 hidden, 6 heads → 58M params
[AUTO-STUDENT] Best candidate: Candidate 2 (48M params)
[AUTO-STUDENT] Saved to: data/generated_students/student_20251103_012345.yaml
```

---

## 🚀 Next Steps

Would you like me to:

1. **Start with MVP** (4-5 hours)
   - Create core auto_student module
   - Basic heuristics
   - Simple integration

2. **Full Implementation** (12-15 hours)
   - Complete all HIGH priority items
   - Add search algorithms
   - Professional CLI

3. **Minimal Extension** (2-3 hours)
   - Just add `core/auto_student/auto_student_builder.py`
   - Use with existing main.py
   - Simple rule-based student generation

**My Recommendation:** Start with **Option 1 (MVP)** to get auto-student working quickly, then iterate based on your needs!

Let me know which approach you prefer, and I'll start implementing! 🎯
