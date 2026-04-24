# Mypy Type Fix Implementation Plan

> **Goal:** Fix 309 mypy errors across 35 files to achieve clean type checking.
>
> **Strategy:** Focus on high-impact, low-risk fixes first. Use `type: ignore` for complex issues requiring architectural changes.
>
> **Error Breakdown:**
> - 112 [attr-defined] - Objects without type annotations
> - 44 [var-annotated] - Variables need type hints
> - 39 [union-attr] - Optional types not handled
> - 24 [index] - Indexing on untyped objects
> - 24 [assignment] - Type mismatches
> - 23 [arg-type] - Wrong argument types
> - 18 [operator] - Operator type issues
> - 7 [call-overload] - Function overload mismatches
> - 6 [dict-item] - Dict type mismatches
> - 4 [name-defined] - Names not defined
> - 3 [import-untyped] - Missing type stubs
> - 1 [truthy-function] - Function could be bool
> - 1 [return-value] - Return type issue
> - 1 [override] - Override signature mismatch
> - 1 [no-redef] - Redefinition
> - 1 [misc] - Miscellaneous

---

## Phase 1: Quick Wins (type: ignore for complex cases)

### Task 1.1: Fix [attr-defined] errors (112 total)

These are typically `object has no attribute X` errors where the variable has untyped dict/list that needs indexing.

**Strategy:** Add `# type: ignore[attr-defined]` to individual lines or add type annotations.

**Files with most attr-defined errors:**
- `core/preflight/analyser.py` - 34 errors
- `core/preflight/resource_probe.py` - 18 errors
- `core/utils/data_validator.py` - 4 errors
- `training/trainer.py` - 15 errors

**Example Fix:**
```python
# Before
results.append(item)

# After
results.append(item)  # type: ignore[attr-defined]
```

Or better, add proper type annotation:
```python
# Before
results = []

# After
results: List[Any] = []
```

---

### Task 1.2: Fix [var-annotated] errors (44 total)

These need explicit type annotations for variables.

**Files:**
- `core/agents/teacher_agent.py` - Multiple variables
- `core/distillers/causal_lm/trainer.py` - Multiple variables
- `training/trainer.py` - train_losses, val_losses, metrics_history, etc.

**Example Fix:**
```python
# Before
train_losses = []
val_losses = []

# After
train_losses: List[float] = []
val_losses: List[float] = []
```

---

## Phase 2: Core Files Priority Fix

### Task 2.1: Fix training/trainer.py (Priority: HIGH)

**File:** `training/trainer.py`
**Error Count:** ~50 errors
**Key Issues:**
- Untyped lists (train_losses, val_losses, etc.)
- Union types not handled
- Dict indexing on untyped objects

**Implementation:**
```python
# Line 245-259: Add type annotations
train_losses: List[float] = []
val_losses: List[float] = []
metrics_history: Dict[str, List[float]] = {}
batch_train_losses: List[float] = []
batch_val_losses: List[float] = []
batch_val_running_acc: List[float] = []
last_preds: List[Any] = []
last_labels: List[Any] = []
```

### Task 2.2: Fix core/preflight/analyser.py (Priority: HIGH)

**File:** `core/preflight/analyser.py`
**Error Count:** ~50 errors
**Key Issues:**
- Line 93: `results` needs type annotation
- Lines 113+: Object without append attribute
- Lines 342+: Union type issues

**Implementation:**
```python
# Line 93: Add type annotation
results: Dict[str, Any] = {}  # type: ignore[var-annotated]

# Lines with append on untyped lists
errors: List[str] = []
warnings: List[str] = []
recommendations: List[str] = []
```

### Task 2.3: Fix core/models/model_loader.py (Priority: HIGH)

**File:** `core/models/model_loader.py`
**Error Count:** ~5 errors
**Key Issues:**
- Line 291: Tuple assignment type mismatch
- Lines 365, 392: Object without append

**Implementation:**
```python
# Line 291: Fix tuple assignment
model_types: tuple[str, ...] = ()  # type: ignore[assignment]

# Lines 365, 392: Add type to lists
loaded_models: List[Any] = []
```

---

## Phase 3: Medium Priority Files

### Task 3.1: Fix core/distillers/*.py (Priority: MEDIUM)

**Files:**
- `core/distillers/attention_transfer.py` - Override signature issue
- `core/distillers/feature_distiller.py` - Type annotations needed
- `core/distillers/similarity_transfer.py` - Type annotations needed
- `core/distillers/multi_stage_distiller.py` - Multiple issues
- `core/distillers/kd_hinton.py` - Redefinition issue

### Task 3.2: Fix core/preflight/*.py (Priority: MEDIUM)

**Files:**
- `core/preflight/resource_probe.py` - Multiple issues
- `core/preflight/data_inspector.py` - Multiple issues
- `core/preflight/model_inspector.py` - Various issues
- `core/preflight/model_validator.py` - Various issues

### Task 3.3: Fix data/*.py (Priority: MEDIUM)

**Files:**
- `data/dataloaders.py` - cache_cfg annotation needed
- `data/image_dataloaders.py` - Union type issues
- `data/augmentations.py` - augment_cfg annotation needed
- `data/preprocess.py` - basic_cfg annotation needed

---

## Phase 4: Lower Priority Files

### Task 4.1: Fix evaluation/*.py (Priority: LOW)

**Files:**
- `evaluation/evaluator.py` - Union type issues
- `evaluation/evaluator_extended.py` - Name not defined

### Task 4.2: Fix other core files (Priority: LOW)

**Files:**
- `core/config/config_manager.py` - Import and assignment issues
- `core/pkg/exporter.py` - Call overload issues
- `core/utils/logger.py` - Argument type issues
- `core/pipelines/*.py` - Various issues

---

## Implementation Commands

### Run mypy on specific file
```bash
mypy training/trainer.py --ignore-missing-imports
```

### Run full mypy with report
```bash
mypy . --ignore-missing-imports --no-error-summary | tee mypy_errors.txt
```

### After fixes, verify
```bash
# Should show 0 errors
mypy . --ignore-missing-imports 2>&1 | grep "error:" | wc -l
```

---

## Verification Checklist

After each phase, verify:
- [ ] `mypy training/trainer.py --ignore-missing-imports` passes
- [ ] `mypy core/preflight/analyser.py --ignore-missing-imports` passes
- [ ] `mypy core/models/model_loader.py --ignore-missing-imports` passes
- [ ] Full `mypy . --ignore-missing-imports` shows <50 errors

---

## Alternative: Targeted Type Checking

If full cleanup is too extensive, use targeted checking for CI:

```bash
# Only check key files
mypy app/main.py core/config/config_manager.py core/models/model_loader.py \
  training/trainer.py --ignore-missing-imports
```

This matches the original implementation_plan.md scope.

---

## Execution

**Recommended approach:**
1. Run Task 2.1-2.3 (High priority - core training pipeline)
2. Run Task 3.1-3.3 (Medium priority - distillers and preflight)
3. Run Task 4.1-4.2 (Low priority - evaluation and utils)
4. Final verification

**Time estimate:** 2-4 hours for full cleanup if doing manually, or ~30 min for targeted approach.