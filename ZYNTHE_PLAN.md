# Zynthe Development Plan

> Last Updated: 2026-04-24
> Focus: Knowledge Distillation + CLI Audit + MUON Optimizer

---

## 1. Skills to Install (Restart Required)

After restarting OpenCode, install these plugins in your `opencode.json`:

```json
{
  "plugin": [
    "superpowers@git+https://github.com/obra/superpowers.git"
  ]
}
```

### Installation Commands

```bash
# Option 1: Claude Code (recommended for these plugins)
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
/plugin install superpowers@claude-plugins-official

# Option 2: OpenCode - add to project opencode.json or global
# Project-level:
echo '{"plugin": ["superpowers@git+https://github.com/obra/superpowers.git"]}' > opencode.json

# Claude-Mem (separate install):
npx claude-mem install --ide opencode

# Awesome Claude Code resources - use as reference, not installable
# Browse: https://github.com/hesreallyhim/awesome-claude-code
```

---

## 2. Current Status Summary

### ✅ Completed (from pending_plan.md)
- Phase 1-3: Complete
- Phase 4-6: Mostly complete
- Evaluation/Visualizer: Complete
- Pipeline infrastructure: Complete

### ⚠️ Remaining Issues
1. CLI duplication: `main.py` vs `main_new.py` 
2. Shared helpers not extracted
3. Integration notebooks missing (Colab/Kaggle)
4. Runtime flow needs verification

### 📋 Distillation Workflow Gaps
- Config validation exists but NOT called in main flow
- Preflight analyzer exists but NOT integrated
- Error handling between components needs audit

---

## 3. Concrete Implementation Plan

### Phase 1: Core Audit & Integration (Priority 0)

| # | Task | File | Action | Est. |
|---|------|------|--------|------|
| 1.1 | Runtime flow verification | `app/runtime.py` | Verify preflight → train → eval chain | Medium |
| 1.2 | Config validation | `core/preflight/analyser.py` | Ensure called in distillation flow | Low |
| 1.3 | CLI entrypoint | `app/main.py`, `app/main_new.py` | Unify/remove duplication | Low |

### Phase 2: Optimizer Integration (Priority 0)

| # | Task | File | Action | Est. |
|---|------|------|--------|------|
| 2.1 | MUON optimizer | `training/optimizer.py` | Add MUON support | Low |
| 2.2 | MUON installation | requirements.txt | Add `pip install git+https://github.com/KellerJordan/Muon` | Low |

### Phase 3: CLI Polish (Priority 1)

| # | Task | File | Action | Est. |
|---|------|------|--------|------|
| 3.1 | Shared helpers | `app/helpers.py` | Extract config/model loading | Low |
| 3.2 | Error messages | Multiple | Improve clarity | Low |
| 3.3 | Smoke test | `tests/test_smoke.py` | Verify CLI works | Low |

### Phase 4: Local Testing (Priority 1)

| # | Task | Platform | Action |
|---|---------|----------|---------|
| 4.1 | Latitude 7490 (CPU) | Run minimal 1-epoch test |
| 4.2 | Smoke tests | Validate CLI commands |

### Phase 5: GPU Validation (Priority 2)

| # | Task | Platform | Action |
|---|---------|----------|---------|
| 5.1 | Kaggle T4 | Push code, run integration |
| 5.2 | Verify outputs | Check distillation artifacts |

---

## 4. MUON Optimizer Details

### What is MUON?
- Layer-wise optimizer using Newton-Schulz orthogonalization
- Research shows significant improvements over AdamW for deep networks
- Works on hidden weight matrices (ndim >= 2)

### Installation
```bash
pip install git+https://github.com/KellerJordan/Muon
```

### Usage in Zynthe
```python
from muon import MuonWithAuxAdam

# Split parameters: Muon for hidden weights, AdamW for embeddings/biases
hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
other_params = [p for p in model.parameters() if p.ndim < 2]

param_groups = [
    dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
    dict(params=other_params, use_muon=False, lr=3e-4, weight_decay=0.01),
]
optimizer = MuonWithAuxAdam(param_groups)
```

### Integration Point
- Add to `training/optimizer.py` in `OptimizerFactory`
- Config option: `train.optimizer: muon`

---

## 5. PTQ (Quantization) - Optional/Later

Post-Training Quantization will be added as a **post-distillation step**, not a full pipeline flow.

```yaml
# Config for optional PTQ
quantization:
  enable: true
  mode: fp16  # float16 for MPS compatibility
```

---

## 6. Error Points Identified

### Potential Runtime Issues
1. **Config validation** - may not be called before model loading
2. **Preflight analyzer** - exists but not integrated in runtime
3. **CLI duplication** - main.py and main_new.py have overlapping code

### Files to Audit
- `app/runtime.py` - main execution flow
- `app/main.py` - CLI entrypoint
- `core/preflight/analyser.py` - validation logic
- `training/optimizer.py` - optimizer factory

---

## 7. Free GPU Platforms

| Platform | GPU | SSH | Notes |
|----------|-----|-----|-------|
| **Kaggle** | T4 (free) | ❌ | 30+ hrs/week, notebook only |
| **Colab** | T4/P100 | ❌ | Free tier available |
| **Hugging Face** | T4 | ❌ | Easy deployment |
| **RunPod** | Varies | ✅ | Free credit sometimes |

**Recommendation**: Use Kaggle for GPU testing (push via git, run notebooks)

---

## 8. Next Steps After Restart

1. **Restart OpenCode** to activate skills
2. **Verify plugins loaded**:
   - Ask: "What superpowers do you have?"
3. **Start Phase 1**:
   - Audit `app/runtime.py` execution flow
   - Find where config validation should be called
4. **Proceed with implementation**

---

## 9. Reference Documents

- `implementation_plan.md` - full phase plan
- `pending_plan.md` - remaining items
- `docs/DISTILLATION_WORKFLOW.md` - 9-phase workflow
- `docs/structure_overview.md` - component guide

---

*Plan generated for Zynthe knowledge distillation toolkit development*