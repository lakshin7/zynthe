# main.py Compatibility Analysis

**Date**: October 23, 2025  
**Status**: ✅ Mostly Compatible, Minor Issues Found

---

## Issues Found

### 1. ⚠️ Incorrect load_models() Signature
**Location**: Line 235
```python
teacher, student, tokenizer = load_models(cfg_manager.resolved_config, cfg_manager.device())
```

**Problem**: `load_models()` expects `(cfg, device)` but config is passed as first arg
- First parameter should be ConfigManager instance, not resolved_config
- Second parameter is device string/None

**Expected signature** (from model_loader.py):
```python
def load_models(cfg: "ConfigManager", device=None):
```

**Fix**:
```python
# Option 1: Pass ConfigManager instance
teacher, student, tokenizer = load_models(cfg_manager, cfg_manager.device())

# Option 2: Modify load_models to accept dict
teacher, student, tokenizer = load_models(cfg_manager.resolved_config, cfg_manager.device())
```

### 2. ⚠️ get_runtime() Method Doesn't Exist
**Location**: Line 254
```python
runtime_device = cfg_manager.get_runtime().get("device", "cpu")
```

**Problem**: ConfigManager doesn't have `get_runtime()` method

**Fix**:
```python
runtime_device = cfg_manager.device()
```

### 3. ✅ Trainer Integration - GOOD
The trainer integration looks correct:
```python
trainer = Trainer(
    teacher=teacher,
    student=student,
    tokenizer=tokenizer,
    config=cfg_manager.resolved_config,  # ✅ Correct
    device=cfg_manager.device(),         # ✅ Correct
    experiment_dir=cfg_manager.experiment_dir  # ✅ Correct
)
```

### 4. ⚠️ distill() Command - MultiStageDistiller Instantiation Issue
**Location**: Lines 113-115
```python
Distiller = getattr(mod, "MultiStageDistiller", None) or getattr(mod, "Distiller", None)
if Distiller is None:
    raise ImportError("No MultiStageDistiller/Distiller class found in module.")
distiller = Distiller(cfg)  # ❌ Wrong parameters
```

**Problem**: MultiStageDistiller needs more parameters:
```python
MultiStageDistiller(
    teacher=teacher,
    student=student,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)
```

**Fix**: The `distill()` command needs to load models and dataloaders first

---

## Fixes Required

### Fix 1: Correct load_models() Call
```python
# main.py line 235
# OLD:
teacher, student, tokenizer = load_models(cfg_manager.resolved_config, cfg_manager.device())

# NEW:
teacher, student, tokenizer = load_models(cfg_manager, cfg_manager.device())
```

### Fix 2: Remove get_runtime() Call
```python
# main.py line 254
# OLD:
runtime_device = cfg_manager.get_runtime().get("device", "cpu")

# NEW:
runtime_device = cfg_manager.device()
```

### Fix 3: Fix distill() Command
```python
# main.py lines 96-133
@app.command()
def distill(
    config: str = typer.Option("configs/default.yaml", help="Path to distill config"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show config and exit"),
):
    """Run a distillation pipeline."""
    cm, cfg = load_config(config)
    rprint(f"[bold green]Loaded distill config:[/bold green] {config}")

    if dry_run:
        rprint("[yellow]Dry-run: printing parsed config and exiting[/yellow]")
        rprint(cfg)
        raise typer.Exit()

    try:
        # Load models
        teacher, student, tokenizer = load_models(cm, cm.device())
        
        # Load dataloaders
        from data.dataloaders import create_dataloaders
        train_loader, val_loader = create_dataloaders(cfg, tokenizer)
        
        # Create multi-stage distiller
        from core.distillers.multi_stage_distiller import MultiStageDistiller
        distiller = MultiStageDistiller(
            teacher=teacher,
            student=student,
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=cm.device()
        )
        
        # Run distillation
        report = distiller.run()
        rprint(f"[bold green]Distillation completed![/bold green]")
        rprint(f"Final accuracy: {report['summary'].get('total_accuracy_gain', 0):.2f}%")
        
    except Exception as exc:
        LOG.exception("Failed to run distillation: %s", exc)
        rprint("[red]Distillation failed — check logs for details.[/red]")
        raise typer.Exit(code=1)
```

---

## Testing Strategy

### Recommended Models for M2 Air

**Teacher-Student Pairs** (all compatible with MPS):

#### Option 1: DistilBERT Family (Recommended for M2) ✅
- **Teacher**: `distilbert-base-uncased` (66M params)
- **Student**: `distilbert-base-uncased` with 3 layers (custom, ~22M params)
- **Why**: Lightweight, fast on M2, proven architecture

#### Option 2: TinyBERT (Very Light) ✅
- **Teacher**: `huawei-noah/TinyBERT_General_6L_768D` (67M params)
- **Student**: `huawei-noah/TinyBERT_General_4L_312D` (14M params)
- **Why**: Designed for distillation, very efficient

#### Option 3: MobileBERT (Mobile-Optimized) ✅
- **Teacher**: `google/bert-base-uncased` (110M params)
- **Student**: `google/mobilebert-uncased` (25M params)
- **Why**: Optimized for mobile/edge devices like M2

### Recommended Kaggle Datasets

#### Option 1: Twitter Sentiment140 (Best for Quick Testing) ✅
- **Link**: https://www.kaggle.com/datasets/kazanova/sentiment140
- **Size**: 1.6M tweets
- **Task**: Binary sentiment (positive/negative)
- **Format**: CSV
- **Why**: Clean, large, easy to process

#### Option 2: Amazon Product Reviews (Realistic) ✅
- **Link**: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
- **Size**: 4M reviews
- **Task**: Binary sentiment
- **Format**: Text files
- **Why**: Real-world data, good quality

#### Option 3: IMDB Reviews (Already Used) ✅
- **Link**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Size**: 50K reviews
- **Task**: Binary sentiment
- **Format**: CSV
- **Why**: Smaller, faster to test, already in your code

---

## Complete Test Setup

I'll create a complete test configuration and dataset downloader for you with:
1. **Models**: TinyBERT (teacher) → TinyBERT-4L (student)
2. **Dataset**: Twitter Sentiment140 (subset for fast testing)
3. **Config**: Optimized for M2 Air with MPS

This will be the most efficient setup for your M2 Air!
