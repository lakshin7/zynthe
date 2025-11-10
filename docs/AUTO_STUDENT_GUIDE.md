# Auto-Student Guide

Complete guide to automatic student architecture generation in the Knowledge Distillation Toolkit.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Usage Examples](#usage-examples)
5. [Configuration Reference](#configuration-reference)
6. [Sizing Strategies](#sizing-strategies)
7. [Integration with Main Pipeline](#integration-with-main-pipeline)
8. [Troubleshooting](#troubleshooting)

## Overview

The **AutoStudentBuilder** module automatically generates student model architectures from teacher models. Instead of manually designing student architectures, you specify:

- **Teacher model** (e.g., BERT, RoBERTa)
- **Compression ratio** (e.g., 50% = half the size)
- **Strategy** (conservative, balanced, aggressive)

The system automatically calculates optimal student dimensions, validates the architecture, and generates ready-to-use configuration files.

### Key Features

✅ **Automatic Architecture Generation**
- Calculates student layers, hidden size, attention heads
- Ensures all constraints are satisfied (divisibility, parameter counts)
- Estimates memory usage and training time

✅ **Multiple Sizing Strategies**
- **Conservative**: Reduces depth primarily, keeps width intact
- **Balanced**: Reduces depth + width proportionally (recommended)
- **Aggressive**: Heavy reduction in all dimensions

✅ **Built-in Validation**
- Checks divisibility constraints (hidden_size % num_heads == 0)
- Verifies parameter counts are reasonable
- Ensures memory feasibility for target hardware
- Auto-fixes common issues

✅ **Multi-Candidate Generation**
- Generate multiple candidates with different compression ratios
- Compare strategies side-by-side
- Find optimal size/performance tradeoff

## Quick Start

### 1. Basic Usage

```python
from core.auto_student import AutoStudentBuilder

# Initialize with teacher model
builder = AutoStudentBuilder(teacher_name="bert-base-uncased")

# Generate student (50% compression, balanced strategy)
student = builder.generate(
    compression_ratio=0.5,
    strategy='balanced',
    validate=True,
    save=True
)

print(f"Generated: {student['num_layers']} layers, "
      f"{student['hidden_size']} hidden, "
      f"{student['total_params']:,} params")
```

**Output:**
```
Generated Student Architecture:
  Layers: 8 (66.7% of teacher)
  Hidden Size: 576 (75.0% of teacher)
  Attention Heads: 9
  Intermediate Size: 2304
  Total Params: 49,449,600 (45.0% of teacher)
```

### 2. Using Generated Config with Main Pipeline

```bash
# Generate student config
python -c "
from core.auto_student import AutoStudentBuilder
builder = AutoStudentBuilder('bert-base-uncased')
builder.generate(compression_ratio=0.5, strategy='balanced', save=True)
"

# Use generated config
python app/main.py --config data/generated_students/student_bert-base-uncased_balanced_50pct_*.yaml
```

### 3. Multi-Candidate Search

```python
# Generate multiple candidates
candidates = builder.generate_multiple(
    compression_ratios=[0.3, 0.5, 0.7],
    strategies=['conservative', 'balanced', 'aggressive']
)

# Find best candidate (e.g., closest to 50M params)
best = min(candidates, key=lambda c: abs(c['total_params'] - 50_000_000))
print(f"Best: {best['strategy']} @ {best['compression_ratio']:.1%}")
```

## Core Concepts

### Compression Ratio

The **compression ratio** determines the target size of the student relative to the teacher:

- `0.3` = 30% of teacher size (~1/3 size)
- `0.5` = 50% of teacher size (half size)
- `0.7` = 70% of teacher size (moderate compression)

**Example:**
```python
# BERT-base teacher: 110M params
builder = AutoStudentBuilder("bert-base-uncased")

# 50% compression → ~55M params
student_50 = builder.generate(compression_ratio=0.5)

# 30% compression → ~33M params
student_30 = builder.generate(compression_ratio=0.3)
```

### Sizing Strategies

Three strategies control how dimensions are reduced:

#### 1. Conservative Strategy
- **Reduces depth primarily** (fewer layers)
- **Keeps width mostly intact** (hidden size unchanged)
- **Best for**: Maintaining accuracy, simpler distillation
- **Formula**: `depth_ratio = sqrt(compression_ratio)`

**Example:**
```python
# BERT-base: 12 layers, 768 hidden
student = builder.generate(compression_ratio=0.5, strategy='conservative')
# Result: 8 layers, 704 hidden (depth reduced, width mostly intact)
```

#### 2. Balanced Strategy (Recommended)
- **Reduces depth + width proportionally**
- **Best balance** between size and performance
- **Formula**: `depth = sqrt(ratio)`, `width = ratio^0.4`

**Example:**
```python
student = builder.generate(compression_ratio=0.5, strategy='balanced')
# Result: 8 layers, 576 hidden (both reduced proportionally)
```

#### 3. Aggressive Strategy
- **Heavy reduction in all dimensions**
- **Maximum compression** for very small models
- **Formula**: `depth = ratio`, `width = ratio^0.6`

**Example:**
```python
student = builder.generate(compression_ratio=0.5, strategy='aggressive')
# Result: 6 layers, 512 hidden (heavy reduction)
```

### Validation & Auto-Fixing

The system performs **7 validation checks**:

1. ✅ Required fields present (num_layers, hidden_size, etc.)
2. ✅ Layer count in range (2-48)
3. ✅ Hidden size in range (128-4096)
4. ✅ Attention heads in range (2-32)
5. ✅ **Divisibility constraint**: `hidden_size % num_attention_heads == 0`
6. ✅ Parameter count reasonable (1M-1B)
7. ✅ Head dimension reasonable (32-256)

**Auto-fixing** handles common issues:
- Divisibility: Rounds hidden_size to nearest multiple
- Too few layers: Sets to minimum (2)
- Too few heads: Sets to minimum (2)

```python
# Example: Auto-fix in action
student = builder.generate(compression_ratio=0.5, strategy='aggressive')
# WARNING: Hidden size (512) not divisible by num_heads (6)
# → Auto-fixed: 512 → 510
```

## Usage Examples

### Example 1: Standard Workflow

```python
from core.auto_student import AutoStudentBuilder

# 1. Initialize with teacher
builder = AutoStudentBuilder(teacher_name="roberta-base")

# 2. Generate student
student = builder.generate(
    compression_ratio=0.5,
    strategy='balanced',
    validate=True,
    save=True
)

# 3. Estimate resources
estimates = builder.estimate_training_time(student, dataset_size=10000)
print(f"Training time: ~{estimates['estimated_time_minutes']:.1f} min")
print(f"Memory usage: ~{estimates['estimated_memory_gb']:.2f} GB")

# 4. Use generated config
# python app/main.py --config data/generated_students/student_roberta-base_*.yaml
```

### Example 2: Custom Teacher Model

```python
# Define custom teacher architecture
custom_teacher = {
    'num_layers': 10,
    'hidden_size': 640,
    'num_attention_heads': 10,
    'intermediate_size': 2560,
    'vocab_size': 30522,
    'total_params': 80_000_000
}

# Build student from custom teacher
builder = AutoStudentBuilder(
    teacher_name="custom-model",
    teacher_config=custom_teacher
)

student = builder.generate(compression_ratio=0.5)
```

### Example 3: Batch Generation & Selection

```python
# Generate multiple candidates
candidates = builder.generate_multiple(
    compression_ratios=[0.3, 0.4, 0.5, 0.6, 0.7],
    strategies=['conservative', 'balanced', 'aggressive'],
    save=True
)

# Filter by memory constraint (e.g., must fit in 8GB)
from core.auto_student import StudentValidator

feasible = []
for c in candidates:
    is_feasible, memory_gb = StudentValidator.check_memory_feasibility(
        c, batch_size=8, available_memory_gb=8.0
    )
    if is_feasible:
        feasible.append((c, memory_gb))
        print(f"{c['strategy']} @ {c['compression_ratio']:.1%}: "
              f"{c['total_params']:,} params, {memory_gb:.2f} GB")

# Select smallest feasible model
best = min(feasible, key=lambda x: x[0]['total_params'])
print(f"\nBest model: {best[0]['strategy']} @ {best[0]['compression_ratio']:.1%}")
```

### Example 4: Integration with Training Pipeline

```python
# Generate student
builder = AutoStudentBuilder("bert-base-uncased")
student = builder.generate(compression_ratio=0.5, save=True)

# Load and use in training
import yaml
from pathlib import Path

config_path = Path("data/generated_students").glob("student_bert-base-uncased_*.yaml")
config_path = sorted(config_path)[-1]  # Get latest

with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Now use cfg with your training pipeline
# trainer = Trainer(cfg)
# trainer.train()
```

## Configuration Reference

### Auto-Student Config (`configs/auto_student.yaml`)

```yaml
# Teacher specification
teacher:
  name: "bert-base-uncased"
  
  # Or use custom config:
  # custom_config:
  #   num_layers: 12
  #   hidden_size: 768
  #   ...

# Auto-student settings
auto_student:
  enabled: true
  compression_ratio: 0.5
  strategy: "balanced"  # conservative, balanced, aggressive
  validate: true
  auto_fix: true
  available_memory: 8.0  # GB
  
  # Multi-candidate generation
  multi_candidate:
    enabled: false
    compression_ratios: [0.3, 0.4, 0.5, 0.6]
    strategies: ["conservative", "balanced", "aggressive"]
  
  save_configs: true
  output_dir: "data/generated_students"

# Data, training, distillation configs...
```

### Generated Student Config Format

```yaml
model:
  name: "bert-base-uncased"
  student_name: "auto_student_balanced"
  
  student_architecture:
    num_layers: 8
    hidden_size: 576
    num_attention_heads: 9
    intermediate_size: 2304
    vocab_size: 30522

train:
  epochs: 3
  batch_size: 8
  lr: 2e-5
  optimizer: "adamw"
  scheduler: "cosine"

distillation:
  method: "kd_hinton"
  temperature: 2.0
  alpha: 0.5

metadata:
  teacher: "bert-base-uncased"
  compression_ratio: 0.5
  strategy: "balanced"
  teacher_params: 110000000
  student_params: 49449600
  compression_achieved: 0.449
```

## Sizing Strategies

### Comparison Table

| Strategy | Compression | Layers | Hidden | Params | Use Case |
|----------|-------------|--------|--------|--------|----------|
| **Conservative** | 50% | 8 (↓33%) | 704 (↓8%) | 83M | High accuracy, simple distill |
| **Balanced** | 50% | 8 (↓33%) | 576 (↓25%) | 49M | **Best default**, balanced |
| **Aggressive** | 50% | 6 (↓50%) | 512 (↓33%) | 45M | Very small models, edge |

### When to Use Each Strategy

**Conservative:**
- ✅ First distillation attempt (easier)
- ✅ Maintaining high accuracy is critical
- ✅ Moderate compression (50-70%)
- ❌ Need very small models (<30M params)

**Balanced (Recommended):**
- ✅ **Default choice** for most use cases
- ✅ Good size/performance tradeoff
- ✅ Any compression ratio (30-70%)
- ✅ Works well with all teachers

**Aggressive:**
- ✅ Maximum compression needed (<30M params)
- ✅ Edge deployment (mobile, IoT)
- ✅ Compute budget very limited
- ❌ Need highest accuracy

### Strategy Selection Guide

```python
# High accuracy needed → Conservative
if accuracy_priority == "high":
    strategy = "conservative"
    compression_ratio = 0.6  # Moderate compression

# Balanced needs → Balanced
elif size_priority == "medium" and accuracy_priority == "medium":
    strategy = "balanced"
    compression_ratio = 0.5

# Very small model needed → Aggressive
elif size_priority == "very_high":
    strategy = "aggressive"
    compression_ratio = 0.3  # Heavy compression
```

## Integration with Main Pipeline

### Method 1: Use Auto-Student Config Directly

```bash
# Generate config
python -c "
from core.auto_student import AutoStudentBuilder
builder = AutoStudentBuilder('bert-base-uncased')
builder.generate(compression_ratio=0.5, save=True)
"

# Run training with generated config
python app/main.py --config data/generated_students/student_bert-base-uncased_balanced_50pct_*.yaml
```

### Method 2: Enable in Config

Edit `configs/auto_student.yaml`:

```yaml
auto_student:
  enabled: true
  compression_ratio: 0.5
  strategy: balanced
```

Then run:

```bash
python app/main.py --config configs/auto_student.yaml
```

### Method 3: Programmatic Integration

```python
# In your training script
from core.auto_student import AutoStudentBuilder

if cfg.get('auto_student', {}).get('enabled', False):
    # Generate student architecture
    builder = AutoStudentBuilder(teacher_name=cfg['teacher']['name'])
    student_config = builder.generate(
        compression_ratio=cfg['auto_student']['compression_ratio'],
        strategy=cfg['auto_student']['strategy']
    )
    
    # Use student_config to build model
    # student_model = build_model_from_config(student_config)
```

## Troubleshooting

### Issue 1: Divisibility Error

**Error:**
```
Hidden size (512) not divisible by num_heads (6)
```

**Solution:**
Enable `auto_fix=True` (default):
```python
student = builder.generate(compression_ratio=0.5, auto_fix=True)
```

Or manually adjust:
```python
student['hidden_size'] = 510  # 510 % 6 == 0
```

### Issue 2: Memory Infeasible

**Error:**
```
Memory infeasible: 12.5 GB > 8.0 GB available
```

**Solution:**
Reduce compression ratio or use aggressive strategy:
```python
# Lower compression (smaller model)
student = builder.generate(compression_ratio=0.3)

# Or aggressive strategy
student = builder.generate(compression_ratio=0.5, strategy='aggressive')
```

### Issue 3: Unknown Teacher Model

**Error:**
```
Unknown teacher model: my-custom-model
```

**Solution:**
Provide custom teacher config:
```python
custom_config = {
    'num_layers': 12,
    'hidden_size': 768,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'vocab_size': 30522,
    'total_params': 110_000_000
}

builder = AutoStudentBuilder(
    teacher_name="my-custom-model",
    teacher_config=custom_config
)
```

### Issue 4: Generated Model Too Small

**Problem:** Student model is smaller than expected

**Check:**
```python
student = builder.generate(compression_ratio=0.5)
print(f"Actual compression: {student['total_params'] / teacher_params:.1%}")
```

**Solution:**
Adjust compression ratio or strategy:
```python
# Increase compression ratio
student = builder.generate(compression_ratio=0.6)

# Or use conservative strategy (keeps more params)
student = builder.generate(compression_ratio=0.5, strategy='conservative')
```

## Advanced Usage

### Custom Validation Constraints

```python
from core.auto_student import StudentValidator

# Modify constraints (class attributes)
StudentValidator.MIN_LAYERS = 4  # Require at least 4 layers
StudentValidator.MAX_PARAMS = 50_000_000  # Cap at 50M params

# Validate with custom constraints
is_valid, issues = StudentValidator.validate(student_config, strict=True)
```

### Memory Profiling

```python
from core.auto_student import StudentValidator

# Check memory for different batch sizes
for batch_size in [4, 8, 16, 32]:
    is_feasible, memory_gb = StudentValidator.check_memory_feasibility(
        student_config,
        batch_size=batch_size,
        seq_length=128,
        available_memory_gb=8.0
    )
    print(f"Batch {batch_size}: {memory_gb:.2f} GB - {'✓' if is_feasible else '✗'}")
```

### Parallel Candidate Generation

```python
from concurrent.futures import ProcessPoolExecutor

def generate_candidate(ratio, strategy):
    builder = AutoStudentBuilder("bert-base-uncased")
    return builder.generate(compression_ratio=ratio, strategy=strategy)

# Generate candidates in parallel
ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
strategies = ['conservative', 'balanced', 'aggressive']

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    for ratio in ratios:
        for strategy in strategies:
            futures.append(executor.submit(generate_candidate, ratio, strategy))
    
    candidates = [f.result() for f in futures]

print(f"Generated {len(candidates)} candidates in parallel")
```

## Best Practices

1. **Start with Balanced Strategy**
   - Use `strategy='balanced'` as default
   - Try conservative if accuracy is critical
   - Try aggressive if size is critical

2. **Validate Memory First**
   - Always check memory feasibility before training
   - Use `estimate_training_time()` to plan resources

3. **Generate Multiple Candidates**
   - Try different compression ratios (0.3, 0.5, 0.7)
   - Compare strategies side-by-side
   - Select based on size/performance tradeoff

4. **Use Auto-Fix**
   - Keep `auto_fix=True` to handle divisibility issues
   - Review fixes in logs

5. **Save Configs**
   - Set `save=True` to keep generated configs
   - Use version control for reproducibility

## Known Teacher Models

The system has built-in support for:

- `bert-base-uncased` (110M params, 12 layers, 768 hidden)
- `bert-large-uncased` (340M params, 24 layers, 1024 hidden)
- `roberta-base` (125M params, 12 layers, 768 hidden)
- `roberta-large` (355M params, 24 layers, 1024 hidden)
- `albert-base-v2` (12M params, 12 layers, 768 hidden)
- `distilbert-base-uncased` (66M params, 6 layers, 768 hidden)

For other models, provide a custom config.

## References

- **Main documentation**: `docs/overview.md`
- **Implementation plan**: `AUTO_STUDENT_IMPLEMENTATION_PLAN.md`
- **Heuristics code**: `core/auto_student/heuristics.py`
- **Validation code**: `core/auto_student/validator.py`
- **Builder code**: `core/auto_student/auto_student_builder.py`

## Support

For issues or questions:
1. Check this guide
2. Review test examples: `test_auto_student.py`
3. Examine generated logs
4. Check validation errors and auto-fixes
