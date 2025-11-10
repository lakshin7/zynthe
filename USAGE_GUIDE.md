# Complete System Usage Guide

This guide demonstrates how to use the fully integrated knowledge distillation toolkit with all components working together.

---

## Quick Start

### 1. Basic Setup

```python
import torch
from core.config.config_manager import ConfigManager
from core.distillers.multi_stage_distiller import MultiStageDistiller
from data.dataloaders import get_dataloaders

# Initialize config manager
config_mgr = ConfigManager(
    config_path="configs/multi_stage.yaml",
    experiments_root="experiments"
)

# Get dataloaders
train_loader, val_loader = get_dataloaders(
    train_path="data/imdb_train.jsonl",
    val_path="data/imdb_val.jsonl",
    batch_size=config_mgr.get('train.batch_size', 8),
    device=config_mgr.device()
)
```

### 2. Load Models

```python
from transformers import AutoModelForSequenceClassification

# Load teacher (larger model)
teacher = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(config_mgr.device())

# Load student (smaller model)
student = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    num_hidden_layers=3  # Smaller
).to(config_mgr.device())
```

### 3. Run Multi-Stage Distillation

```python
# Create multi-stage distiller
multi_stage = MultiStageDistiller(
    teacher=teacher,
    student=student,
    config=config_mgr.resolved_config,
    train_loader=train_loader,
    val_loader=val_loader,
    device=config_mgr.device()
)

# Execute training
report = multi_stage.run()

# Access results
print(f"Final Accuracy: {report['summary']['total_accuracy_gain']:.2f}%")
print(f"Stages Completed: {report['summary']['total_stages']}")
```

---

## Configuration Examples

### Basic 3-Stage Pipeline

**File**: `configs/basic_multi_stage.yaml`

```yaml
# Training settings
train:
  epochs: 5
  batch_size: 8  # Auto-adjusted for MPS
  lr: 2e-5
  weight_decay: 0.01
  warmup_steps: 100

# Model configuration
model:
  name: distilbert-base-uncased
  type: transformer
  num_labels: 2

# Data paths
data:
  train_path: data/imdb_train.jsonl
  val_path: data/imdb_val.jsonl

# Multi-stage distillation
multi_stage:
  stages:
    # Stage 1: Knowledge Distillation (Logit alignment)
    - name: kd_alignment
      type: kd
      epochs: 2
      config:
        temperature: 4.0
        alpha: 0.7  # Weight for KD loss
    
    # Stage 2: Feature Transfer (Layer-wise)
    - name: feature_transfer
      type: feature
      epochs: 2
      config:
        teacher_layers: ['layer_2', 'layer_3']
        student_layers: ['layer_1', 'layer_2']
        feature_weight: 0.5
    
    # Stage 3: Similarity Transfer (Relational)
    - name: similarity_transfer
      type: similarity
      epochs: 3
      config:
        layer: 'layer_2'
        similarity_metric: cosine
        weight: 0.4
        progressive: true

# Device settings
device:
  prefer_mps: true
  prefer_cuda: false
```

### Advanced 5-Stage Pipeline

**File**: `configs/advanced_multi_stage.yaml`

```yaml
train:
  epochs: 10
  batch_size: 16
  lr: 3e-5

model:
  name: bert-base-uncased
  type: transformer

data:
  train_path: data/imdb_train.jsonl
  val_path: data/imdb_val.jsonl

multi_stage:
  # Enable loss weight scheduling
  loss_schedule:
    initial_weights:
      alpha: 0.9
      beta: 0.6
      gamma: 0.4
    schedule_type: cosine
  
  stages:
    # Stage 1: Pure KD
    - name: initial_kd
      type: kd_hinton
      epochs: 2
      config:
        temperature: 5.0
        alpha: 0.9
        use_hints: true
    
    # Stage 2: Feature Distillation
    - name: deep_features
      type: feature
      epochs: 2
      config:
        teacher_layers: ['layer_3', 'layer_6', 'layer_9']
        student_layers: ['layer_1', 'layer_2', 'layer_3']
        feature_weight: 0.6
        use_projector: true
    
    # Stage 3: Attention Transfer
    - name: attention_transfer
      type: attention
      epochs: 2
      config:
        attention_weight: 0.5
        match_heads: true
    
    # Stage 4: Similarity Transfer
    - name: relational_knowledge
      type: similarity
      epochs: 3
      config:
        layer: 'layer_3'
        similarity_metric: cosine
        progressive: true
        graph_mode: true  # Use graph-based similarity
    
    # Stage 5: Quantization-Aware Training (optional)
    - name: quantization
      type: qat
      epochs: 1
      config:
        bits: 8
        calibration_steps: 100

device:
  prefer_mps: false
  prefer_cuda: true
```

---

## Using Individual Distillers

### KD-Hinton (Classical KD)

```python
from core.distillers.kd_hinton import KDHintonDistiller

kd_distiller = KDHintonDistiller(
    teacher=teacher,
    student=student,
    temperature=4.0,
    alpha=0.7,  # KD weight
    use_hints=True,
    device='mps'
)

# Training loop
for batch in train_loader:
    inputs, labels = batch
    
    # Forward pass
    teacher_outputs = teacher(inputs)
    student_outputs = student(inputs)
    
    # Compute loss
    loss, metrics = kd_distiller.compute_loss(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        targets=labels
    )
    
    # Backward pass (your optimizer)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {metrics['total']:.4f}")
```

### Feature Distillation

```python
from core.distillers.feature_distiller import FeatureDistiller

feature_distiller = FeatureDistiller(
    teacher=teacher,
    student=student,
    teacher_layers=['layer_2', 'layer_3'],
    student_layers=['layer_1', 'layer_2'],
    feature_weight=0.5,
    device='mps'
)

# Use same training loop as KD
```

### Similarity Transfer

```python
from core.distillers.similarity_transfer import SimilarityTransfer

sim_distiller = SimilarityTransfer(
    teacher=teacher,
    student=student,
    layer='layer_2',
    similarity_metric='cosine',  # or 'euclidean', 'angular', 'gram'
    progressive=True,
    graph_mode=False,
    cross_modality=False,
    device='mps'
)

# Training
for batch in train_loader:
    inputs, labels = batch
    
    # Get features with hooks
    teacher_features = {}
    student_features = {}
    
    def get_hook(name, features_dict):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
    
    # Register hooks
    teacher.layer_2.register_forward_hook(
        get_hook('layer_2', teacher_features)
    )
    student.layer_2.register_forward_hook(
        get_hook('layer_2', student_features)
    )
    
    # Forward
    teacher_outputs = teacher(inputs)
    student_outputs = student(inputs)
    
    # Compute similarity loss
    loss, metrics = sim_distiller.compute_loss(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        targets=labels,
        student_features=student_features,
        teacher_features=teacher_features
    )
    
    print(f"SAS: {metrics['sas']:.4f}")
```

---

## Distiller Registry

### Register Custom Distiller

```python
from core.distillers.registry import DistillerRegistry
from core.distillers.base_distiller import BaseDistiller

# Create custom distiller
class MyCustomDistiller(BaseDistiller):
    def __init__(self, teacher, student, **kwargs):
        super().__init__(teacher, student, **kwargs)
        # Your initialization
    
    def compute_loss(self, student_outputs, teacher_outputs, targets=None, **kwargs):
        # Your loss computation
        loss = your_loss_function(student_outputs, teacher_outputs)
        metrics = {'custom_metric': loss.item()}
        return loss, metrics

# Register it
registry = DistillerRegistry()
registry.register('custom', MyCustomDistiller)

# Use in multi-stage config
config = {
    'multi_stage': {
        'stages': [
            {
                'name': 'custom_stage',
                'type': 'custom',  # Uses your distiller
                'epochs': 2,
                'config': {}
            }
        ]
    }
}
```

### List Available Distillers

```python
from core.distillers.registry import DistillerRegistry

registry = DistillerRegistry()
available = registry.list_available()
print(f"Available: {available}")
# Output: ['kd', 'kd_hinton', 'feature', 'similarity', 'similarity_transfer', 'attention']

# Get specific distiller
kd_class = registry.get('kd')
print(f"KD Class: {kd_class.__name__}")
# Output: KDHintonDistiller
```

---

## Config Manager Features

### Device Auto-Detection

```python
from core.config.config_manager import ConfigManager

config_mgr = ConfigManager("configs/my_config.yaml")

# Auto-detects best device
device = config_mgr.device()
print(f"Using device: {device}")
# Output: mps (on M2), cuda (on GPU), or cpu

# Get device-specific settings
batch_size = config_mgr.get('train.batch_size')
print(f"Batch size: {batch_size}")
# Output: 8 (MPS) or 32 (CUDA)
```

### Config Validation

```python
from core.config.config_manager import ConfigManager

try:
    config_mgr = ConfigManager("configs/my_config.yaml")
    print("✅ Config valid!")
except ValueError as e:
    print(f"❌ Invalid config: {e}")

# Required sections:
# - train: epochs, batch_size, lr
# - model: name, type
# - data: train_path, val_path
```

### Experiment Management

```python
config_mgr = ConfigManager(
    config_path="configs/my_config.yaml",
    experiments_root="experiments"
)

# Auto-creates experiment directory
exp_dir = config_mgr.experiment_dir
print(f"Experiment: {exp_dir}")
# Output: experiments/20251023T064700Z_10c83faa

# Save configs
config_mgr.save_resolved_config()
# Saves to: experiments/{id}/resolved_config.yaml
```

---

## Advanced Features

### Layer Freezing

```python
multi_stage = MultiStageDistiller(...)

# Freeze specific layers during a stage
config = {
    'multi_stage': {
        'stages': [
            {
                'name': 'freeze_early',
                'type': 'feature',
                'epochs': 2,
                'config': {
                    'freeze_layers': [0, 1]  # Freeze first 2 layers
                }
            }
        ]
    }
}
```

### Knowledge Replay

```python
# Enable knowledge replay across stages
config = {
    'multi_stage': {
        'use_replay': True,
        'replay_buffer_size': 1000,
        'stages': [...]
    }
}

# Knowledge from earlier stages is stored and replayed
# in later stages to prevent catastrophic forgetting
```

### Adaptive Loss Scheduling

```python
config = {
    'multi_stage': {
        'loss_schedule': {
            'initial_weights': {
                'alpha': 0.9,
                'beta': 0.6,
                'gamma': 0.4
            },
            'schedule_type': 'cosine',  # or 'linear', 'exponential'
            'warmup_steps': 100
        },
        'stages': [...]
    }
}

# Loss weights adapt during training:
# - Start with high α (logit distillation)
# - Gradually increase β (feature distillation)
# - Finally focus on γ (similarity transfer)
```

---

## Output & Reporting

### Report Structure

```python
report = multi_stage.run()

# Summary
print(report['summary'])
# {
#     'total_stages': 3,
#     'model_type': 'transformer',
#     'compression_ratio': 2.8,
#     'total_accuracy_gain': 5.2
# }

# Stage details
for stage in report['stages']:
    print(f"Stage {stage['stage']}: {stage['name']}")
    print(f"  Accuracy: {stage['metrics']['val_accuracy']:.2f}%")
    print(f"  Loss: {stage['metrics']['val_loss']:.4f}")

# Final metrics
print(report['final_metrics'])
# {
#     'final_accuracy': 89.5,
#     'total_accuracy_gain': 5.2,
#     'final_loss': 0.23
# }
```

### Checkpoint Management

```python
# Checkpoints saved automatically
# Location: experiments/{id}/stage_{i}_checkpoint.pt

# Load checkpoint
checkpoint = torch.load('experiments/{id}/stage_1_checkpoint.pt')
student.load_state_dict(checkpoint['student_state_dict'])

# Checkpoint contains:
# - student_state_dict: Model weights
# - metrics: Training metrics
# - config: Stage configuration
# - epoch: Final epoch number
```

### Report Files

```
experiments/20251023T064700Z_10c83faa/
├── resolved_config.yaml          # Config used
├── multi_stage_report.json       # Full report (JSON)
├── multi_stage_report.yaml       # Full report (YAML)
├── stage_1_checkpoint.pt         # Stage 1 weights
├── stage_2_checkpoint.pt         # Stage 2 weights
└── stage_3_checkpoint.pt         # Final weights
```

---

## Testing Your Setup

### Quick Test

```python
# Run system integration test
python test_complete_system.py

# Expected output:
# ✅ All core components imported successfully
# ✅ Config Manager working
# ✅ Distiller Registry functional
# ✅ Test models created
# ✅ Individual distillers initialized
# ✅ Multi-Stage methods verified
# ✅ Config validation working
# ✅ Compatibility verified
```

### Pipeline Test

```python
# Run end-to-end pipeline test
python test_multi_stage_pipeline.py

# Expected output:
# ✅ Config parsing successful
# ✅ Individual stages functional
# ✅ Multi-stage sequence executed
# ✅ Config manager integration working
# ✅ Backward compatibility maintained
# ✅ Error handling robust
```

---

## Troubleshooting

### Common Issues

#### 1. Device Mismatch
```python
# Solution: Use ConfigManager
config_mgr = ConfigManager("configs/my_config.yaml")
device = config_mgr.device()  # Auto-detects

# Move models to correct device
teacher = teacher.to(device)
student = student.to(device)
```

#### 2. Missing Config Sections
```yaml
# Ensure all required sections present:
train:
  epochs: 5
  batch_size: 8
  lr: 2e-5

model:
  name: distilbert-base-uncased
  type: transformer

data:
  train_path: data/imdb_train.jsonl
  val_path: data/imdb_val.jsonl
```

#### 3. Dict vs Tensor Outputs
```python
# All distillers now handle:
# - Dict: {'logits': tensor, ...}
# - Object: obj.logits
# - Tensor: tensor directly

# No action needed - automatic handling
```

#### 4. QAT Not Available
```
Warning: QAT not available. Stage 'qat' will be skipped.

# Solution: Install torch-quantization
pip install pytorch-quantization

# Or: Remove QAT stage from config
```

---

## Best Practices

### 1. Stage Ordering

**Recommended sequence**:
1. KD (Logit alignment) - 2-3 epochs
2. Feature (Layer transfer) - 2-3 epochs
3. Similarity (Relational knowledge) - 3-4 epochs
4. Attention (Fine-grained) - 2 epochs
5. QAT (Quantization) - 1-2 epochs

### 2. Loss Weights

**Guidelines**:
- Start with high α (0.7-0.9) for KD
- Use moderate β (0.5-0.7) for features
- Use lower γ (0.3-0.5) for similarity
- Enable adaptive scheduling for best results

### 3. Layer Selection

**For transformers**:
- Use middle-to-late layers for features
- Match corresponding layers (teacher_3 → student_2)
- Use final layer for similarity

### 4. Batch Sizes

**Device-specific**:
- MPS (M2): 8-16
- CUDA (GPU): 32-64
- CPU: 4-8

### 5. Learning Rates

**Typical ranges**:
- KD stage: 5e-5 to 1e-4
- Feature stage: 2e-5 to 5e-5
- Similarity stage: 1e-5 to 3e-5
- Use warmup_steps: 100-500

---

## Summary

✅ **Complete Integration**: All components work together  
✅ **Config Management**: Auto-validation and device detection  
✅ **Multi-Stage Pipeline**: Sequential distillation with checkpointing  
✅ **Flexible**: Use individual distillers or full pipeline  
✅ **Extensible**: Easy to add custom distillers  
✅ **Production Ready**: Comprehensive testing and error handling  

**Happy Distilling! 🎉**
