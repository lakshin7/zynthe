# Zynthe Toolkit - Advanced Optimizer System

## 🎯 Overview

Production-ready optimizer system with:
- **Multi-optimizer support**: AdamW, Adam, SGD, Lion, 8-bit AdamW, 1-bit Adam (future)
- **Phase-aware optimization**: Automatic LR adjustment for distillation/quantization/finetuning
- **Gradient management**: Clipping, centralization, noise injection
- **Adaptive LR tuning**: Based on DEI (Distillation Efficacy Index) and CAS (Compression-Aware Score)
- **Parameter grouping**: Simple and layer-wise learning rate decay
- **Checkpoint support**: Fault-tolerant training
- **Lookahead wrapper**: Improved convergence

---

## 📦 Installation

The optimizer system is already integrated into the Zynthe Toolkit. No additional installation required!

---

## 🚀 Quick Start

### Basic Usage

```python
from training.optimizer import get_optimizer
import torch.nn as nn

# Your model
model = nn.TransformerModel(...)

# Simple usage
optimizer = get_optimizer(
    model,
    lr=2e-5,
    weight_decay=0.01,
    phase='distillation'
)
```

### Advanced Usage with Config

```python
from training.optimizer import OptimizerFactory

config = {
    'optimizer': 'adamw',
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'layer_wise_lr': True,  # Enable layer-wise learning rates
    'beta1': 0.9,
    'beta2': 0.999
}

optimizer = OptimizerFactory.get_optimizer(
    model,
    config,
    phase='distillation'
)
```

---

## 🔧 Integration with Trainer

### Training Loop Pattern

```python
from training.optimizer import (
    OptimizerFactory,
    GradientManager,
    AdaptiveOptimizer
)

# Initialize optimizer
optimizer = OptimizerFactory.get_optimizer(model, config, phase='distillation')

# Wrap with adaptive tuning
adaptive_opt = AdaptiveOptimizer(optimizer, enable_auto_tune=True)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient management
        grad_norm = GradientManager.clip_gradients(model, max_norm=1.0)
        GradientManager.centralize_gradients(model)  # Optional: for distillation
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Scheduler step (if using)
        scheduler.step()
    
    # Evaluate and auto-tune
    metrics = evaluate_model(model, val_loader)
    adaptive_opt.auto_tune(metrics, epoch=epoch)
```

---

## 📝 Configuration File Integration

### YAML Config Example

```yaml
# configs/advanced_optimizer.yaml

train:
  epochs: 10
  batch_size: 16
  
  # Optimizer settings
  optimizer: "adamw"
  learning_rate: 5e-5
  weight_decay: 0.01
  
  # Advanced optimizer features
  layer_wise_lr: true       # Layer-wise learning rate decay
  max_grad_norm: 1.0       # Gradient clipping
  centralize_grads: true    # Gradient centralization for distillation
  
  # Adaptive tuning
  dynamic_lr: true
  patience: 3
  factor: 0.5
  min_lr: 1e-7
  
  # Phase (changes automatically)
  phase: "distillation"     # or "quantization" or "finetuning"

# Optimizer-specific settings
adamw:
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8

sgd:
  momentum: 0.9
  nesterov: true
```

---

## 🎨 Supported Optimizers

### 1. **AdamW** (Default, Recommended)
```python
config = {'optimizer': 'adamw', 'learning_rate': 2e-5, 'weight_decay': 0.01}
```
- Decoupled weight decay
- Best for transformer models
- Memory efficient

### 2. **Adam**
```python
config = {'optimizer': 'adam', 'learning_rate': 2e-5}
```
- Standard Adam optimizer
- Good for general use

### 3. **SGD with Momentum**
```python
config = {
    'optimizer': 'sgd',
    'learning_rate': 1e-3,
    'momentum': 0.9,
    'nesterov': True
}
```
- Good for CNNs
- Requires higher learning rate

### 4. **Lion** (Memory Efficient)
```python
config = {'optimizer': 'lion', 'learning_rate': 1e-4}
```
- 2x more memory efficient than AdamW
- Requires 3x lower learning rate
- Requires: `pip install lion-pytorch`

### 5. **8-bit AdamW** (Large Models)
```python
config = {'optimizer': 'adamw8bit', 'learning_rate': 2e-5}
```
- For very large models
- Requires: `pip install bitsandbytes`

---

## 🧠 Phase-Aware Optimization

The optimizer automatically adjusts learning rates based on the training phase:

| Phase | LR Multiplier | Use Case |
|-------|--------------|----------|
| **Distillation** | 1.0x | Standard knowledge distillation |
| **Quantization** | 0.5x | Lower LR for QAT stability |
| **Finetuning** | 1.5x | Higher LR for final tuning |

```python
# Phase is automatically set from config
optimizer = OptimizerFactory.get_optimizer(model, config, phase='quantization')
# LR will be automatically reduced by 50% for stability
```

---

## 🎯 Gradient Management

### 1. Gradient Clipping (Prevent Exploding Gradients)

```python
from training.optimizer import GradientManager

# After backward pass
grad_norm = GradientManager.clip_gradients(model, max_norm=1.0)
print(f"Gradient norm: {grad_norm:.4f}")
```

### 2. Gradient Centralization (Improve Distillation)

```python
# Improves stability in knowledge distillation
GradientManager.centralize_gradients(model)
```

### 3. Gradient Noise Injection (QAT Robustness)

```python
# Add small noise during quantization-aware training
GradientManager.inject_gradient_noise(model, noise_scale=0.01)
```

### 4. Gradient Statistics (Monitoring)

```python
stats = GradientManager.get_gradient_stats(model)
print(f"Grad norm: {stats['grad_norm']:.4f}")
print(f"Grad mean: {stats['grad_mean']:.4f}")
print(f"Grad std:  {stats['grad_std']:.4f}")
```

---

## 📊 Adaptive LR Tuning

The optimizer can automatically adjust learning rates based on evaluation metrics:

```python
from training.optimizer import AdaptiveOptimizer

adaptive_opt = AdaptiveOptimizer(
    optimizer,
    enable_auto_tune=True,
    patience=3,        # Epochs to wait before reducing
    factor=0.5,        # Multiply LR by this factor
    min_lr=1e-7       # Minimum learning rate
)

# After each epoch
metrics = {
    'dei': 1.5,        # Distillation Efficacy Index
    'cas': 0.3,        # Compression-Aware Score
    'accuracy': 0.92
}

actions = adaptive_opt.auto_tune(metrics, epoch=epoch)
if actions['lr_changed']:
    print(f"LR adjusted: {actions['old_lr']:.2e} → {actions['new_lr']:.2e}")
    print(f"Reason: {actions['action']}")
```

### Tuning Rules

1. **DEI Emergency Reduction**: If DEI < 0.8 → Reduce LR by 50%
2. **CAS Boost**: If CAS improving rapidly (>20%) → Increase LR by 10%
3. **Plateau Detection**: If no improvement for `patience` epochs → Reduce LR by `factor`

---

## 💾 Checkpoint Support

### Save Optimizer State

```python
from training.optimizer import OptimizerCheckpoint

OptimizerCheckpoint.save_checkpoint(
    optimizer,
    path='checkpoints/optimizer_epoch_10.pt',
    epoch=10,
    best_metric=0.95
)
```

### Load Optimizer State

```python
metadata = OptimizerCheckpoint.load_checkpoint(
    optimizer,
    path='checkpoints/optimizer_epoch_10.pt'
)

print(f"Resumed from epoch {metadata['epoch']}")
print(f"Best metric: {metadata['best_metric']}")
```

---

## 🚀 Advanced Features

### Layer-Wise Learning Rate Decay

Earlier layers get lower learning rates (better for transfer learning):

```python
config = {
    'optimizer': 'adamw',
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'layer_wise_lr': True  # Enable layer-wise decay
}

optimizer = OptimizerFactory.get_optimizer(model, config)

# Earlier layers: LR × 0.95^(num_layers - layer_idx)
# Classifier head: Full LR
```

### Lookahead Optimizer

Improves convergence by maintaining "slow" and "fast" weights:

```python
from training.optimizer import LookaheadOptimizer

base_opt = OptimizerFactory.get_optimizer(model, config)
optimizer = LookaheadOptimizer(
    base_opt,
    k=5,           # Update slow weights every 5 steps
    alpha=0.5      # Slow weights step size
)
```

---

## 📈 Example: Complete Training Script

```python
from training.optimizer import (
    OptimizerFactory,
    GradientManager,
    AdaptiveOptimizer,
    OptimizerCheckpoint
)
from training.scheduler import get_scheduler

# Load config
config = load_yaml('configs/advanced.yaml')

# Create optimizer
optimizer = OptimizerFactory.get_optimizer(
    model,
    config['train'],
    phase='distillation'
)

# Create scheduler
scheduler = get_scheduler(
    optimizer,
    config['train'],
    num_training_steps=len(train_loader) * config['train']['epochs']
)

# Wrap with adaptive tuning
adaptive_opt = AdaptiveOptimizer(
    optimizer,
    enable_auto_tune=config['train'].get('dynamic_lr', True)
)

# Training loop
for epoch in range(config['train']['epochs']):
    model.train()
    for batch in train_loader:
        # Forward pass
        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = compute_loss(outputs, batch['labels'])
        
        # Backward pass
        loss.backward()
        
        # Gradient management
        grad_norm = GradientManager.clip_gradients(
            model,
            max_norm=config['train'].get('max_grad_norm', 1.0)
        )
        
        if config['train'].get('centralize_grads', False):
            GradientManager.centralize_gradients(model)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    # Evaluate
    metrics = evaluate(model, val_loader)
    
    # Auto-tune LR
    tune_actions = adaptive_opt.auto_tune(metrics, epoch=epoch)
    
    # Save checkpoint
    if metrics['accuracy'] > best_accuracy:
        OptimizerCheckpoint.save_checkpoint(
            optimizer,
            path=f'checkpoints/best_optimizer.pt',
            epoch=epoch,
            best_metric=metrics['accuracy']
        )
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_optimizer.py
```

Expected output:
```
✅ ALL TESTS PASSED!

The Advanced Optimizer System is production-ready!

Key Features:
  • Multi-optimizer support (AdamW, Adam, SGD, Lion, etc.)
  • Phase-aware optimization
  • Gradient management (clipping, centralization, noise)
  • Adaptive LR tuning based on DEI/CAS metrics
  • Parameter grouping (simple and layer-wise)
  • Checkpoint save/load support
  • Lookahead wrapper for improved convergence
```

---

## 🔬 API Reference

### OptimizerFactory

```python
OptimizerFactory.get_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    phase: str = "distillation"
) -> Optimizer
```

### GradientManager

```python
GradientManager.clip_gradients(model, max_norm=1.0) -> float
GradientManager.centralize_gradients(model) -> None
GradientManager.inject_gradient_noise(model, noise_scale=0.01) -> None
GradientManager.get_gradient_stats(model) -> Dict[str, float]
```

### AdaptiveOptimizer

```python
AdaptiveOptimizer(
    optimizer: Optimizer,
    enable_auto_tune: bool = True,
    patience: int = 3,
    factor: float = 0.5,
    min_lr: float = 1e-7
)

adaptive_opt.auto_tune(metrics: Dict[str, float], epoch: int) -> Dict[str, Any]
```

### OptimizerCheckpoint

```python
OptimizerCheckpoint.save_checkpoint(
    optimizer: Optimizer,
    path: str,
    epoch: int,
    best_metric: float = None
) -> None

OptimizerCheckpoint.load_checkpoint(
    optimizer: Optimizer,
    path: str
) -> Dict[str, Any]
```

---

## 🎓 Best Practices

1. **Use AdamW for transformers**: Best convergence for BERT-like models
2. **Enable gradient clipping**: Prevents training instability
3. **Use layer-wise LR for fine-tuning**: Better transfer learning
4. **Enable adaptive tuning**: Automatically adjusts to training dynamics
5. **Save checkpoints regularly**: Enables fault-tolerant training
6. **Monitor gradient stats**: Catch training issues early

---

## 🐛 Troubleshooting

### High Gradient Norms
```python
# Increase clipping threshold
grad_norm = GradientManager.clip_gradients(model, max_norm=5.0)
```

### Slow Convergence
```python
# Try layer-wise LR or Lookahead
config['layer_wise_lr'] = True
# or
optimizer = LookaheadOptimizer(base_optimizer)
```

### Training Instability
```python
# Use gradient centralization
GradientManager.centralize_gradients(model)
```

---

## 📚 References

- AdamW: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Lion: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)
- Lookahead: [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)
- Gradient Centralization: [Gradient Centralization](https://arxiv.org/abs/2004.01461)

---

## ✅ Status

**Production Ready** - v2.0.0

All tests passing ✅  
Integrated with Zynthe Toolkit ✅  
Documentation complete ✅  

---

**Next Steps**: Integrate into `trainer.py` for automatic usage!
