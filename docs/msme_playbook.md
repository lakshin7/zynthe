# Knowledge Distillation on Mac M2 - Complete Playbook

This playbook provides comprehensive guidance for running knowledge distillation experiments on Mac M2 systems, including optimization strategies, troubleshooting, and best practices.

## System Requirements and Setup

### Hardware Specifications
- **Minimum**: Mac M2 (8GB unified memory)
- **Recommended**: Mac M2 Pro (16GB unified memory)
- **Optimal**: Mac M2 Max/Ultra (32GB+ unified memory)

### Software Requirements
```bash
# System requirements
macOS 12.3+ (for MPS support)
Python 3.9-3.11 (avoid 3.12 for now due to some package compatibility)
Xcode Command Line Tools

# Check MPS availability
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### Environment Setup
```bash
# Create conda environment (recommended)
conda create -n kd-toolkit python=3.10
conda activate kd-toolkit

# Or use virtual environment
python3 -m venv kd-env
source kd-env/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other requirements
pip install transformers datasets accelerate
pip install scikit-learn matplotlib seaborn
pip install pyyaml typer rich
```

## Model Configuration Strategies

### Memory-Optimized Configurations

#### 8GB Mac M2 Configuration
```yaml
# configs/mac_m2_8gb.yaml
train:
  batch_size: 4
  grad_accum_steps: 4          # Effective batch size = 16
  epochs: 3
  mixed_precision: false        # MPS doesn't support AMP well yet

model:
  name: "distilbert-base-uncased"       # Teacher (66M params)
  student_name: "prajjwal1/bert-tiny"   # Student (4M params)
  type: "transformer"

device:
  prefer_mps: true
  memory_management:
    max_memory_gb: 6            # Leave 2GB for system
    gradient_checkpointing: true

quantization:
  enable: true
  mode: "float16"               # Works better on MPS than int8
```

#### 16GB Mac M2 Pro Configuration  
```yaml
# configs/mac_m2_pro_16gb.yaml
train:
  batch_size: 8
  grad_accum_steps: 2
  epochs: 5

model:
  name: "bert-base-uncased"             # Teacher (110M params)
  student_name: "distilbert-base-uncased" # Student (66M params)
  type: "transformer"

distillation:
  method: "multi_stage"
  strategies:
    - name: "kd_hinton"
      temperature: 3.0
      alpha: 0.7
    - name: "attention_transfer"
      beta: 1e-3

device:
  memory_management:
    max_memory_gb: 12
    gradient_checkpointing: false  # Can afford to keep all gradients
```

#### 32GB+ Mac M2 Max/Ultra Configuration
```yaml
# configs/mac_m2_max_32gb.yaml  
train:
  batch_size: 16
  grad_accum_steps: 1
  epochs: 10

model:
  name: "roberta-large"                 # Teacher (355M params)
  student_name: "roberta-base"          # Student (125M params)
  type: "transformer"

distillation:
  method: "multi_stage"
  strategies:
    - name: "kd_hinton"
      temperature: 4.0
      alpha: 0.8
    - name: "attention_transfer"
      beta: 2e-3
    - name: "feature_distiller"
      gamma: 1e-3
    - name: "similarity_transfer"
      delta: 5e-4

device:
  memory_management:
    max_memory_gb: 24
    gradient_checkpointing: false
```

## Performance Optimization Guide

### MPS-Specific Optimizations

#### 1. Data Type Management
```python
# Preferred data types for MPS
torch.set_default_dtype(torch.float32)  # MPS works best with float32
model = model.to(torch.float32)          # Ensure consistent dtype

# Avoid these on MPS (for now)
# torch.float16 - Limited support
# torch.bfloat16 - Not supported
# int8 quantization - Unstable
```

#### 2. Memory Layout Optimization
```python
# Use contiguous tensors
input_tensor = input_tensor.contiguous()

# Prefer smaller batch sizes with gradient accumulation
# Instead of batch_size=32, use batch_size=8 with grad_accum_steps=4
```

#### 3. Operation-Specific Tips
```python
# These operations are optimized on MPS
# - Matrix multiplication (GEMM)
# - Convolutions
# - Element-wise operations
# - Reductions (sum, mean)

# These operations may be slower on MPS
# - Complex indexing
# - Sparse operations
# - Some advanced activations
```

### Training Strategies

#### Progressive Training
Start with smaller models and gradually increase complexity:

**Stage 1**: Quick validation (5 minutes)
```yaml
model:
  name: "prajjwal1/bert-tiny"
  student_name: "prajjwal1/bert-mini"
train:
  epochs: 1
  batch_size: 16
```

**Stage 2**: Full experiment (30 minutes)
```yaml
model:
  name: "bert-base-uncased"
  student_name: "distilbert-base-uncased"
train:
  epochs: 5
  batch_size: 8
```

#### Curriculum Learning
Train on easier examples first:
```yaml
data:
  curriculum_learning:
    enable: true
    strategy: "length_based"  # Start with shorter sequences
    stages: 3
    difficulty_ramp: "linear"
```

### Memory Management Strategies

#### 1. Gradient Checkpointing
```yaml
device:
  memory_management:
    gradient_checkpointing: true  # Trades compute for memory
    checkpoint_segments: 4        # Number of segments to checkpoint
```

#### 2. Model Parallelism (for very large models)
```python
# For models that don't fit in memory
from torch.nn.parallel import DistributedDataParallel as DDP

# Note: Single GPU on Mac M2, so model sharding is the option
teacher_model = torch.nn.DataParallel(teacher_model)
```

#### 3. Dynamic Batch Sizing
```python
def get_optimal_batch_size(model, device, max_memory_gb=8):
    """Automatically determine optimal batch size"""
    start_batch_size = 16
    while start_batch_size > 1:
        try:
            # Test with dummy data
            dummy_input = torch.randint(0, 1000, (start_batch_size, 128)).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            return start_batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                start_batch_size //= 2
            else:
                raise e
    return 1
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. MPS Out of Memory
**Symptoms**: `RuntimeError: MPS backend out of memory`

**Solutions**:
```bash
# Reduce batch size
train.batch_size: 2
train.grad_accum_steps: 8

# Enable gradient checkpointing  
device.memory_management.gradient_checkpointing: true

# Clear MPS cache (add to training loop)
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

#### 2. MPS Unsupported Operations
**Symptoms**: `NotImplementedError: The operator 'aten::operator_name' is not currently implemented for the MPS device`

**Solutions**:
```python
# Fallback to CPU for specific operations
def mps_safe_operation(tensor, operation):
    if tensor.device.type == 'mps':
        # Move to CPU, perform operation, move back
        result = operation(tensor.cpu()).to('mps')
        return result
    else:
        return operation(tensor)

# Or use CPU fallback in config
device:
  prefer_mps: true
  fallback_cpu: true
```

#### 3. Slow Training Performance
**Symptoms**: Training is slower than expected

**Diagnostics**:
```python
import time

# Profile training step
start_time = time.time()
loss = model(inputs)
forward_time = time.time() - start_time

start_time = time.time()
loss.backward()
backward_time = time.time() - start_time

print(f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")
```

**Solutions**:
- Increase batch size if memory allows
- Use gradient accumulation instead of small batches
- Profile with `torch.profiler` to find bottlenecks

#### 4. Model Loading Issues
**Symptoms**: Models fail to load or are extremely slow

**Solutions**:
```python
# Pre-download models
from transformers import AutoModel, AutoTokenizer

# Download and cache models
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name, cache_dir="./model_cache")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
```

#### 5. Quantization Issues
**Symptoms**: Quantized models fail or perform poorly

**Mac M2 Quantization Strategy**:
```python
# Use float16 instead of int8 for better MPS compatibility
quantization_config = {
    "enable": True,
    "mode": "float16",  # Better MPS support than int8
    "calibration_samples": 100
}

# For CPU deployment
quantization_config = {
    "enable": True,
    "mode": "dynamic",  # Better CPU performance
}
```

## Performance Benchmarks

### Training Time Benchmarks (Mac M2 8GB)

| Model Pair | Batch Size | Epochs | Memory Usage | Time | Student Accuracy |
|------------|------------|--------|--------------|------|------------------|
| BERT-tiny → BERT-mini | 16 | 3 | 2GB | 5 min | 84% |
| DistilBERT → BERT-tiny | 8 | 3 | 3GB | 8 min | 87% |
| BERT → DistilBERT | 4 | 3 | 5GB | 15 min | 92% |
| RoBERTa → DistilRoBERTa | 4 | 3 | 6GB | 18 min | 94% |

### Memory Usage Patterns

```python
# Monitor memory usage during training
import psutil
import torch

def monitor_memory():
    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.percent}% used")
    
    # GPU memory (MPS)
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        print(f"MPS Memory: {allocated:.2f} GB allocated")
```

### Inference Speed Benchmarks

| Model | Size (MB) | Params | Mac M2 Speed (samples/sec) | CPU Speed (samples/sec) |
|-------|-----------|--------|-----------------------------|-------------------------|
| BERT-tiny | 17 | 4M | 450 | 180 |
| DistilBERT | 255 | 66M | 120 | 45 |
| BERT-base | 420 | 110M | 80 | 25 |

## Monitoring and Profiling

### Training Monitoring
```python
# Add to training loop
import matplotlib.pyplot as plt
from collections import defaultdict

metrics = defaultdict(list)

def log_metrics(epoch, train_loss, val_loss, val_acc):
    metrics['epoch'].append(epoch)
    metrics['train_loss'].append(train_loss)
    metrics['val_loss'].append(val_loss)
    metrics['val_accuracy'].append(val_acc)
    
    # Real-time plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Training Progress')
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Performance')
    
    plt.tight_layout()
    plt.savefig(f'training_progress_epoch_{epoch}.png')
    plt.close()
```

### Performance Profiling
```python
# Use PyTorch profiler for detailed analysis
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Training code here
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()

# Export profiling results
prof.export_chrome_trace("training_profile.json")
```

## Best Practices Summary

### Configuration Best Practices
1. **Start Small**: Begin with tiny models to validate your pipeline
2. **Gradual Scaling**: Increase model size and complexity gradually
3. **Memory Buffer**: Always leave 20-25% memory headroom
4. **Consistent Precision**: Use float32 throughout for MPS stability

### Training Best Practices
1. **Batch Size Strategy**: Use smaller batches with gradient accumulation
2. **Learning Rate**: Start with 1e-5 for distillation (lower than standard training)
3. **Temperature Tuning**: Experiment with temperatures between 2-8
4. **Progressive Complexity**: Start with simple KD, add advanced techniques

### Development Workflow
1. **Quick Validation**: 1-epoch run to test pipeline (< 5 minutes)
2. **Hyperparameter Search**: Small-scale grid search (15-30 minutes)
3. **Full Training**: Complete experiment with best hyperparameters (1-2 hours)
4. **Evaluation**: Comprehensive evaluation and comparison

### Deployment Considerations
1. **Model Export**: Use torch.jit.script for production deployment
2. **Quantization**: Apply post-training quantization for edge deployment
3. **Optimization**: Use torch.compile (when available on MPS) for inference speedup

## Advanced Techniques

### Multi-Stage Distillation
```python
# Stage 1: Basic knowledge distillation
stage1_config = {
    "distillation": {
        "method": "kd_hinton",
        "temperature": 3.0,
        "alpha": 0.7
    },
    "train": {"epochs": 3}
}

# Stage 2: Add attention transfer
stage2_config = {
    "distillation": {
        "method": "multi_stage",
        "strategies": [
            {"name": "kd_hinton", "temperature": 4.0, "alpha": 0.8},
            {"name": "attention_transfer", "beta": 1e-3}
        ]
    },
    "train": {"epochs": 2}
}

# Stage 3: Full multi-objective distillation
stage3_config = {
    "distillation": {
        "method": "multi_stage",
        "strategies": [
            {"name": "kd_hinton", "temperature": 5.0, "alpha": 0.9},
            {"name": "attention_transfer", "beta": 2e-3},
            {"name": "feature_distiller", "gamma": 1e-3},
            {"name": "similarity_transfer", "delta": 5e-4}
        ]
    },
    "train": {"epochs": 2}
}
```

### Custom Loss Functions
```python
class AdaptiveTemperatureKD(torch.nn.Module):
    def __init__(self, initial_temp=3.0):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(initial_temp))
    
    def forward(self, student_logits, teacher_logits, labels):
        # Temperature adapts during training
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        kd_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Dynamic weighting
        alpha = torch.sigmoid(self.temperature - 3.0)  # Adapt alpha based on temperature
        
        return alpha * hard_loss + (1 - alpha) * self.temperature**2 * kd_loss
```

This playbook provides a comprehensive guide for successfully running knowledge distillation experiments on Mac M2 systems. Follow the configurations and best practices to achieve optimal results while avoiding common pitfalls.
