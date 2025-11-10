# Preflight Analyzer - Automated Pre-Training Analysis

## Overview

The Preflight Analyzer is a comprehensive system that automatically inspects models, validates data, probes hardware resources, and generates optimized configurations before starting knowledge distillation training. It acts as a "pre-flight checklist" to ensure everything is correctly configured and compatible.

## Features

### 🔍 Model Inspector
- **Automatic model type detection**: Vision, NLP, Multimodal, Audio, Video
- **Architecture family classification**: CNN, Transformer, Hybrid, RNN
- **Parameter counting**: Total and trainable parameters
- **Compression ratio calculation**: Teacher/student size comparison
- **Compatibility checking**: Output shapes, architecture mismatches
- **Distillation strategy recommendation**: Based on compression and architecture
- **Auto layer mapping**: Intelligent teacher→student layer alignment

### 📊 Data Inspector
- **Dataset schema validation**: Format and structure checks
- **Task type detection**: Classification, Regression, Generation, QA, etc.
- **Data distribution analysis**: Class balance, sample statistics
- **Batch size recommendations**: Based on dataset size and type
- **Preprocessing suggestions**: Required, recommended, and optional steps
- **Multi-modality support**: Text, vision, audio, multimodal data

### 🖥️ Resource Probe
- **Device detection**: CUDA, MPS (Apple Silicon), CPU
- **Memory profiling**: RAM, VRAM availability
- **Precision support**: FP32, FP16, BF16, INT8, TF32
- **Compute capabilities**: cuDNN, NCCL, AMP support
- **Memory usage estimation**: Model + optimizer + activations
- **Batch size optimization**: Memory-aware recommendations

### ⚙️ Configuration Optimizer
- **Device selection**: Best available accelerator
- **Precision mode**: Mixed precision when beneficial
- **Batch size tuning**: Optimal for hardware and data
- **DataLoader settings**: Workers, pin_memory
- **Distillation method**: Recommended strategy
- **Layer mapping**: Auto-generated alignment

## Quick Start

### Basic Usage

```python
from core.preflight import run_preflight_check

# Run complete preflight analysis
report = run_preflight_check(
    teacher_model=teacher,
    student_model=student,
    dataset=train_dataset,
    config=config,
    save_report=True,
    output_dir="preflight_reports"
)

# Check if ready to proceed
if report['can_proceed']:
    # Use optimized configuration
    optimized_config = report['optimized_config']
    
    batch_size = optimized_config['batch_size']
    device = optimized_config['device']
    precision = optimized_config['precision']
    strategy = optimized_config['distillation_strategy']
    
    # Start training with optimal settings...
else:
    # Fix issues
    print("Blockers:", report['blockers'])
    print("Warnings:", report['warnings'])
```

### Advanced Usage

```python
from core.preflight import PreflightAnalyzer

# Create analyzer
analyzer = PreflightAnalyzer(
    teacher_model=teacher,
    student_model=student,
    dataset=dataset,
    config=config,
    output_dir="preflight_reports"
)

# Run full analysis with verbose output
report = analyzer.run_preflight(verbose=True)

# Save reports in multiple formats
analyzer.save_report(report, format='json')  # Machine-readable
analyzer.save_report(report, format='txt')   # Human-readable
analyzer.save_report(report, format='yaml')  # Config format

# Update config with optimized settings
updated_config = analyzer.update_config(
    save_path='configs/optimized.yaml'
)
```

### Individual Inspectors

```python
from core.preflight import ModelInspector, DataInspector, ResourceProbe

# Model analysis only
model_inspector = ModelInspector(teacher, student)
model_report = model_inspector.inspect()
print(model_inspector.generate_report())

# Data analysis only
data_inspector = DataInspector(dataset)
data_report = data_inspector.validate()
print(data_inspector.generate_report())

# Resource profiling only
resource_probe = ResourceProbe()
resource_profile = resource_probe.probe()
print(resource_probe.generate_report())
```

## Report Structure

### Comprehensive Report

```python
{
    'timestamp': '2025-10-23T02:02:51.123456',
    'can_proceed': True,
    'confidence': 'high',  # high, medium, low
    'blockers': [],  # Critical issues
    'warnings': [],  # Non-critical issues
    'recommendations': [],  # Suggestions
    
    'model_analysis': {
        'teacher': {...},
        'student': {...},
        'compatibility': {...},
        'compression_ratio': 4.8,
        'recommended_strategy': {...},
        'layer_mapping': [...]
    },
    
    'data_analysis': {
        'data_type': 'text',
        'task_type': 'classification',
        'statistics': {...},
        'batch_recommendations': {...}
    },
    
    'resource_profile': {
        'devices': {...},
        'memory': {...},
        'precision': {...},
        'recommendations': {...}
    },
    
    'optimized_config': {
        'device': 'cuda',
        'precision': 'fp16',
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
        'use_amp': True,
        'distillation_strategy': {...},
        'layer_mapping': [...]
    }
}
```

## Examples

### Example 1: BERT Teacher → DistilBERT Student

```python
from transformers import BertModel, BertConfig
from core.preflight import run_preflight_check

# Models
teacher_config = BertConfig(hidden_size=768, num_hidden_layers=12)
teacher = BertModel(teacher_config)

student_config = BertConfig(hidden_size=384, num_hidden_layers=6)
student = BertModel(student_config)

# Run preflight
report = run_preflight_check(teacher, student, dataset)

# Output:
# Teacher: nlp (109.5M params)
# Student: nlp (22.7M params)
# Compression: 4.8x
# Strategy: logit + feature + attention
# Device: cuda
# Precision: bf16
# Batch size: 32
```

### Example 2: ResNet50 → MobileNet

```python
import torchvision.models as models
from core.preflight import run_preflight_check

# Models
teacher = models.resnet50(pretrained=True)
student = models.mobilenet_v2(pretrained=False)

# Run preflight
report = run_preflight_check(teacher, student, dataset)

# Output:
# Teacher: vision (25.6M params)
# Student: vision (3.5M params)
# Compression: 7.3x
# Strategy: multi_stage (logit + feature + hint)
# Device: cuda
# Precision: fp16
# Batch size: 64
```

### Example 3: Config-Only Analysis (No Models Yet)

```python
from core.preflight import PreflightAnalyzer

# Just config and dataset
config = {
    'model': {
        'teacher': 'bert-base-uncased',
        'student': 'distilbert-base-uncased'
    },
    'data': {
        'dataset_path': 'glue/sst2',
        'batch_size': 16
    }
}

analyzer = PreflightAnalyzer(config=config, dataset=dataset)
report = analyzer.run_preflight()

# Will analyze data and resources, recommend settings
```

## Output Files

The preflight analyzer generates several files:

1. **JSON Report** (`preflight_report_TIMESTAMP.json`)
   - Machine-readable
   - Complete analysis results
   - Easy to parse programmatically

2. **Text Report** (`preflight_report_TIMESTAMP.txt`)
   - Human-readable
   - Formatted for easy review
   - Includes all detailed analyses

3. **Optimized Config** (`optimized_config.yaml`)
   - Updated configuration file
   - Ready to use for training
   - Contains optimal settings

## Common Issues and Solutions

### Issue 1: Incompatible Output Shapes

**Problem:**
```
Blocker: Output shape mismatch: teacher=[768], student=[384]
```

**Solution:**
```python
# Add projection layer to student
import torch.nn as nn

class StudentWithProjection(nn.Module):
    def __init__(self, student, teacher_dim=768):
        super().__init__()
        self.student = student
        self.projection = nn.Linear(384, 768)
    
    def forward(self, x):
        out = self.student(x)
        return self.projection(out)

student = StudentWithProjection(student)
```

### Issue 2: Class Imbalance Detected

**Problem:**
```
Warning: Severe class imbalance detected (ratio: 15.2:1)
```

**Solution:**
```python
from torch.utils.data import WeightedRandomSampler

# Compute class weights
class_counts = report['data_analysis']['statistics']['class_distribution']
class_weights = [1.0 / count for count in class_counts.values()]

# Use weighted sampler
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

### Issue 3: Insufficient Memory

**Problem:**
```
Warning: Estimated memory usage (24.5 GB) exceeds available (16.0 GB)
```

**Solution:**
```python
# Reduce batch size
optimal_batch = report['optimized_config']['batch_size']
reduced_batch = optimal_batch // 2

# Or use gradient accumulation
accumulation_steps = 2
effective_batch = reduced_batch * accumulation_steps
```

## Integration with Training Pipeline

```python
from core.preflight import run_preflight_check
from training.trainer import KDTrainer

# 1. Run preflight check
print("Running preflight analysis...")
report = run_preflight_check(
    teacher_model=teacher,
    student_model=student,
    dataset=train_dataset,
    config=config,
    save_report=True
)

# 2. Check if can proceed
if not report['can_proceed']:
    print(f"❌ Cannot proceed: {report['blockers']}")
    exit(1)

# 3. Use optimized configuration
opt = report['optimized_config']

print(f"✅ Preflight passed!")
print(f"   Device: {opt['device']}")
print(f"   Batch size: {opt['batch_size']}")
print(f"   Strategy: {opt['distillation_strategy']['primary_method']}")

# 4. Start training with optimal settings
trainer = KDTrainer(
    teacher=teacher,
    student=student,
    train_dataset=train_dataset,
    device=opt['device'],
    batch_size=opt['batch_size'],
    use_amp=opt['use_amp'],
    distillation_method=opt['distillation_strategy'],
    layer_mapping=opt.get('layer_mapping')
)

trainer.train()
```

## API Reference

### PreflightAnalyzer

```python
class PreflightAnalyzer:
    def __init__(
        self,
        teacher_model: Optional[nn.Module] = None,
        student_model: Optional[nn.Module] = None,
        dataset: Optional[Dataset] = None,
        config: Optional[Dict] = None,
        output_dir: Optional[str] = None
    )
    
    def run_preflight(self, verbose: bool = True) -> Dict[str, Any]
    def save_report(self, report: Dict, format: str = 'json') -> Path
    def update_config(self, config: Dict = None, save_path: str = None) -> Dict
```

### ModelInspector

```python
class ModelInspector:
    def __init__(
        self,
        teacher: Optional[nn.Module] = None,
        student: Optional[nn.Module] = None
    )
    
    def inspect(self) -> Dict[str, Any]
    def generate_report(self) -> str
```

### DataInspector

```python
class DataInspector:
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        config: Optional[Dict] = None
    )
    
    def validate(self) -> Dict[str, Any]
    def generate_report(self) -> str
```

### ResourceProbe

```python
class ResourceProbe:
    def probe(self) -> Dict[str, Any]
    def estimate_memory_usage(
        self,
        model_params: int,
        batch_size: int,
        sequence_length: Optional[int] = None,
        precision: str = 'fp32'
    ) -> Dict[str, float]
    def recommend_optimal_batch_size(
        self,
        model_params: int,
        available_memory: float,
        sequence_length: Optional[int] = None,
        precision: str = 'fp32'
    ) -> Dict[str, Any]
    def generate_report(self) -> str
```

## Testing

Run the comprehensive test suite:

```bash
python test_preflight.py
```

Tests cover:
- Model inspection (Vision, NLP, Multimodal)
- Data validation (Classification, Generation, QA)
- Resource probing (CUDA, MPS, CPU)
- Full integration analysis
- Edge cases and error handling

## Performance

- Model inspection: < 1 second
- Data validation: < 5 seconds (for 10K samples)
- Resource probing: < 1 second
- Full analysis: < 10 seconds total

## Requirements

- PyTorch >= 1.10
- transformers (optional, for NLP models)
- torchvision (optional, for vision models)
- psutil (for system resource detection)

## Future Enhancements

- [ ] Distributed training configuration
- [ ] Quantization-aware recommendations
- [ ] Model pruning compatibility checks
- [ ] Benchmark performance predictions
- [ ] Cloud deployment recommendations
- [ ] Cost estimation for cloud training
- [ ] Automatic hyperparameter tuning suggestions

## Support

For issues or questions:
1. Check the test suite (`test_preflight.py`) for examples
2. Review the generated reports for detailed diagnostics
3. Consult the main documentation in `docs/`

---

**Note**: The preflight analyzer is designed to catch issues early and optimize configurations automatically. Always run it before starting long training runs to save time and resources!
