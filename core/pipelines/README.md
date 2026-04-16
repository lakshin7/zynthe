# Zynthe Pipeline System

## Overview

The Pipeline System provides a modular, composable architecture for knowledge distillation in Zynthe. It enables flexible combination of multiple distillation techniques with different execution strategies.

## Key Features

- 🔌 **Pluggable Architecture**: Mix and match distillers dynamically
- ⚡ **Flexible Execution**: Sequential, parallel, conditional, or hybrid modes
- 🎯 **Memory Optimized**: Designed for Google Colab T4 GPU (16GB VRAM)
- 🛠️ **Fluent API**: Easy-to-use builder pattern
- 📊 **Observable**: Per-stage metrics and profiling
- 🔄 **Backward Compatible**: Works with existing distillers

## Quick Start

### Single Distiller Pipeline

```python
from core.pipelines import PipelineBuilder

# Build a simple KD-Hinton pipeline
pipeline = PipelineBuilder() \
    .add_distiller('kd_hinton', temperature=4.0, alpha=0.7) \
    .build(teacher, student, device)

# Use it
pipeline.setup()
loss, metrics = pipeline(batch)
```

### Multi-Stage Pipeline

```python
# Combine multiple distillers
pipeline = PipelineBuilder() \
    .add_stage('logit_distillation', weight=0.7) \
        .add_distiller('kd_hinton', temperature=4.0) \
    .add_stage('feature_matching', weight=0.3) \
        .add_distiller('feature', layers=[6, 8]) \
        .add_distiller('attention', layers=[8]) \
    .with_mode('sequential') \
    .build(teacher, student, device)
```

### From Configuration

```python
config = {
    'distillation': {
        'pipeline': {
            'type': 'multi_stage',
            'mode': 'hybrid',
            'stages': [
                {
                    'name': 'logit_stage',
                    'weight': 0.6,
                    'distillers': [
                        {'type': 'kd_hinton', 'config': {'temperature': 4.0}}
                    ]
                },
                {
                    'name': 'feature_stage',
                    'weight': 0.4,
                    'mode': 'parallel',
                    'distillers': [
                        {'type': 'feature', 'config': {'layers': [6, 8]}},
                        {'type': 'attention', 'config': {'layers': [8]}}
                    ]
                }
            ]
        }
    }
}

pipeline = PipelineBuilder.from_config(config, teacher, student, device)
```

## Architecture

```
Pipeline System
├── BasePipeline (Abstract)
│   ├── setup() - Initialize components
│   ├── forward() - Execute on batch
│   ├── compute_loss() - Calculate loss
│   └── get_metrics() - Collect metrics
│
├── SingleDistillerPipeline
│   └── Wraps individual distillers (KD-Hinton, Feature, etc.)
│
├── MultiStagePipeline
│   ├── Execution Modes:
│   │   ├── Sequential - Chain distillers in order
│   │   ├── Parallel - Run independently, aggregate
│   │   ├── Conditional - Route based on conditions
│   │   └── Hybrid - Mix sequential + parallel
│   │
│   └── Stages (each with weight, mode, distillers)
│
├── PipelineBuilder
│   ├── Fluent API for construction
│   └── Configuration-based building
│
└── PipelineRegistry
    └── Discovery and instantiation
```

## Execution Modes

### Sequential
Runs stages one after another. Best for:
- Curriculum learning
- Progressive distillation
- Memory-constrained environments

### Parallel
Runs all stages simultaneously. Best for:
- Multiple independent objectives
- Balanced multi-task learning

### Conditional
Routes based on runtime conditions. Best for:
- Dynamic distillation strategies
- Adaptive learning

### Hybrid
Combines sequential and parallel within stages. Best for:
- Complex multi-objective optimization
- Fine-grained control

## Available Distillers

- `kd_hinton` - Classical logit distillation
- `feature` - Intermediate feature matching
- `attention` - Attention map transfer
- `similarity` - Gram matrix similarity

## Configuration Schema

```yaml
distillation:
  pipeline:
    type: multi_stage  # or 'single'
    mode: hybrid       # or 'sequential', 'parallel', 'conditional'
    
    stages:
      - name: logit_distillation
        weight: 0.7
        distillers:
          - type: kd_hinton
            config:
              temperature: 4.0
              alpha: 0.7
      
      - name: feature_matching
        weight: 0.3
        mode: parallel  # Stage-specific mode
        distillers:
          - type: feature
            config:
              layers: [6, 8, 10]
          - type: attention
            config:
              layers: [8, 10]
```

## Google Colab Testing

A complete testing notebook is provided:

📓 **[notebooks/pipeline_colab_test.ipynb](../notebooks/pipeline_colab_test.ipynb)**

Open in Colab and test all pipeline features on T4 GPU:
- Single distiller pipelines
- Multi-stage pipelines
- Configuration-based building
- Memory profiling
- Training integration

## API Reference

### PipelineBuilder

```python
builder = PipelineBuilder()

# Add stages
builder.add_stage(name: str, weight: float = 1.0, mode: str = None)

# Add distillers to current stage
builder.add_distiller(distiller_type: str, **config)

# Set execution mode
builder.with_mode(mode: str)  # 'sequential', 'parallel', 'conditional', 'hybrid'

# Set configuration
builder.with_config(config: dict)

# Build pipeline
pipeline = builder.build(teacher, student, device)
```

### Pipeline Interface

```python
# Setup (called once)
pipeline.setup()

# Forward pass (called per batch)
loss, metrics = pipeline(batch)

# Get metrics
metrics = pipeline.get_metrics()
# Returns: PipelineMetrics with total_loss, component_losses, execution_time_ms, memory_allocated_mb

# Cleanup (called after training)
pipeline.cleanup()
```

## Performance Tips for T4 GPU

1. **Enable Mixed Precision**: Reduces memory by 2x
2. **Gradient Checkpointing**: Set `checkpoint_gradients: true`
3. **Batch Size**: Start with 8-16 for transformer models
4. **Stage Weights**: Normalize weights to prevent gradient explosion
5. **Sequential Mode**: Use for memory-constrained scenarios

## Examples

### Example 1: Progressive Distillation

```python
# Start with logits, then add feature matching
pipeline = PipelineBuilder() \
    .add_stage('early_training', weight=1.0) \
        .add_distiller('kd_hinton') \
    .add_stage('late_training', weight=0.5, condition=lambda b, o: epoch > 5) \
        .add_distiller('feature') \
    .with_mode('conditional') \
    .build(teacher, student, device)
```

### Example 2: Ensemble of Distillers

```python
# Combine multiple distillers in parallel
pipeline = PipelineBuilder() \
    .add_stage('ensemble', weight=1.0, mode='parallel') \
        .add_distiller('kd_hinton', temperature=4.0) \
        .add_distiller('feature', layers=[4, 6, 8]) \
        .add_distiller('attention', layers=[6, 8]) \
    .build(teacher, student, device)
```

### Example 3: Hybrid Strategy

```python
# Logits first, then parallel feature/attention
pipeline = PipelineBuilder() \
    .add_stage('stage1_logits', weight=0.5) \
        .add_distiller('kd_hinton') \
    .add_stage('stage2_features', weight=0.5, mode='parallel') \
        .add_distiller('feature', layers=[6, 8]) \
        .add_distiller('attention', layers=[8]) \
    .with_mode('hybrid') \
    .build(teacher, student, device)
```

## Roadmap

- [ ] Pipeline optimization engine (auto-suggest distillers)
- [ ] Auto-weight tuning based on loss contributions
- [ ] Conditional routing based on validation metrics
- [ ] Multi-GPU support
- [ ] Pipeline visualization
- [ ] Performance profiling dashboard

## Contributing

To add a new distiller to the pipeline system:

1. Implement your distiller extending `BaseDistiller`
2. Register it in `DistillerRegistry`
3. It automatically works with the pipeline system!

No additional integration needed - the `SingleDistillerPipeline` wrapper handles everything.

## License

Part of the Zynthe project. See main LICENSE file.
