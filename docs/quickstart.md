# Quickstart Guide

## Installation

```bash
pip install zynthe
```

For vision model support:
```bash
pip install zynthe[vision]
```

For evaluation & benchmarking:
```bash
pip install zynthe[eval]
```

For everything:
```bash
pip install zynthe[all]
```

## Your First Distillation

### Option 1: DistillationToolkit (Recommended)

The simplest way to run distillation — select a goal and go:

```python
from zynthe import DistillationToolkit

# Load your teacher and student models (HuggingFace)
from transformers import AutoModelForSequenceClassification

teacher = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
student = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Create toolkit and build a plan
toolkit = DistillationToolkit(teacher, student)
plan = toolkit.build_plan(goal="balanced")

# Run distillation
report = toolkit.run(plan, train_loader, val_loader, output_dir="./output")
```

Available goals: `"quick"`, `"balanced"`, `"full"`, `"compression"`, `"vision"`, `"causal_lm"`.

### Option 2: Config-Driven Pipeline

For more control, use YAML configs with the runtime:

```python
from zynthe import UnifiedTrainingRuntime, RuntimeOptions

runtime = UnifiedTrainingRuntime()
options = RuntimeOptions(config_path="configs/default.yaml")

result = runtime.run(options)
print(f"Success: {result.success}")
print(f"Experiment: {result.experiment_id}")
```

### Option 3: Individual Distillers

Use specific distillation methods directly:

```python
from zynthe.core.distillers import KDHintonDistiller

distiller = KDHintonDistiller(
    teacher=teacher,
    student=student,
    temperature=4.0,
    alpha=0.7,
)
loss = distiller.compute_loss(student_logits, teacher_logits, labels)
```

## Multi-Modality Support

Zynthé auto-detects model architectures across 5 modalities:

```python
from zynthe import AdapterRegistry

registry = AdapterRegistry()
adapter = registry.detect(model)  # Returns text/code/vision/multimodal/vlm adapter

# Or explicitly:
adapter = registry.get("vision")
```

## Available Distillation Methods

| Method | Import | Use Case |
|--------|--------|----------|
| Hinton KD | `KDHintonDistiller` | Classic logit-based distillation |
| Attention Transfer | `AttentionTransferDistiller` | Transfer attention patterns |
| Feature Distillation | `FeatureDistiller` | Match intermediate features |
| Similarity Transfer | `SimilarityTransfer` | Match pairwise similarity |
| Causal LM | `SafeCausalLMTrainer` | Language model distillation |

## Configuration

See the `configs/` directory for YAML examples covering each method.

Key config sections:
- `model`: Teacher/student model names and architecture type
- `data`: Dataset paths and loading options
- `distillation`: Method selection and hyperparameters
- `train`: Epochs, batch size, learning rate, optimizer
- `quantization`: Optional PTQ/QAT post-distillation

## Next Steps

- [API Reference](api_reference.md) — Full public API documentation
- [Distillation Methods](distillation_methods.md) — Deep dive into each method
- [Adapters](adapters.md) — Multi-modality adapter system
