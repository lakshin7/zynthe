# API Reference

## Primary Entry Point

### `DistillationToolkit`

The recommended way to use Zynthé. Wraps the multi-stage distillation pipeline with sensible defaults.

```python
from zynthe import DistillationToolkit

toolkit = DistillationToolkit(teacher, student)
plan = toolkit.build_plan(goal="balanced")
report = toolkit.run(plan, train_loader, val_loader)
```

**Parameters:**
- `teacher` — Teacher PyTorch model
- `student` — Student PyTorch model

**Methods:**
- `build_plan(goal)` — Build a multi-stage distillation plan. Goals: `"quick"`, `"balanced"`, `"full"`, `"compression"`, `"vision"`, `"causal_lm"`
- `run(plan, train_loader, val_loader, **kwargs)` — Execute the plan
- `preview(plan)` — Show what the plan will do without executing

---

## Distillers

Individual distillation strategies that can be used standalone.

### `KDHintonDistiller`

Classic Hinton knowledge distillation using softened logits.

```python
from zynthe.core.distillers import KDHintonDistiller
```

**Key Parameters:**
- `temperature` (float) — Softmax temperature (default: 4.0)
- `alpha` (float) — Weight for distillation loss vs task loss (default: 0.7)

### `AttentionTransferDistiller`

Transfers attention map patterns from teacher to student.

```python
from zynthe.core.distillers import AttentionTransferDistiller
```

### `FeatureDistiller`

Matches intermediate feature representations between teacher and student.

```python
from zynthe.core.distillers import FeatureDistiller
```

### `SimilarityTransfer`

Matches pairwise similarity structures in feature spaces.

```python
from zynthe.core.distillers import SimilarityTransfer
```

### `SafeCausalLMTrainer`

Specialized trainer for causal language model distillation with built-in safety checks, checkpointing, and regression gates.

```python
from zynthe.core.distillers import SafeCausalLMTrainer
```

---

## Adapters

Multi-modality support via the adapter system.

### `AdapterRegistry`

Central registry for auto-detecting and managing model adapters.

```python
from zynthe import AdapterRegistry

registry = AdapterRegistry()
adapter = registry.detect(model)     # Auto-detect
adapter = registry.get("vision")     # Explicit
registry.list_available()            # ['code', 'multimodal', 'text', 'vision', 'vlm']
```

### Built-in Adapters

| Adapter | Modality | Supported Architectures |
|---------|----------|------------------------|
| `TextModelAdapter` | `text` | BERT, RoBERTa, GPT-2, LLaMA, Mistral, T5, BART |
| `CodeModelAdapter` | `code` | CodeBERT, CodeLlama, StarCoder, DeepSeek-Coder |
| `VisionModelAdapter` | `vision` | ViT, BEiT, Swin, DeiT, ConvNeXt |
| `MultimodalModelAdapter` | `multimodal` | CLIP, SigLIP, BLIP, FLAVA |
| `VLMModelAdapter` | `vlm` | LLaVA, InternVL, Qwen-VL, Phi-Vision |

### Custom Adapters

```python
from zynthe.core.adapters import ModelAdapter, AdapterRegistry

class MyVideoAdapter(ModelAdapter):
    modality = "video"
    # implement prepare_batch, extract_outputs, get_hookable_layers, align_dimensions

registry = AdapterRegistry()
registry.register("video", MyVideoAdapter, priority=0)
```

---

## Configuration

### `ConfigManager`

YAML-based configuration with dot-notation access and runtime overrides.

```python
from zynthe import ConfigManager

config = ConfigManager("configs/default.yaml")
config.set("train.epochs", 5)
config.set("train.lr", 1e-4)
print(config.device())
```

---

## Model Loading

### `load_models`

Convenience function for loading teacher/student model pairs.

```python
from zynthe import load_models

teacher, student, tokenizer = load_models(
    config,
    device="cuda",
)
```

### `ModelLoader`

Full-featured loader with quantization, compilation, and adapter support.

```python
from zynthe import ModelLoader

loader = ModelLoader(config, device="cuda")
bundle = loader.load(return_bundle=True)
# bundle.teacher, bundle.student, bundle.tokenizer, bundle.metadata
```

---

## Runtime

### `UnifiedTrainingRuntime`

Config-driven orchestration engine for end-to-end distillation runs.

```python
from zynthe import UnifiedTrainingRuntime, RuntimeOptions

runtime = UnifiedTrainingRuntime()
options = RuntimeOptions(config_path="configs/default.yaml")
result = runtime.run(options)
```

**RuntimeResult fields:**
- `success` (bool) — Whether the run completed
- `experiment_id` (str) — Unique experiment identifier
- `metrics` (dict) — Final metrics
- `error` (str | None) — Error message if failed

---

## Presets

Pre-configured distillation strategies.

```python
from zynthe import list_presets, get_preset, describe_preset

list_presets()          # ['quick', 'balanced', 'full', ...]
describe_preset("balanced")
config = get_preset("balanced")
```
