---
# Knowledge Distillation & Quantization Toolkit: Enterprise Overview

## Purpose
This toolkit provides a robust, extensible, and enterprise-ready framework for knowledge distillation and quantization of deep learning models. It enables efficient transfer of knowledge from large teacher models to smaller student models, with support for advanced distillation strategies and quantization workflows.

## Key Features
- **Enterprise-Ready Structure:** Modular, type-annotated, and logging-enabled codebase.
- **Student/Teacher Initialization:** Unified model loading and configuration via `ConfigManager` and `load_models`.
- **Distillation Strategies:** Supports Hinton KD, Attention Transfer, Feature Distillation, and Similarity Transfer.
- **Quantization:** Post-training quantization (PTQ) and hooks for QAT.
- **Extensibility:** Easy to add new distillers, models, or quantization methods.
- **Testing & Examples:** Minimal working examples and automated tests for reliability.
- **Documentation:** Clear, maintainable, and up-to-date docs for onboarding and scaling.

## Workflow Overview
1. **Configuration:**
   - All experiments are driven by YAML config files (see `configs/default.yaml`).
   - Device, model, distillation, data, and quantization settings are centrally managed.

2. **Model Initialization:**
   - Use `ConfigManager` to load and resolve configs.
   - `load_models(cfg, device)` loads both teacher and student models, as well as the tokenizer, on the correct device.
   - Models are wrapped with `ModelWrapper` for unified operations (forward, save, quantize, etc.).

3. **Distillation:**
   - Choose a distillation strategy (e.g., Hinton KD, Attention, Feature, Similarity) via config.
   - Use the appropriate distiller class (e.g., `KDHintonDistiller`) to compute distillation loss.
   - The `MultiStageDistiller` enables chaining multiple strategies for advanced workflows.

4. **Quantization:**
   - Use PTQ or QAT runners for model compression.
   - Quantized models are compatible with deployment on resource-constrained devices.

5. **Evaluation & Reporting:**
   - Use the `evaluation` module for metrics, benchmarking, and reporting.
   - Automated scripts and test cases ensure reliability and reproducibility.

## Extending the Toolkit
- **Add a New Distiller:**
  1. Create a new class in `core/distillers/` inheriting from `BaseDistiller`.
  2. Implement the `compute_loss` and `forward` methods.
  3. Register the new distiller in `MultiStageDistiller` if needed.
- **Add a New Model Type:**
  1. Update `core/models/model_loader.py` to support new model architectures.
  2. Add any custom wrappers or heads in `core/models/`.
- **Add a New Quantization Method:**
  1. Implement in `core/quant/` and expose via the CLI in `app/main.py`.

## Enterprise-Readiness Highlights
- **Type Hints & Docstrings:** All major classes and functions are type-annotated and documented.
- **Logging:** Consistent use of Python logging for traceability and debugging.
- **Testing:** Automated tests for all critical paths (see `test/`).
- **Error Handling:** Robust error handling and informative exceptions throughout.
- **Modularity:** Each component is decoupled and easily swappable.
- **Scalability:** Designed for both research and production/enterprise deployment.

## Example: Minimal Distillation Script
See `examples/minimal_distill.py` for a runnable script that demonstrates the full workflow:
```python
from core.config.config_manager import ConfigManager
from core.models.model_loader import load_models
from core.models.model_wrapper import ModelWrapper
from core.distillers.kd_hinton import KDHintonDistiller
import torch

cfg = ConfigManager(config_path="configs/default.yaml")
device = cfg.device()
teacher, student, tokenizer = load_models(cfg, device=device)
student_wrapper = ModelWrapper(student, device=device, tokenizer=tokenizer)
input_ids = torch.randint(0, 100, (2, 8)).to(device)
labels = torch.randint(0, 2, (2,)).to(device)
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits
student_logits = student_wrapper.forward(input_ids).logits
distiller = KDHintonDistiller(temperature=2.0, alpha=0.5)
loss = distiller.compute_loss(student_logits, teacher_logits, labels)
print(f"Distillation loss: {loss.item():.4f}")
```

## Directory Structure
- `core/` — Main framework modules (models, distillers, quant, config, utils)
- `examples/` — Example scripts for quickstart and advanced usage
- `test/` — Automated tests for all major features
- `configs/` — Experiment and training configuration files
- `docs/` — Documentation and design notes

## For Future: LLM/Agent Integration
- The toolkit is designed to support future integration with LLMs or agents for dynamic, context-aware distillation strategies.
- Hooks and modular APIs are in place for easy extension.

---
For more details, see the codebase and individual module docstrings.