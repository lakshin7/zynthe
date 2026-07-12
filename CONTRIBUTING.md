# Contributing to Zynthé

Thank you for your interest in contributing to Zynthé! This guide will help you set up a development environment and submit high-quality contributions.

---

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/lakshin7/zynthe.git
cd zynthe
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install in development mode

```bash
pip install -e ".[dev,eval]"
```

This installs the library in editable mode along with all development tools (pytest, black, isort, flake8, mypy).

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_distiller_api.py -v

# Run with coverage
pytest tests/ --cov=src/zynthe --cov-report=term-missing
```

---

## Code Style

We use the following tools to maintain consistent code quality:

### Formatting

```bash
# Format code with black
black src/zynthe tests/

# Sort imports
isort src/zynthe tests/
```

### Linting

```bash
# Run flake8
flake8 src/zynthe tests/

# Run type checking
mypy src/zynthe/core/distillers/toolkit.py --ignore-missing-imports
```

### Conventions

- **Type annotations**: Use `from __future__ import annotations` at the top of every module. Annotate all public function signatures.
- **Docstrings**: Use Google-style docstrings for all public classes, methods, and functions.
- **Logging**: Use `logging.getLogger(__name__)` instead of `print()`. The logger infrastructure is in `src/zynthe/core/utils/logger.py`.
- **Naming**: The PyPI package name is `zynthe` (lowercase). The branded display name is **Zynthé** (with accent). Use `zynthe` for imports and code, `Zynthé` for documentation.
- **Errors**: Use the `ZyntheError` hierarchy in `zynthe.core.utils.exceptions` (`ConfigError`, `DistillationError`, `AdapterError`, `PreflightError`, `QuantizationError`, `RegistryError`) instead of bare `ValueError` / `RuntimeError` for library-level failures. This lets downstream code `except ZyntheError` cleanly.
- **Determinism**: For tests touching `torch`, prefer pytest fixtures that seed `torch.manual_seed` + `torch.cuda.manual_seed_all` and (when toggling autotune off) ensure the global `cudnn.deterministic` flag is whatever the rest of the suite expects.

---

## Project Structure

```
src/zynthe/
├── __init__.py              # Public API exports (44+ symbols)
├── core/
│   ├── adapters/            # Modality-specific model adapters
│   ├── config/              # ConfigManager (YAML/dict validation)
│   ├── distillers/          # KD methods (Hinton, Feature, Attention, Similarity)
│   │   ├── toolkit.py       # DistillationToolkit + Distiller (main API)
│   │   ├── presets.py        # Battle-tested distillation configs
│   │   └── multi_stage_distiller.py  # Orchestrator for multi-stage pipelines
│   ├── models/              # Model loading, saving, wrapping
│   ├── pipelines/           # Pipeline builder and registry
│   ├── preflight/           # Hardware/config validation before training
│   ├── quant/               # PTQ and QAT quantization
│   └── utils/               # Device management, logging, metrics
├── data/                    # DataLoader factories and augmentations
├── evaluation/              # Evaluator, ModelComparator, visualizer
└── training/                # Trainer, optimizer, scheduler
```

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the full test suite** and ensure all tests pass.
4. **Format your code** with `black` and `isort`.
5. **Open a PR** with a clear description of the changes.

### PR Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Code is formatted (`black --check src/ tests/`)
- [ ] Type annotations added for new public functions
- [ ] Docstrings added for new public classes/methods
- [ ] No new `print()` statements (use `logging` instead)

---

## Authoring a Distiller

A "distiller" is any subclass of `zynthe.core.distillers.base_distiller.BaseDistiller`
that overrides `compute_loss(...)` and (optionally) `_register_hooks()`.

### Template

```python
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from zynthe.core.distillers.base_distiller import BaseDistiller


class MyDistiller(BaseDistiller):
    """One-line description.

    Reference: Paper / equation / section.

    Config schema::

        my_distiller:
          alpha: 0.5
    """

    def __init__(self, teacher, student, config=None, device=None, **kwargs):
        super().__init__(teacher, student, config=config, device=device, **kwargs)
        cfg = (config or {}).get("my_distiller", {})
        self.alpha = float(cfg.get("alpha", 0.5))

    def compute_loss(
        self,
        student_outputs: Any,
        teacher_outputs: Any,
        targets: Optional[torch.Tensor] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_a = F.mse_loss(student_outputs.logits, teacher_outputs.logits.detach())
        loss_b = F.cross_entropy(student_outputs.logits, targets) if targets is not None else 0.0
        total = self.alpha * loss_a + (1 - self.alpha) * loss_b
        return total, {"a": loss_a.item(), "b": float(loss_b) if torch.is_tensor(loss_b) else 0.0}
```

### Rules

1. **Return `(total_loss, loss_dict)`** from `compute_loss`; the loss dict feeds the
   logging/visualizer layer.
2. **Detach the teacher.** Any tensor you read from `teacher_outputs` /
   `teacher_features` should be detached (`teacher_outputs.logits.detach()`) so
   gradients don't flow through the frozen teacher.
3. **Use `self._extract_logits_tensor(outputs)`** to pull float32 logits that are
   numerically safe (it upcasts fp16/bf16 and clamps `inf`/`nan`).
4. **Use `self._move_to_device(batch)`** if you bypass `training_step`.
5. **Register a class with the distiller registry** by adding it to
   `DistillerRegistry` in `zynthe/core/distillers/multi_stage_distiller.py`.
   Use the `requires=[...]` metadata (`"logits"`, `"features"`, `"attentions"`,
   `"gram"`) so pipelines can validate that the chosen adapter exposes what
   you need before the first forward pass.
6. **Write a behavior test** in `tests/test_distillers.py` (or a dedicated
   `tests/test_<your_distiller>.py`). Two minimum cases:
   - **Zero-loss sentinel**: a teacher/student pair where the student exactly
     copies the teacher (same modules, same weights) → loss should be `<1e-6`.
   - **Reference value**: for closed-form losses (Hinton, MSE on Gram), compare
     `compute_loss` against a hand-evaluated reference within `1e-5`.
7. **Never catch `ZyntheError`** inside the distiller — let it propagate so the
   runtime can surface actionable errors.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
