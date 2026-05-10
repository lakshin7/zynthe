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

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
