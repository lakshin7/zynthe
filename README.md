# Zynthé 🧠 ⚡

> **Zynthé** — *From "Synthesize". Knowledge distillation is the process of synthesizing knowledge from a massive, complex model down into a smaller, more focused one.*

**Universal Knowledge Distillation Toolkit**  
A Python library for compressing large language models into smaller, deployable student models — across text, code, vision, and multimodal architectures.

[![PyPI version](https://badge.fury.io/py/zynthe.svg)](https://badge.fury.io/py/zynthe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Large transformer models (BERT-110M, GPT-2-124M, ViT-86M) deliver state-of-the-art accuracy but are often too expensive for edge deployment, mobile inference, and cost-sensitive production. **Zynthé** solves this by training a smaller "student" model to replicate the behavior of a larger "teacher" model using a unified, multi-stage distillation pipeline.

---

## 🌟 Key Features

- **Modality Agnostic**: Works out-of-the-box with Text, Code, Vision, VLM, and Multimodal models via an auto-detecting `AdapterRegistry`.
- **Multi-Stage Distillation**: Supports sequential distillation stages:
  - **Logit Distillation** (KD Hinton)
  - **Feature Alignment** (Hidden representations)
  - **Attention Transfer** (Attention patterns)
  - **Structural Similarity** (Cross-layer relationships)
- **Zero-Config Presets**: 7 battle-tested distillation plans (`quick_start`, `balanced`, `compression_max`, etc.).
- **Preflight Analysis**: Validates configurations, hardware compatibility, and estimates memory before training starts.
- **Post-Training Quantization (PTQ)**: Further compress your distilled models to INT8.
- **Rich Visualization**: Auto-generates training curves, distillation gap plots, and metric grids.

---

## 📦 Installation

Install Zynthé via pip:

```bash
# Core installation
pip install zynthe

# With evaluation and plotting dependencies
pip install zynthe[eval]

# For computer vision support
pip install zynthe[vision]
```

---

## 🚀 Quickstart

Zynthé makes complex multi-stage distillation as simple as 5 lines of code:

<p align="center">
  <img src="docs/assets/typing_demo.svg" alt="Zynthé Typing Demo" width="100%">
</p>

---

## 📊 Example Results

**BERT-base (110M) → DistilBERT (66M) on SST-2 (Single NVIDIA T4)**

| Metric | Teacher | Student | Distilled Student |
|--------|---------|---------|-------------------|
| **Accuracy** | 92.1% | ~81.5% (scratch) | **86.8%** |
| **Model Size** | 418 MB | 255 MB | **255 MB** |
| **Quantized (PTQ)** | - | - | **64 MB** (6.5x smaller) |

---

## 🛠️ Architecture

Zynthé uses a modular, plugin-based architecture:

- `DistillationToolkit`: The primary user-facing facade.
- `AdapterRegistry`: Automatically normalizes I/O across HuggingFace architectures.
- `PipelineBuilder`: Constructs deterministic, multi-stage training plans.
- `PreflightAnalyzer`: Catches OOMs and shape mismatches before training.
- `Evaluator` & `ModelComparator`: Side-by-side student vs. teacher analysis.

---

## 📚 Documentation

- [Quickstart Guide](docs/quickstart.md) — Get up and running in minutes.
- [API Reference](docs/api_reference.md) — Comprehensive guide to the 44 public API symbols.
- [Adapters System](docs/adapters.md) — How to add custom model architectures.
- [Kaggle Examples](examples/kaggle_t4_test.py) — End-to-end notebooks for T4/P100 environments.

---

## 📄 License

This project is licensed under the **MIT License**. 

You are free to use, modify, distribute, and use this software for commercial purposes. See the [LICENSE](LICENSE) file for the full text.

> **Note:** Models distilled using Zynthé inherit the license of their respective teacher models and datasets. Please ensure you comply with the licensing terms of the models you are distilling.