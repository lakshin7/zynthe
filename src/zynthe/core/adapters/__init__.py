"""
Model Adapters — Multi-Platform Architecture Normalization
============================================================

Provides a pluggable adapter layer that normalizes model I/O
across diverse architectures (text encoders, code models, vision models,
multimodal encoders, VLMs) so that a single distillation pipeline
can operate on any combination of teacher and student.

Supported modalities:

- **Text**: BERT, RoBERTa, GPT-2, LLaMA, Mistral, T5, etc.
- **Code**: CodeBERT, CodeLlama, StarCoder, DeepSeek-Coder, etc.
- **Vision**: ViT, BEiT, Swin, DeiT, ConvNeXt, etc.
- **Multimodal**: CLIP, SigLIP, BLIP, FLAVA, etc.
- **VLM**: LLaVA, InternVL, Qwen-VL, Phi-Vision, etc.

Usage::

    from zynthe.core.adapters import AdapterRegistry

    registry = AdapterRegistry()
    adapter = registry.detect(model)           # auto-detect
    adapter = registry.get("text")             # explicit

    batch  = adapter.prepare_batch(raw_batch, model)
    output = adapter.extract_outputs(model(**batch))
    layers = adapter.get_hookable_layers(model)
"""

from __future__ import annotations

from .adapter_registry import AdapterRegistry
from .base_adapter import ModelAdapter
from .code_adapter import CodeModelAdapter
from .multimodal_adapter import MultimodalModelAdapter
from .text_adapter import TextModelAdapter
from .vision_adapter import VisionModelAdapter
from .vlm_adapter import VLMModelAdapter

__all__ = [
    "ModelAdapter",
    "TextModelAdapter",
    "CodeModelAdapter",
    "VisionModelAdapter",
    "MultimodalModelAdapter",
    "VLMModelAdapter",
    "AdapterRegistry",
]
