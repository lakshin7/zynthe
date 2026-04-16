"""
Model Adapters — Multi-Platform Architecture Normalization
============================================================

Provides a pluggable adapter layer that normalizes model I/O
across diverse architectures (text encoders, vision models,
multimodal encoders, VLMs) so that a single distillation pipeline
can operate on any combination of teacher and student.

Usage::

    from core.adapters import AdapterRegistry

    registry = AdapterRegistry()
    adapter = registry.detect(model)           # auto-detect
    adapter = registry.get("text")(model)       # explicit

    batch  = adapter.prepare_batch(raw_batch, model)
    output = adapter.extract_outputs(model(**batch))
    layers = adapter.get_hookable_layers(model)
"""

from .base_adapter import ModelAdapter
from .text_adapter import TextModelAdapter
from .vision_adapter import VisionModelAdapter
from .multimodal_adapter import MultimodalModelAdapter
from .vlm_adapter import VLMModelAdapter
from .adapter_registry import AdapterRegistry

__all__ = [
    "ModelAdapter",
    "TextModelAdapter",
    "VisionModelAdapter",
    "MultimodalModelAdapter",
    "VLMModelAdapter",
    "AdapterRegistry",
]
