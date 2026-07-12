"""
Adapter Registry — Auto-Detection and Instantiation
=====================================================

Provides :class:`AdapterRegistry` which:

1. Inspects a model's architecture (module names, forward signature, config).
2. Returns the most appropriate :class:`ModelAdapter` subclass.
3. Supports explicit overrides via ``registry.get("text")``.

Detection priority:
    VLM → Multimodal → Vision → Code → Text → GenericHF (last-resort fallback).
"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

from .audio_adapter import AudioAdapter
from .base_adapter import ModelAdapter
from .code_adapter import CodeModelAdapter
from .diffusion_adapter import DiffusionAdapter
from .generic_hf_adapter import GenericHFAdapter
from .multimodal_adapter import MultimodalModelAdapter
from .seq2seq_adapter import Seq2SeqAdapter
from .text_adapter import TextModelAdapter
from .vision_adapter import VisionModelAdapter
from .vlm_adapter import VLMModelAdapter


class AdapterRegistry:
    """Central registry for model adapters.

    Usage::

        registry = AdapterRegistry()

        # Auto-detect
        adapter = registry.detect(model)

        # Explicit
        adapter = registry.get("vision")
    """

    def __init__(self) -> None:
        # Ordered from most specific to least specific.
        # detect() will return the first adapter that claims support.
        # GenericHFAdapter is always last so it acts as the universal
        # fallback for any HF model the typed adapters couldn't claim.
        self._detection_order: list = [
            VLMModelAdapter,
            DiffusionAdapter,
            AudioAdapter,
            MultimodalModelAdapter,
            Seq2SeqAdapter,
            VisionModelAdapter,
            CodeModelAdapter,
            TextModelAdapter,
            GenericHFAdapter,
        ]

        # Name → singleton instance mapping.
        self._by_name: Dict[str, ModelAdapter] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Pre-register built-in adapters by modality name."""
        self._by_name["text"] = TextModelAdapter()
        self._by_name["code"] = CodeModelAdapter()
        self._by_name["vision"] = VisionModelAdapter()
        self._by_name["multimodal"] = MultimodalModelAdapter()
        self._by_name["seq2seq"] = Seq2SeqAdapter()
        self._by_name["vlm"] = VLMModelAdapter()
        self._by_name["diffusion"] = DiffusionAdapter()
        self._by_name["audio"] = AudioAdapter()
        self._by_name["generic"] = GenericHFAdapter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, model: nn.Module) -> ModelAdapter:
        """Inspect *model* and return the best-matching adapter.

        Walks the detection order (VLM → Multimodal → Vision → Code → Text
        → GenericHF) and returns the first adapter whose
        ``supports_model()`` returns ``True``.

        The :class:`GenericHFAdapter` matches any HF model whose forward
        signature includes at least one well-known input kwarg
        (``input_ids`` / ``pixel_values`` / ``input_features`` /
        ``input_values``), so any architecture not claimed by the more
        specific adapters lands here.  No more silent fallback to the
        text adapter.

        Args:
            model: A PyTorch model to inspect.

        Returns:
            An adapter instance.
        """
        for adapter_cls in self._detection_order:
            adapter = self._by_name.get(adapter_cls.modality)
            if adapter is None:
                adapter = adapter_cls()
                self._by_name[adapter_cls.modality] = adapter

            if adapter.supports_model(model):
                return adapter

        # Final fallback — only if forward() is missing entirely or
        # has no known input keys.
        return self._by_name.get("generic", self._by_name["text"])


    def get(self, modality: str) -> ModelAdapter:
        """Return an adapter by explicit modality name.

        Args:
            modality: One of ``"text"``, ``"code"``, ``"vision"``,
                ``"multimodal"``, ``"vlm"``, ``"generic"``.

        Returns:
            Adapter instance.

        Raises:
            KeyError: If *modality* is not registered.
        """
        modality = modality.lower().strip()
        if modality not in self._by_name:
            available = sorted(self._by_name.keys())
            raise KeyError(f"Unknown modality '{modality}'. " f"Available: {available}")
        return self._by_name[modality]

    def register(
        self,
        modality: str,
        adapter_cls: Type[ModelAdapter],
        *,
        priority: int = -1,
    ) -> None:
        """Register a custom adapter.

        Args:
            modality: Modality name (e.g. ``"video"``).
            adapter_cls: The adapter class (must extend ModelAdapter).
            priority: Position in detection order.  ``-1`` = end
                (lowest priority).  ``0`` = front (highest priority).
        """
        if not issubclass(adapter_cls, ModelAdapter):
            raise TypeError(f"{adapter_cls.__name__} must extend ModelAdapter")

        adapter = adapter_cls()
        adapter.modality = modality
        self._by_name[modality] = adapter

        if priority < 0:
            self._detection_order.append(adapter_cls)
        else:
            self._detection_order.insert(priority, adapter_cls)

    def list_available(self) -> list:
        """Return sorted list of registered modality names."""
        return sorted(self._by_name.keys())

    def __repr__(self) -> str:
        return f"AdapterRegistry(modalities={self.list_available()})"
