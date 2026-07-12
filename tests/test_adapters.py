"""Behavior tests for :class:`AdapterRegistry`.

Each adapter's :meth:`supports_model` is a heuristic on the model's
``forward`` signature and/or module-name patterns. These tests build
fake ``nn.Module`` stubs that match each heuristic and verify the
registry routes them to the right adapter.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from zynthe.core.adapters.adapter_registry import AdapterRegistry


# ----------------------------------------------------------------------------
# Stubs with shaped forward signatures matching each adapter.
# ----------------------------------------------------------------------------


class _TextOnly(nn.Module):
    """Model whose forward takes ``input_ids`` and nothing vision-shaped."""

    def forward(self, input_ids, attention_mask=None, labels=None):
        return type("O", (), {"logits": torch.zeros(1, 2)})()


class _VisionOnly(nn.Module):
    """Model whose forward takes ``pixel_values`` and not ``input_ids``."""

    def forward(self, pixel_values):
        return type("O", (), {"logits": torch.zeros(1, 2)})()


class _VisionTextPair(nn.Module):
    """CLIP-style: vision tower + text encoder, no lm_head."""

    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(3, 3)
        self.text_model = nn.Linear(3, 3)

    def forward(self, input_ids=None, pixel_values=None):
        return type("O", (), {"logits": torch.zeros(1, 2)})()


class _VLM(nn.Module):
    """LLaVA-style: vision_tower + language_model + projector."""

    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Linear(3, 3)
        self.language_model = nn.Linear(3, 3)
        self.projector = nn.Linear(3, 3)

    def forward(self, input_ids=None, pixel_values=None):
        return type("O", (), {"logits": torch.zeros(1, 2)})()


class _CodeLM(nn.Module):
    """Code-gen style: input_ids + a code-only module suffix."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(3, 3)
        self.lm_head = nn.Linear(3, 4)

    def forward(self, input_ids, attention_mask=None):
        return type("O", (), {"logits": torch.zeros(1, 2)})()


class _Unknown(nn.Module):
    """Module the registry has never seen."""

    def forward(self, x):
        return type("O", (), {"logits": torch.zeros(1, 2)})()


# ----------------------------------------------------------------------------
# Sanity: stubs have the expected forward signatures.
# ----------------------------------------------------------------------------


def test_text_only_has_input_ids_signature() -> None:
    import inspect

    sig = inspect.signature(_TextOnly().forward)
    assert "input_ids" in sig.parameters


def test_vision_only_has_pixel_values_signature() -> None:
    import inspect

    sig = inspect.signature(_VisionOnly().forward)
    assert "pixel_values" in sig.parameters
    assert "input_ids" not in sig.parameters


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------


def test_registry_detects_text_only() -> None:
    reg = AdapterRegistry()
    adapter = reg.detect(_TextOnly())
    assert adapter.modality == "text"


def test_registry_detects_vision_only() -> None:
    reg = AdapterRegistry()
    adapter = reg.detect(_VisionOnly())
    assert adapter.modality == "vision"


def test_registry_detects_multimodal_clip_style() -> None:
    reg = AdapterRegistry()
    adapter = reg.detect(_VisionTextPair())
    # Has vision_model + text_model → multimodal (clip-style), not VLM
    # (no language_model module name).
    assert adapter.modality == "multimodal"


def test_registry_detects_vlm_with_lm_head() -> None:
    reg = AdapterRegistry()
    adapter = reg.detect(_VLM())
    # VLM detection requires vision_tower AND (language_model OR projector).
    assert adapter.modality == "vlm"


def test_registry_falls_back_to_generic() -> None:
    """Phase 1 used to fall back to ``text`` for unrecognised modules;
    Phase 2 promotes :class:`GenericHFAdapter` to be the universal
    fallback when no typed adapter matches (the registry tries
    Generic last so it only claims modules with at least one well-known
    HF kwarg).  For an opaque ``forward(x)`` module neither matched
    and the registry now resolves to ``generic`` by ``AdapterRegistry``
    design.
    """
    reg = AdapterRegistry()
    adapter = reg.detect(_Unknown())
    assert adapter.modality == "generic"


# ----------------------------------------------------------------------------
# Explicit lookup via get()
# ----------------------------------------------------------------------------


def test_registry_get_by_name_returns_correct_modality() -> None:
    reg = AdapterRegistry()
    assert reg.get("text").modality == "text"
    assert reg.get("vision").modality == "vision"
    assert reg.get("vlm").modality == "vlm"
    assert reg.get("multimodal").modality == "multimodal"
    assert reg.get("code").modality == "code"


def test_registry_get_unknown_raises_keyerror() -> None:
    reg = AdapterRegistry()
    with pytest.raises(KeyError, match="Unknown modality"):
        reg.get("does-not-exist")
