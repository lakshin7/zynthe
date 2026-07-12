"""Tests for the Phase 2 adapters: GenericHFAdapter + new typed
adapters added (Seq2Seq / Audio / Diffusion).

Each adapter test builds a minimal ``nn.Module`` whose forward signature
or module-name pattern matches the detector's heuristic; the registry
must route the model to the right adapter.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from zynthe.core.adapters.adapter_registry import AdapterRegistry
from zynthe.core.adapters.generic_hf_adapter import GenericHFAdapter


# ----------------------------------------------------------------------------
# GenericHFAdapter
# ----------------------------------------------------------------------------


class _AnyHFMod(nn.Module):
    """Forward uses ``input_ids`` (the textbook HF transformer kwarg)."""

    def forward(self, input_ids, attention_mask=None):
        return torch.tensor([[0.0]])


def test_generic_adapter_claims_model_with_input_ids_kwarg() -> None:
    """GenericHFAdapter matches any HF-shaped model — that is its job."""
    a = GenericHFAdapter()
    assert a.supports_model(_AnyHFMod())


def test_generic_adapter_rejects_module_without_forward() -> None:
    """A bare ``nn.Module`` with no forward should not be claimed."""
    a = GenericHFAdapter()

    class _Bare(nn.Module):
        pass

    assert not a.supports_model(_Bare())


def test_generic_adapter_rejects_module_with_no_known_kwargs() -> None:
    """A module whose forward only accepts an opaque ``x`` should not
    be claimed (no well-known HF kwarg).
    """
    a = GenericHFAdapter()

    class _NoInputs(nn.Module):
        def forward(self, x):
            return torch.zeros(1, 2)

    assert not a.supports_model(_NoInputs())


def test_generic_adapter_detects_pixel_values_kwarg() -> None:
    """A forward that takes only ``pixel_values`` (vision-family) is
    also matched by GenericHFAdapter.
    """
    a = GenericHFAdapter()

    class _PureVision(nn.Module):
        def forward(self, pixel_values):
            return torch.tensor([[0.0]])

    assert a.supports_model(_PureVision())


def test_generic_adapter_detects_input_features_kwarg() -> None:
    """A forward with ``input_features`` (Whisper) matches."""

    a = GenericHFAdapter()

    class _Whisperish(nn.Module):
        def forward(self, input_features, attention_mask=None):
            return torch.tensor([[0.0]])

    assert a.supports_model(_Whisperish())


def test_generic_adapter_prepare_batch_filters_keys() -> None:
    """``prepare_batch`` returns only keys accepted by the model's
    forward signature.
    """
    a = GenericHFAdapter()
    model = _AnyHFMod()
    batch = {
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
        "extra_field_we_dont_have": torch.tensor([1.0]),
    }
    filtered = a.prepare_batch(batch, model)
    assert set(filtered.keys()) == {"input_ids"}


def test_generic_adapter_returns_a_finite_extracted_outputs() -> None:
    """``extract_outputs`` wraps a ModelOutput-shaped object into the
    standard normalized dict.
    """
    a = GenericHFAdapter()

    class _Out:
        logits = torch.tensor([[0.5]])
        hidden_states = (torch.zeros(1, 8),)
        attentions = None

    norm = a.extract_outputs(_Out())
    assert norm["logits"] is not None
    assert norm["hidden_states"] is not None


# ----------------------------------------------------------------------------
# Registry integration
# ----------------------------------------------------------------------------


def test_registry_includes_generic_in_detection_order() -> None:
    """The GenericHFAdapter should be the LAST entry in the detection
    chain so it only fires when no typed adapter claimed the model.
    """
    reg = AdapterRegistry()
    last = reg._detection_order[-1]
    assert last is GenericHFAdapter


def test_registry_resolves_generic_by_name() -> None:
    """``registry.get("generic")`` returns the generic instance."""
    reg = AdapterRegistry()
    adapter = reg.get("generic")
    assert adapter is reg._by_name["generic"]


def test_registry_falls_back_to_generic_for_unknown_forward() -> None:
    """A module whose forward takes an opaque ``x`` doesn't match any
    typed adapter and doesn't match GenericHFAdapter (no known kwarg)
    — registry must return *some* adapter (text by historical fallback
    or generic by the new fallback).
    """

    class _Opaque(nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0]])

    reg = AdapterRegistry()
    adapter = reg.detect(_Opaque())
    # Either text (legacy) or generic is acceptable; we pick generic
    # if its supports_model returned True, else text.
    assert adapter.modality in {"generic", "text"}


# ----------------------------------------------------------------------------
# Adapter extras
# ----------------------------------------------------------------------------


def test_generic_adapter_finds_hookable_layers_by_name() -> None:
    """Generic hook discovery returns up to 4 module names matching
    block patterns.
    """
    a = GenericHFAdapter()

    class _BlockMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])

        def forward(self, x):
            return self.layers[0](x)

    layers = a.get_hookable_layers(_BlockMod())
    # 4 layers (or fewer if filter excluded) — at least one must match.
    assert len(layers) > 0
    for name in layers:
        assert name



# ----------------------------------------------------------------------------
# Seq2SeqAdapter (encoder + decoder, no LM head)
# ----------------------------------------------------------------------------


class _EncoderDecoder(nn.Module):
    """Stub matching the T5 / BART / Marian module naming convention."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.decoder = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    def forward(self, input_ids=None, decoder_input_ids=None, labels=None):
        return torch.tensor([[0.0]])


class _DecoderOnly(nn.Module):
    """GPT-2 style: decoder.* but no encoder.*  Should NOT match seq2seq."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    def forward(self, input_ids=None):
        return torch.tensor([[0.0]])


def test_seq2seq_adapter_matches_encoder_decoder() -> None:
    from zynthe.core.adapters.seq2seq_adapter import Seq2SeqAdapter

    a = Seq2SeqAdapter()
    assert a.supports_model(_EncoderDecoder())


def test_seq2seq_adapter_rejects_decoder_only() -> None:
    from zynthe.core.adapters.seq2seq_adapter import Seq2SeqAdapter

    a = Seq2SeqAdapter()
    assert not a.supports_model(_DecoderOnly())


def test_registry_routes_t5_to_seq2seq() -> None:
    reg = AdapterRegistry()
    adapter = reg.detect(_EncoderDecoder())
    assert adapter.modality == "seq2seq"


# ----------------------------------------------------------------------------
# AudioAdapter
# ----------------------------------------------------------------------------


class _Whisperish(nn.Module):
    def forward(self, input_features, decoder_input_ids=None):
        return torch.tensor([[0.0]])


class _Wav2Vecish(nn.Module):
    def forward(self, input_values, attention_mask=None):
        return torch.tensor([[0.0]])


class _MelNotWav(nn.Module):
    """Has input_features but no audio encoder structure."""

    def forward(self, input_features, labels=None):
        return torch.tensor([[0.0]])


def test_audio_adapter_matches_whisper_input_features() -> None:
    from zynthe.core.adapters.audio_adapter import AudioAdapter

    a = AudioAdapter()
    assert a.supports_model(_Whisperish())


def test_audio_adapter_matches_wav2vec_input_values() -> None:
    from zynthe.core.adapters.audio_adapter import AudioAdapter

    a = AudioAdapter()
    assert a.supports_model(_Wav2Vecish())


def test_registry_routes_whisper_to_audio() -> None:
    reg = AdapterRegistry()
    adapter = reg.detect(_Whisperish())
    # Order: VLM, Diffusion, Audio, Multimodal, Seq2Seq, Vision, Code, Text, Generic.
    # Audio comes after Diffusion but before others. The Whisperish stub
    # has input_features — Audio claims first.
    assert adapter.modality == "audio"


# ----------------------------------------------------------------------------
# DiffusionAdapter
# ----------------------------------------------------------------------------


class _UNetish(nn.Module):
    """SD-style UNet with down_blocks / mid_block / up_blocks."""

    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.up_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.mid_block = nn.Linear(4, 4)

    def forward(self, sample, timestep, encoder_hidden_states=None):
        return torch.tensor([[0.0]])


class _NoDownNoUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.something_else = nn.Linear(4, 4)

    def forward(self, x):
        return x


def test_diffusion_adapter_matches_unet_block_structure() -> None:
    from zynthe.core.adapters.diffusion_adapter import DiffusionAdapter

    a = DiffusionAdapter()
    assert a.supports_model(_UNetish())


def test_diffusion_adapter_rejects_unrelated_module() -> None:
    from zynthe.core.adapters.diffusion_adapter import DiffusionAdapter

    a = DiffusionAdapter()
    assert not a.supports_model(_NoDownNoUp())


def test_registry_routes_unet_to_diffusion() -> None:
    reg = AdapterRegistry()
    adapter = reg.detect(_UNetish())
    assert adapter.modality == "diffusion"


# ----------------------------------------------------------------------------
# generic + adapters ordering stays consistent
# ----------------------------------------------------------------------------


def test_registry_detection_order_includes_all_typed_adapters() -> None:
    reg = AdapterRegistry()
    modalities = [cls.modality for cls in reg._detection_order]
    assert "seq2seq" in modalities
    assert "audio" in modalities
    assert "diffusion" in modalities
    assert "generic" in modalities
    # Generic must come last.
    assert modalities[-1] == "generic"
