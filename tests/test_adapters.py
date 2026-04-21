"""Tests for Phase 2 multi-platform model adapters.

Verifies:
1. Each adapter correctly identifies its target model type.
2. AdapterRegistry auto-detects the right adapter.
3. Adapters correctly prepare batches and extract outputs.
4. Dimension alignment works for mismatched teacher/student sizes.

All tests use mock models — NO real HuggingFace downloads needed.
Safe on low-end laptops and Colab CPU.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace


# ── Mock Models ───────────────────────────────────────────────────────

class MockTextModel(nn.Module):
    """Simulates a BERT-like text model."""

    def __init__(self, hidden=32, labels=2):
        super().__init__()
        self.bert = nn.Module()
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.classifier = nn.Linear(hidden, labels)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        logits = torch.randn(batch_size, 2)
        hidden = tuple(torch.randn(batch_size, 8, 32) for _ in range(3)) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden, attentions=None, loss=None)


class MockVisionModel(nn.Module):
    """Simulates a ViT-like vision model."""

    def __init__(self, hidden=64, classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden, 16, 16)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(4)])
        self.head = nn.Linear(hidden, classes)

    def forward(self, pixel_values, labels=None, output_hidden_states=False):
        batch_size = pixel_values.shape[0]
        logits = torch.randn(batch_size, 10)
        return SimpleNamespace(logits=logits, hidden_states=None, attentions=None, loss=None)


class MockCLIPModel(nn.Module):
    """Simulates a CLIP-like dual-encoder model."""

    def __init__(self, text_dim=32, vision_dim=64, embed_dim=16):
        super().__init__()
        self.text_model = nn.Module()
        self.text_model.encoder = nn.Module()
        self.text_model.encoder.layers = nn.ModuleList([nn.Linear(text_dim, text_dim) for _ in range(2)])
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([nn.Linear(vision_dim, vision_dim) for _ in range(3)])
        self.text_projection = nn.Linear(text_dim, embed_dim)
        self.visual_projection = nn.Linear(vision_dim, embed_dim)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        batch = (input_ids.shape[0] if input_ids is not None
                 else pixel_values.shape[0] if pixel_values is not None
                 else 1)
        return SimpleNamespace(
            logits_per_image=torch.randn(batch, batch),
            logits_per_text=torch.randn(batch, batch),
            text_embeds=torch.randn(batch, 16),
            image_embeds=torch.randn(batch, 16),
            logits=None, hidden_states=None, attentions=None, loss=None,
        )


class MockVLMModel(nn.Module):
    """Simulates a LLaVA-like VLM."""

    def __init__(self, vision_dim=64, lm_dim=128, vocab_size=100):
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.encoder = nn.Module()
        self.vision_tower.encoder.layers = nn.ModuleList([nn.Linear(vision_dim, vision_dim) for _ in range(2)])
        self.multi_modal_projector = nn.Linear(vision_dim, lm_dim)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([nn.Linear(lm_dim, lm_dim) for _ in range(4)])
        self.language_model.lm_head = nn.Linear(lm_dim, vocab_size)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, labels=None, **kwargs):
        batch = 2
        return SimpleNamespace(
            logits=torch.randn(batch, 8, 100),
            hidden_states=None, attentions=None, loss=None,
        )


# ── Test: Adapter supports_model ──────────────────────────────────────

class TestAdapterSupportsModel:
    def test_text_adapter_detects_text_model(self):
        from core.adapters.text_adapter import TextModelAdapter
        adapter = TextModelAdapter()
        assert adapter.supports_model(MockTextModel()) is True

    def test_text_adapter_rejects_vision_model(self):
        from core.adapters.text_adapter import TextModelAdapter
        adapter = TextModelAdapter()
        assert adapter.supports_model(MockVisionModel()) is False

    def test_vision_adapter_detects_vision_model(self):
        from core.adapters.vision_adapter import VisionModelAdapter
        adapter = VisionModelAdapter()
        assert adapter.supports_model(MockVisionModel()) is True

    def test_multimodal_adapter_detects_clip(self):
        from core.adapters.multimodal_adapter import MultimodalModelAdapter
        adapter = MultimodalModelAdapter()
        assert adapter.supports_model(MockCLIPModel()) is True

    def test_vlm_adapter_detects_vlm(self):
        from core.adapters.vlm_adapter import VLMModelAdapter
        adapter = VLMModelAdapter()
        assert adapter.supports_model(MockVLMModel()) is True

    def test_vlm_adapter_rejects_text_model(self):
        from core.adapters.vlm_adapter import VLMModelAdapter
        adapter = VLMModelAdapter()
        assert adapter.supports_model(MockTextModel()) is False


# ── Test: AdapterRegistry auto-detection ──────────────────────────────

class TestAdapterRegistry:
    def test_detect_text_model(self):
        from core.adapters import AdapterRegistry
        registry = AdapterRegistry()
        adapter = registry.detect(MockTextModel())
        assert adapter.modality == "text"

    def test_detect_vision_model(self):
        from core.adapters import AdapterRegistry
        registry = AdapterRegistry()
        adapter = registry.detect(MockVisionModel())
        assert adapter.modality == "vision"

    def test_detect_clip_model(self):
        from core.adapters import AdapterRegistry
        registry = AdapterRegistry()
        adapter = registry.detect(MockCLIPModel())
        assert adapter.modality == "multimodal"

    def test_detect_vlm_model(self):
        from core.adapters import AdapterRegistry
        registry = AdapterRegistry()
        adapter = registry.detect(MockVLMModel())
        assert adapter.modality == "vlm"

    def test_get_by_name(self):
        from core.adapters import AdapterRegistry
        registry = AdapterRegistry()
        adapter = registry.get("text")
        assert adapter.modality == "text"

    def test_get_unknown_raises(self):
        from core.adapters import AdapterRegistry
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.get("unknown_modality")

    def test_list_available(self):
        from core.adapters import AdapterRegistry
        registry = AdapterRegistry()
        available = registry.list_available()
        assert "text" in available
        assert "vision" in available
        assert "multimodal" in available
        assert "vlm" in available


# ── Test: Batch preparation ───────────────────────────────────────────

class TestBatchPreparation:
    def test_text_adapter_filters_batch(self):
        from core.adapters.text_adapter import TextModelAdapter
        adapter = TextModelAdapter()
        model = MockTextModel()

        batch = {
            "input_ids": torch.randint(0, 50, (2, 8)),
            "attention_mask": torch.ones(2, 8),
            "labels": torch.randint(0, 2, (2,)),
            "pixel_values": torch.randn(2, 3, 224, 224),  # should be removed
        }
        prepared = adapter.prepare_batch(batch, model)
        assert "input_ids" in prepared
        assert "attention_mask" in prepared
        assert "pixel_values" not in prepared  # text model doesn't accept this

    def test_vision_adapter_keeps_pixel_values(self):
        from core.adapters.vision_adapter import VisionModelAdapter
        adapter = VisionModelAdapter()
        model = MockVisionModel()

        batch = {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "labels": torch.randint(0, 10, (2,)),
            "input_ids": torch.randint(0, 50, (2, 8)),  # should be removed
        }
        prepared = adapter.prepare_batch(batch, model)
        assert "pixel_values" in prepared
        assert "labels" in prepared
        assert "input_ids" not in prepared


# ── Test: Output extraction ───────────────────────────────────────────

class TestOutputExtraction:
    def test_text_adapter_extracts_logits(self):
        from core.adapters.text_adapter import TextModelAdapter
        adapter = TextModelAdapter()
        raw = SimpleNamespace(logits=torch.randn(2, 5), loss=None, hidden_states=None, attentions=None)
        result = adapter.extract_outputs(raw)
        assert result["logits"] is not None
        assert result["logits"].shape == (2, 5)

    def test_multimodal_adapter_extracts_embeds(self):
        from core.adapters.multimodal_adapter import MultimodalModelAdapter
        adapter = MultimodalModelAdapter()
        raw = SimpleNamespace(
            logits_per_image=torch.randn(2, 2),
            logits_per_text=torch.randn(2, 2),
            text_embeds=torch.randn(2, 16),
            image_embeds=torch.randn(2, 16),
            logits=None, hidden_states=None, attentions=None, loss=None,
        )
        result = adapter.extract_outputs(raw)
        assert result["image_embeds"] is not None
        assert result["text_embeds"] is not None
        assert result["logits"] is not None  # falls back to logits_per_image


# ── Test: Hookable layer discovery ────────────────────────────────────

class TestHookableLayers:
    def test_text_adapter_finds_bert_layers(self):
        from core.adapters.text_adapter import TextModelAdapter
        adapter = TextModelAdapter()
        model = MockTextModel()
        layers = adapter.get_hookable_layers(model)
        matching = [lyr for lyr in layers if "encoder.layer" in lyr]
        assert len(matching) > 0

    def test_vision_adapter_finds_encoder_layers(self):
        from core.adapters.vision_adapter import VisionModelAdapter
        adapter = VisionModelAdapter()
        model = MockVisionModel()
        layers = adapter.get_hookable_layers(model)
        matching = [lyr for lyr in layers if "encoder.layer" in lyr]
        assert len(matching) > 0

    def test_vlm_adapter_finds_all_components(self):
        from core.adapters.vlm_adapter import VLMModelAdapter
        adapter = VLMModelAdapter()
        model = MockVLMModel()
        layers = adapter.get_hookable_layers(model)
        assert any("vision_tower" in lyr for lyr in layers)
        assert any("language_model" in lyr for lyr in layers)


# ── Test: Dimension alignment ─────────────────────────────────────────

class TestDimensionAlignment:
    def test_text_adapter_aligns_hidden_dims(self):
        from core.adapters.text_adapter import TextModelAdapter
        adapter = TextModelAdapter()

        teacher_feats = {"layer_6": torch.randn(2, 8, 768)}
        student_feats = {"layer_6": torch.randn(2, 8, 256)}

        aligned_t, aligned_s = adapter.align_dimensions(teacher_feats, student_feats)
        assert aligned_t["layer_6"].shape[-1] == 768
        assert aligned_s["layer_6"].shape[-1] == 768  # projected to teacher dim

    def test_vision_adapter_aligns_spatial_dims(self):
        from core.adapters.vision_adapter import VisionModelAdapter
        adapter = VisionModelAdapter()

        teacher_feats = {"stage3": torch.randn(2, 512, 14, 14)}
        student_feats = {"stage3": torch.randn(2, 256, 7, 7)}

        aligned_t, aligned_s = adapter.align_dimensions(teacher_feats, student_feats)
        assert aligned_t["stage3"].shape == (2, 512, 14, 14)
        assert aligned_s["stage3"].shape == (2, 512, 14, 14)
