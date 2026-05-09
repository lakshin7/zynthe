# Modality Adapters

Zynthé's adapter system normalizes model I/O across architectures so a single distillation pipeline works with any teacher-student combination.

## Detection Priority

When auto-detecting via `AdapterRegistry.detect(model)`, adapters are tried in this order:

1. **VLM** — Generative vision-language models (LLaVA, InternVL, Qwen-VL)
2. **Multimodal** — Dual-encoder models (CLIP, SigLIP, BLIP)
3. **Vision** — Pure vision models (ViT, Swin, ConvNeXt)
4. **Code** — Code-specific models (CodeLlama, StarCoder, DeepSeek-Coder)
5. **Text** — Text-only models (BERT, GPT-2, LLaMA) — default fallback

## Adapter Interface

Every adapter implements 4 methods:

| Method | Purpose |
|--------|---------|
| `prepare_batch(batch, model)` | Filter batch keys to match model's forward signature |
| `extract_outputs(raw_output)` | Normalize model output into `{logits, hidden_states, attentions, loss}` |
| `get_hookable_layers(model)` | Return module names suitable for forward hooks |
| `align_dimensions(teacher_feats, student_feats)` | Project features so dimensions match |

Plus `supports_model(model)` for auto-detection.

## Detection Heuristics

| Adapter | Detection Logic |
|---------|----------------|
| **VLM** | Has vision tower/encoder AND language model/lm_head |
| **Multimodal** | Has text_model/text_encoder AND vision_model/visual_encoder, but NO lm_head |
| **Vision** | Has `pixel_values` in forward signature, but NOT `input_ids` |
| **Code** | Model config `model_type` or `_name_or_path` matches code model patterns |
| **Text** | Has `input_ids` in forward signature (default fallback) |

## Architecture Coverage

### Text Adapter
- **Encoders**: BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa
- **Decoders**: GPT-2, LLaMA, Mistral, Phi, Gemma
- **Encoder-Decoders**: T5, BART, mBART

### Code Adapter
- **Encoders**: CodeBERT, GraphCodeBERT, UniXcoder
- **Decoders**: CodeLlama, StarCoder, DeepSeek-Coder, CodeGemma, Codestral
- **Encoder-Decoders**: CodeT5, CodeT5+, PLBART

### Vision Adapter
- **Vision Transformers**: ViT, BEiT, DeiT, Swin
- **CNNs**: ConvNeXt, EfficientNet, ResNet

### Multimodal Adapter
- **Dual Encoders**: CLIP, SigLIP, ALIGN
- **Fusion Models**: BLIP, FLAVA, BridgeTower

### VLM Adapter
- **Generative VLMs**: LLaVA, InternVL, Qwen-VL, Phi-Vision, mPLUG-Owl

## Custom Adapters

```python
from zynthe.core.adapters import ModelAdapter, AdapterRegistry

class AudioModelAdapter(ModelAdapter):
    modality = "audio"

    def prepare_batch(self, batch, model):
        # Filter to audio-model-compatible keys
        ...

    def extract_outputs(self, raw_output):
        from zynthe.core.utils.device_utils import normalize_model_output
        return normalize_model_output(raw_output)

    def get_hookable_layers(self, model):
        # Return transformer layer names
        ...

    def align_dimensions(self, teacher_features, student_features):
        # Project along last dimension
        ...

    def supports_model(self, model):
        # Heuristic for audio model detection
        ...

# Register it
registry = AdapterRegistry()
registry.register("audio", AudioModelAdapter, priority=0)  # highest priority
```
