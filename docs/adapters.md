# Modality Adapters

Zynthé's adapter system normalizes model I/O across architectures so a
single distillation pipeline works with any teacher-student combination.

## Detection Priority

When auto-detecting via `AdapterRegistry.detect(model)`, adapters are
tried in this order:

1. **VLM** — Generative vision-language models (LLaVA, InternVL, Qwen-VL)
2. **Diffusion** — UNet-style denoisers (Stable Diffusion / SDXL)
3. **Audio** — ASR models (Whisper, Wav2Vec2, HuBERT)
4. **Multimodal** — Dual-encoder models (CLIP, SigLIP, BLIP)
5. **Seq2Seq** — Encoder-decoder transformers (T5, BART, Marian)
6. **Vision** — Pure vision models (ViT, Swin, ConvNeXt)
7. **Code** — Code-specific models (CodeLlama, StarCoder, DeepSeek-Coder)
8. **Text** — Text-only models (BERT, GPT-2, LLaMA)
9. **Generic** — Universal HuggingFace-shaped fallback (added in Phase 2)

Generic is intentionally last: it only fires when no typed adapter claimed
the model — it is the honest universal fallback, not a duplicate of the
text-only path.

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
| **Diffusion** | Module names include `down_blocks` AND `up_blocks` (UNet layout) |
| **Audio** | Forward signature includes `input_features` (Whisper) or `input_values` (Wav2Vec / HuBERT) |
| **Multimodal** | Has text_model/text_encoder AND vision_model/visual_encoder, but NO lm_head |
| **Seq2Seq** | Has encoder.* AND decoder.* module-name prefixes AND no `language_model` / `lm_head` |
| **Vision** | Has `pixel_values` in forward signature, but NOT `input_ids` |
| **Code** | Model config `model_type` or `_name_or_path` matches code model patterns |
| **Text** | Has `input_ids` in forward signature |
| **Generic** | Has any well-known HF kwarg (`input_ids`, `pixel_values`, `input_features`, `input_values`); explicit fallback |

## Architecture Coverage

### Text Adapter
- **Encoders**: BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa
- **Decoders**: GPT-2, LLaMA, Mistral, Phi, Gemma
- **Encoder-Decoders**: T5, BART, mBART (also covered by Seq2Seq below)

### Code Adapter
- **Encoders**: CodeBERT, GraphCodeBERT, UniXcoder
- **Decoders**: CodeLlama, StarCoder, DeepSeek-Coder, CodeGemma, Codestral
- **Encoder-Decoders**: CodeT5, CodeT5+, PLBART

### Vision Adapter
- **Vision Transformers**: ViT, BEiT, DeiT, Swin
- **CNNs**: ConvNeXt, EfficientNet, ResNet (also covered by Generic)

### Multimodal Adapter
- **Dual Encoders**: CLIP, SigLIP, ALIGN
- **Fusion Models**: BLIP, FLAVA, BridgeTower

### VLM Adapter
- **Generative VLMs**: LLaVA, InternVL, Qwen-VL, Phi-Vision, mPLUG-Owl

### Seq2Seq Adapter (Phase 2, added)
- **Encoder-Decoder Transformers**: T5, BART, Flan-T5, mBART, Marian, Pegasus

### Audio Adapter (Phase 2, added)
- **ASR Models**: Whisper, Wav2Vec2, HuBERT, SpeechT5, SeamlessM4T
- Detection: forward signature includes `input_features` (Whisper) or
  `input_values` (Wav2Vec2 / HuBERT).

### Diffusion Adapter (Phase 2, experimental)
- **UNet-Style Denoisers**: Stable Diffusion / SDXL UNets
- Detection: module-name pattern matches `down_blocks` / `mid_block` /
  `up_blocks`.
- **Caveat:** Phase 2 stub normalises batch I/O and extracts the
  `sample` tensor from `UNet2DConditionOutput`. Real diffusion KD losses
  arrive in Phase 4.

### Generic Adapter (Phase 2, added)
- Universal fallback. Matches any `nn.Module` whose forward signature
  has at least one well-known HuggingFace kwarg (`input_ids`,
  `pixel_values`, `input_features`, `input_values`).
- Used as the **explicit fallback** when no typed adapter claims a
  model — replaces the historical silent fallback to TextAdapter.

## Universal-Model Smoke Gate

A standalone script `scripts/smoke/universal_smoke.py` runs the 5-family
proof:

| Pair | Teacher | Student | Task | Adapter routed |
|------|---------|---------|------|----------------|
| bert | `hf-internal-testing/tiny-bert` | `prajjwal1/bert-tiny` | sequence_classification | text |
| vit  | `facebook/deit-tiny-patch16-224` | (re-init) | image_classification | vision |
| gpt2 | `sshleifer/tiny-gpt2` | (re-init) | causal_lm | text |
| clip | `openai/clip-vit-base-patch32` | (re-init) | vision_language_contrastive | multimodal |
| resnet | `torchvision resnet18` | (re-init) | image_classification | generic (fallback) |

Run via Modal:

```
modal run scripts/smoke/run_smoke_modal.py --gpu L4 --pairs all --steps 5
```

Exit code `0` means all 5 pairs built a pipeline, loaded the right
adapter, and ran forward + backward (loss is allowed to be non-finite;
smoke proves the pipeline runs, not that it converges). Non-finite
loss is logged as a warning rather than a hard failure.

Verified end-to-end on Modal L4 (commit `f20ee90` + successors):

| Adapter route | Pair |
|--------------|------|
| text         | bert  |
| vision       | vit   |
| text         | gpt2  |
| multimodal   | clip  |
| generic      | resnet |

A CI nightly schedule (`cron`) is out of scope for Phase 2 — the Modal
runner is developer-only. A scheduled run can be added in Phase 5
once the maintainer adds `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` to
GitHub secrets.
