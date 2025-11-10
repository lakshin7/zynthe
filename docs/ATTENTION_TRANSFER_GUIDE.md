# Advanced Attention Transfer - Complete Guide

## 🎯 Overview

Zynthe's **Attention Transfer (AT)** module has been upgraded with cutting-edge techniques for knowledge distillation. This guide covers all classical and advanced methods available.

---

## 📊 Feature Matrix

| Method | Description | Use Case | Config Flag |
|--------|-------------|----------|-------------|
| **Spatial Attention** | L2 distance on spatial attention maps | CNNs, ViTs | `type: ["spatial"]` |
| **Self-Attention Matching** | Align transformer attention heads | BERT, GPT models | `type: ["self"]` |
| **Affinity/Relational** | Cosine relational structure (RKD) | Deep similarity learning | `type: ["affinity"]` |
| **Probabilistic Attention** | KL divergence on attention distributions | Uncertainty-aware KD | `type: ["probabilistic"]` |
| **SCAT** | Spatial-Channel Split Attention | CNNs with channel attention | `type: ["scat"]` |
| **Attention Rollout** ⭐ | Aggregate multi-head attention across layers | Transformer interpretability | `use_attention_rollout: true` |
| **Cross-layer Attention Flow** ⭐ | Backward attention propagation | Deep networks, layer alignment | `use_cross_layer_flow: true` |
| **Dual Attention Matching** ⭐ | Feature + token space alignment | Multimodal KD (CLIP, BLIP) | `use_dual_matching: true` |
| **Temporal Attention** ⭐ | Time-axis attention alignment | Video models | `use_temporal_attention: true` |

⭐ = NEW advanced methods

---

## 🏗️ Architecture Components

### 1. **Attention Extractor Module**

Hooks into intermediate layers to extract attention maps.

```python
from core.distillers.attention_transfer import AttentionExtractor

# Create extractor
extractor = AttentionExtractor(
    model=teacher_model,
    layer_names=["layer3", "layer4"],
    model_type="transformer"  # or "cnn", "multimodal", "video"
)

# Extract attention maps
attention_maps = extractor.extract_attention_maps()
```

**Capabilities:**
- **CNNs**: Extracts feature maps → computes spatial attention
- **Transformers**: Extracts self-attention scores from each block
- **Multimodal**: Extracts both self-attention and cross-attention
- **Video**: Extracts temporal attention weights

---

### 2. **Attention Matcher**

Aligns teacher & student attentions with resize and normalization.

```python
from core.distillers.attention_transfer import AttentionMatcher

matcher = AttentionMatcher(
    normalization="softmax",  # "l2", "softmax", "sigmoid", "none"
    interpolation_mode="bilinear",
    layer_mapping={"layer4": "layer2"}  # Map teacher → student
)

# Match and align layers
matched_pairs = matcher.match_layers(teacher_attentions, student_attentions)
```

**Features:**
- Resize maps (F.interpolate) for mismatched resolutions
- Normalize using L2, softmax, or sigmoid
- Match corresponding layers based on depth correlation

---

### 3. **Attention Loss Composer**

Flexible multi-loss computation.

```python
from core.distillers.attention_transfer import AttentionLossComposer

composer = AttentionLossComposer(
    loss_types=["l2", "kl", "contrastive"],
    weights=[0.5, 0.3, 0.2],
    temperature=2.0
)

loss = composer.compute(student_attn, teacher_attn)
```

**Supported Loss Types:**
- **L2**: Base AT (Zagoruyko & Komodakis, 2017)
- **KL**: Probabilistic matching
- **Contrastive**: For embeddings or cross-modal
- **Relational**: Gram matrix matching (||QtQt^T - QsQs^T||_F²)

---

## 🚀 Usage Examples

### Example 1: Basic Attention Transfer

```yaml
# configs/my_attention.yaml
distillation:
  method: "attention_transfer"
  temperature: 2.0
  alpha: 0.5

attention_transfer:
  enabled: true
  type: ["spatial", "relational"]
  normalization: "softmax"
  loss_types: ["l2", "kl"]
  loss_weights: [0.7, 0.3]
  weight: 0.25
```

```bash
python app/main.py --config configs/my_attention.yaml
```

---

### Example 2: Advanced Transformer with Attention Rollout

```yaml
attention_transfer:
  enabled: true
  type: ["self", "probabilistic"]
  
  # Enable attention rollout for interpretability
  use_attention_rollout: true
  
  # Dual matching for feature + token alignment
  use_dual_matching: true
  
  loss_types: ["l2", "kl", "contrastive"]
  loss_weights: [0.4, 0.4, 0.2]
```

---

### Example 3: Multimodal (CLIP-style)

```yaml
model:
  name: "openai/clip-vit-base-patch32"
  student_name: "openai/clip-vit-base-patch16"
  type: "multimodal"

attention_transfer:
  enabled: true
  type: ["spatial", "self", "relational"]
  
  # Essential for multimodal
  use_dual_matching: true
  
  # Extract from specific layers
  teacher_layers:
    - "vision_model.encoder.layers.11"
    - "text_model.encoder.layers.11"
  
  student_layers:
    - "vision_model.encoder.layers.5"
    - "text_model.encoder.layers.5"
  
  # Multimodal benefits from contrastive
  loss_types: ["l2", "contrastive", "relational"]
  loss_weights: [0.4, 0.4, 0.2]
```

---

### Example 4: Video Models with Temporal Attention

```yaml
model:
  name: "facebook/timesformer-base-finetuned-k400"
  type: "video"
  num_frames: 8

attention_transfer:
  enabled: true
  
  # Temporal attention for video
  use_temporal_attention: true
  use_attention_rollout: true
  use_cross_layer_flow: true
  
  loss_types: ["l2", "relational"]
```

---

## 📐 Advanced Methods Explained

### 1. Attention Rollout

**What it does**: Aggregates multi-head attention across layers to trace information flow.

**Reference**: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)

**Use case**: Understanding how attention propagates through deep transformers.

**Math**:
```
Rollout(L) = ∏ᴸᵢ₌₁ (Aᵢ + I) / 2
```
where Aᵢ is attention at layer i, I is identity matrix.

---

### 2. Cross-layer Attention Flow

**What it does**: Propagates teacher's attention backward to guide earlier student layers.

**Use case**: Aligning attention patterns across different depths.

**Implementation**: Uses gradient flow to backpropagate attention guidance.

**Warning**: Computationally intensive, requires custom backprop hooks.

---

### 3. Dual Attention Matching

**What it does**: Combines feature-space attention (activations) with token-space attention (self-attn).

**Use case**: Very useful for multimodal KD where both visual features and text tokens need alignment.

**Loss**:
```
L_dual = λ₁ * MSE(A_feat^s, A_feat^t) + λ₂ * MSE(A_token^s, A_token^t)
```

---

### 4. Temporal Attention Transfer

**What it does**: For video models — aligns temporal attention weights across frames.

**Use case**: Video transformers (TimeSformer, VideoMAE, etc.)

**Implementation**: Adds time-axis to attention extractor, handles [B, T, H, W] tensors.

---

## 📊 Evaluation Metrics

Zynthe automatically computes and saves:

### 1. Attention Alignment Score

```python
distiller.compute_attention_alignment_score(teacher_attns, student_attns)
```

Returns:
- **Cosine similarity**: Average cosine similarity across layers
- **L2 distance**: Average L2 distance
- **KL divergence**: Average KL divergence
- **Correlation**: Pearson correlation coefficient

### 2. Interpretability Score

```python
distiller.compute_interpretability_score(student_attentions, gradients)
```

Measures how well student attention aligns with Grad-CAM style importance.

### 3. Attention Visualization

```python
distiller.visualize_attention_comparison(
    teacher_attentions,
    student_attentions,
    save_path="attention_heatmap.png"
)
```

Generates side-by-side heatmaps for debugging.

---

## 🔧 Configuration Reference

### Complete Config Template

```yaml
attention_transfer:
  # Core settings
  enabled: true
  type: ["spatial", "self", "relational", "probabilistic"]
  
  # Advanced methods
  use_attention_rollout: true
  use_dual_matching: true
  use_cross_layer_flow: false
  use_temporal_attention: false
  
  # Layer configuration
  teacher_layers: ["encoder.layer.6", "encoder.layer.11"]
  student_layers: ["encoder.layer.3", "encoder.layer.5"]
  layer_mapping:
    "encoder.layer.11": "encoder.layer.5"
  
  # Normalization
  normalization: "softmax"  # "l2", "softmax", "sigmoid", "none"
  
  # Loss composition
  loss_types: ["l2", "kl", "contrastive", "relational"]
  loss_weights: [0.4, 0.3, 0.2, 0.1]
  
  # Overall weight
  weight: 0.25
  temperature: 2.0
```

---

## 🎨 Visualization Features

### Attention Map Comparison

Automatically generates:
- Side-by-side heatmaps (teacher vs student)
- Layer-wise attention evolution
- Attention flow diagrams (for rollout)

Saved to: `experiments/<exp_id>/attention_maps/`

### Metrics Dashboard

CSV file with:
- Per-layer cosine similarity
- Per-layer L2 distance
- Per-layer KL divergence
- Correlation scores

Saved to: `experiments/<exp_id>/attention_metrics.csv`

---

## 🧪 Testing

### Quick Test Script

```python
from core.distillers.attention_transfer import AttentionTransferDistiller

# Load models
teacher, student, tokenizer = load_models(config)

# Create distiller
distiller = AttentionTransferDistiller.from_config(
    teacher=teacher,
    student=student,
    config=config
)

# Evaluate attention quality
metrics = distiller.evaluate_attention_quality(val_loader, device)

print(f"Cosine Similarity: {metrics['alignment_scores']['cosine_similarity']:.4f}")
print(f"L2 Distance: {metrics['alignment_scores']['l2_distance']:.4f}")
print(f"Correlation: {metrics['alignment_scores']['correlation']:.4f}")
```

---

## 📚 References

1. **Attention Transfer**: Zagoruyko & Komodakis, "Paying More Attention to Attention" (2017)
2. **Attention Rollout**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)
3. **Relational KD**: Park et al., "Relational Knowledge Distillation" (2019)
4. **Self-Attention Distillation**: Zhang et al., "Be Your Own Teacher" (2019)

---

## 🐛 Troubleshooting

### Issue: "No matched layers for comparison"

**Solution**: Check layer names with:
```python
print(dict(model.named_modules()).keys())
```

### Issue: "Shape mismatch error"

**Solution**: Enable automatic resizing:
```yaml
attention_transfer:
  normalization: "softmax"  # Handles shape differences
```

### Issue: "Memory issues with video models"

**Solution**: Reduce batch size and use gradient accumulation:
```yaml
train:
  batch_size: 2
  grad_accum_steps: 8
```

---

## 🚀 Future Extensions

Planned features:
- [ ] Grad-CAM integration for interpretability
- [ ] Attention-guided pruning
- [ ] Dynamic layer mapping based on correlation matrix
- [ ] Support for object detection (attention on bounding boxes)
- [ ] Audio-visual attention for multimodal speech models

---

## 📞 Support

For questions or issues with attention transfer:
1. Check the configuration examples in `configs/attention_*.yaml`
2. Review layer names with `print(model.named_modules())`
3. Enable visualization: `visualize_attention: true`
4. Check attention metrics in `experiments/<exp_id>/attention_metrics.csv`

---

**Happy Distilling! 🎓✨**
