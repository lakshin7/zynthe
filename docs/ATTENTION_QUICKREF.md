# 🎯 Attention Transfer Quick Reference

## Method Selection Table

| Your Model Type | Recommended Config | Methods to Enable |
|----------------|-------------------|-------------------|
| **Text Transformers** (BERT, GPT) | `type: ["spatial", "self"]` | `use_attention_rollout: true` |
| **Vision Models** (ResNet→MobileNet) | `type: ["spatial", "scat"]` | Standard AT |
| **Vision Transformers** (ViT) | `type: ["self", "probabilistic"]` | `use_attention_rollout: true`<br>`use_dual_matching: true` |
| **Multimodal** (CLIP, BLIP) | `type: ["spatial", "relational"]` | `use_dual_matching: true`<br>`loss_types: ["l2", "contrastive"]` |
| **Video Models** (TimeSformer) | `type: ["spatial", "self"]` | `use_temporal_attention: true`<br>`use_cross_layer_flow: true` |

---

## Configuration Patterns

### Pattern 1: Basic (Fast, Low Memory)
```yaml
attention_transfer:
  enabled: true
  type: ["spatial"]
  loss_types: ["l2"]
  weight: 0.25
```

### Pattern 2: Balanced (Recommended)
```yaml
attention_transfer:
  enabled: true
  type: ["spatial", "relational"]
  loss_types: ["l2", "kl"]
  loss_weights: [0.7, 0.3]
  use_attention_rollout: true
  weight: 0.25
```

### Pattern 3: Advanced (Best Quality, Slower)
```yaml
attention_transfer:
  enabled: true
  type: ["spatial", "self", "relational", "probabilistic"]
  loss_types: ["l2", "kl", "contrastive"]
  loss_weights: [0.5, 0.3, 0.2]
  use_attention_rollout: true
  use_dual_matching: true
  weight: 0.3
```

---

## Loss Types Cheat Sheet

| Loss Type | Formula | Use Case | Speed |
|-----------|---------|----------|-------|
| **l2** | `MSE(A_s, A_t)` | Base AT, general purpose | ⚡⚡⚡ Fast |
| **kl** | `KL(A_s \|\| A_t)` | Probabilistic matching | ⚡⚡ Medium |
| **contrastive** | `1 - cos(A_s, A_t)` | Embeddings, multimodal | ⚡⚡ Medium |
| **relational** | `\|\|A_sA_s^T - A_tA_t^T\|\|_F^2` | Deep similarity | ⚡ Slower |

---

## Normalization Guide

| Normalization | When to Use | Effect |
|--------------|-------------|---------|
| **softmax** | Transformers, probabilistic | Converts to probability distribution |
| **l2** | CNNs, general | Unit norm, preserves relative magnitude |
| **sigmoid** | Binary attention | Squashes to [0, 1] range |
| **none** | Raw features | No normalization |

---

## Advanced Methods Matrix

| Method | Computational Cost | Memory | Accuracy Gain | Use When |
|--------|-------------------|---------|---------------|----------|
| **Attention Rollout** | 🟢 Low | 🟢 Low | 🟡 +0.5-1% | Interpretability needed |
| **Cross-layer Flow** | 🔴 High | 🔴 High | 🟢 +1-2% | Deep models, many layers |
| **Dual Matching** | 🟡 Medium | 🟡 Medium | 🟢 +1-2% | Multimodal models |
| **Temporal Attention** | 🔴 High | 🔴 High | 🟢 +1-3% | Video models only |

---

## Command Templates

### Run with Attention Transfer
```bash
python app/main.py --config configs/attention_transfer_advanced.yaml
```

### Test on Multimodal (Template)
```bash
python app/main.py --config configs/attention_multimodal.yaml
```

### Evaluate Attention Quality
```python
from core.distillers.attention_transfer import AttentionTransferDistiller

distiller = AttentionTransferDistiller.from_config(teacher, student, config)
metrics = distiller.evaluate_attention_quality(val_loader, device)
print(f"Alignment: {metrics['alignment_scores']['cosine_similarity']:.4f}")
```

---

## Layer Name Finder

**Find layer names in your model:**
```python
# List all layers
for name, module in model.named_modules():
    print(name)

# Or for specific pattern
layers = [n for n, _ in model.named_modules() if 'encoder.layer' in n]
print(layers)
```

**Common layer patterns:**
- BERT: `"encoder.layer.6"`, `"encoder.layer.11"`
- ResNet: `"layer3"`, `"layer4"`
- ViT: `"encoder.layer.6"`, `"encoder.layer.11"`
- CLIP Vision: `"vision_model.encoder.layers.11"`
- CLIP Text: `"text_model.encoder.layers.11"`

---

## Troubleshooting Quick Fixes

| Error | Fix |
|-------|-----|
| "No matched layers" | Check layer names with `model.named_modules()` |
| "Shape mismatch" | Set `normalization: "softmax"` |
| "Out of memory" | Reduce `batch_size`, disable `use_cross_layer_flow` |
| "Loss is NaN" | Lower `temperature`, check `loss_weights` sum to 1.0 |
| "Too slow" | Use only `loss_types: ["l2"]`, disable advanced methods |

---

## Performance Tuning

### For Speed (Mac M2 8GB)
```yaml
attention_transfer:
  type: ["spatial"]
  loss_types: ["l2"]
  use_attention_rollout: false
  use_dual_matching: false
  weight: 0.2
```

### For Quality (Mac M2 16GB+)
```yaml
attention_transfer:
  type: ["spatial", "self", "relational"]
  loss_types: ["l2", "kl", "contrastive"]
  loss_weights: [0.5, 0.3, 0.2]
  use_attention_rollout: true
  use_dual_matching: true
  weight: 0.3
```

---

## Metric Interpretation

### Cosine Similarity
- **0.9-1.0**: Excellent alignment ✅
- **0.7-0.9**: Good alignment 👍
- **0.5-0.7**: Moderate alignment 🟡
- **<0.5**: Poor alignment ❌

### L2 Distance
- **<0.1**: Very close ✅
- **0.1-0.5**: Reasonable 👍
- **0.5-1.0**: Some difference 🟡
- **>1.0**: Large difference ❌

### KL Divergence
- **<0.5**: Similar distributions ✅
- **0.5-2.0**: Moderate difference 👍
- **2.0-5.0**: Significant difference 🟡
- **>5.0**: Very different ❌

---

## One-Line Examples

**Basic spatial attention:**
```yaml
attention_transfer: {enabled: true, type: ["spatial"], loss_types: ["l2"], weight: 0.25}
```

**Transformer with rollout:**
```yaml
attention_transfer: {enabled: true, type: ["self"], use_attention_rollout: true, weight: 0.25}
```

**Multimodal dual matching:**
```yaml
attention_transfer: {enabled: true, type: ["spatial"], use_dual_matching: true, loss_types: ["l2", "contrastive"], weight: 0.3}
```

---

## File Outputs

After training, find these in `experiments/<exp_id>/`:
- `attention_metrics.csv` - Per-layer alignment scores
- `attention_maps/` - Visualization heatmaps
- `COMPARISON_REPORT.md` - Full evaluation report
- `attention_config.yaml` - Config used for this run

---

**🔥 Pro Tip**: Start with Pattern 2 (Balanced), then optimize based on `attention_metrics.csv` results!
