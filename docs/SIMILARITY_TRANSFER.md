# Similarity Transfer - The Geometric Soul of Knowledge Distillation

## Overview

**Similarity Transfer** is a relational knowledge distillation technique that goes beyond individual feature and logit transfer. It captures the "geometric soul" of the teacher's understanding by preserving structural relationships between samples.

**Status**: ✅ Fully Implemented and Tested  
**Location**: `core/distillers/similarity_transfer.py`  
**Configuration**: `configs/similarity_transfer.yaml`  
**Test Script**: `test_similarity_transfer.py`

---

## Mathematical Foundation

### Core Concept
Instead of transferring individual features, Similarity Transfer preserves **pairwise relationships** between samples:

```
L_sim = ||S_teacher - S_student||²_F
```

Where:
- `S` is the pairwise similarity matrix: `S = normalize(F) @ normalize(F)^T`
- `F` = feature embeddings `[batch_size, feature_dim]`
- `||·||²_F` = Frobenius norm (element-wise squared differences)

### Structural Alignment Score (SAS)
Measures how well the student preserves teacher's relational structure:

```
SAS = 1 - (||S_t - S_s||_F / √(batch_size²)) ∈ [0, 1]
```

**Higher is better**: `1.0` = perfect structural alignment

---

## Key Features

### 1. **Pairwise Similarity Matrices** 🔗
Three metrics for computing sample relationships:

#### Cosine Similarity (Default)
- **Formula**: `S_ij = (f_i · f_j) / (||f_i|| ||f_j||)`
- **Best for**: General relational knowledge
- **Properties**: Angle-based, rotation invariant
- **Use case**: Standard distillation with relationship preservation

#### Euclidean Distance
- **Formula**: `S_ij = exp(-||f_i - f_j||² / T)`
- **Best for**: Scale-sensitive relationships
- **Properties**: Distance-based, temperature-controlled
- **Use case**: When absolute distances matter

#### Graph-Based
- **Formula**: `S_ij = A_ij` (learned adjacency matrix)
- **Best for**: Sparse relational structures
- **Properties**: Adaptive topology, threshold-based
- **Use case**: Large batch sizes, sparse relationships

### 2. **Progressive Layer Transfer** 📈
Hierarchical learning: shallow → deep

- **Start**: Transfer from shallow layers first
- **Progress**: Gradually add deeper layers each epoch
- **Benefits**: 
  - Stable training (complexity increases gradually)
  - Better convergence (foundational knowledge first)
  - Mimics human learning (simple → complex)

**Example**:
```
Epoch 0-2: layer_2 only
Epoch 3-5: layer_2 + layer_4
Epoch 6+:  layer_2 + layer_4 + layer_6
```

### 3. **Cross-Modality Alignment** 🌉
For multimodal models (CLIP-style vision + text):

- **Aligns**: Vision and text embeddings across modalities
- **Loss**: Cross-modal similarity preservation
- **Use case**: Distilling multimodal transformers
- **Inspired by**: CLIP training methodology

### 4. **Graph-Based Similarity** 🕸️
Research-grade structural distillation:

- **Sparse adjacency**: Only strong relationships preserved
- **Adaptive thresholding**: Learns optimal connectivity
- **Benefits**:
  - Memory efficient for large batches
  - Focuses on significant relationships
  - Prevents overfitting to noise

### 5. **Multi-Stage Pipeline Integration** 🔗
Perfect fit as **Stage 3** in the distillation pipeline:

```
Stage 1: KD (Logit Alignment)        → α = 0.9
Stage 2: Feature (Layer Refinement)  → β = 0.6
Stage 3: Similarity (Relational)     → γ = 0.4  ← YOU ARE HERE
Stage 4: Attention (Imitation)       → δ = 0.3
Stage 5: QAT (Quantization)          → int8
```

**Why Stage 3?**
- Builds on foundation from KD + Feature stages
- Adds relational understanding before fine-grained attention
- Captures "geometric soul" of teacher's knowledge representation

---

## Usage

### Basic Configuration

```yaml
distiller:
  type: "similarity"
  config:
    layer: "transformer.layer.5"  # Single layer
    similarity_metric: "cosine"    # cosine / euclidean / graph
    weight: 1.0                    # Similarity loss weight
    temperature: 4.0               # KD temperature
    kd_weight: 0.5                 # Combined KD weight
    normalize: true                # Normalize features
```

### Progressive Layer Transfer

```yaml
distiller:
  type: "similarity"
  config:
    layers: ["transformer.layer.2", "transformer.layer.4", "transformer.layer.6"]
    similarity_metric: "cosine"
    progressive: true               # Enable progressive transfer
    progressive_epochs: 3           # Add layer every 3 epochs
```

### Graph-Based Similarity

```yaml
distiller:
  type: "similarity"
  config:
    layer: "transformer.layer.5"
    similarity_metric: "graph"      # Graph-based
    graph_mode: true                # Enable graph features
    graph_threshold: 0.5            # Adaptive threshold
```

### Multi-Stage Integration

```yaml
multi_stage:
  stages:
    - name: "logit_alignment"
      type: "kd"
      epochs: 2
      
    - name: "feature_distillation"
      type: "feature"
      epochs: 2
      
    - name: "similarity_transfer"  # Stage 3
      type: "similarity"
      epochs: 3
      config:
        layer: "transformer.layer.5"
        similarity_metric: "cosine"
        progressive: true
```

---

## Python API

### Basic Usage

```python
from core.distillers.similarity_transfer import SimilarityTransfer, create_similarity_config

# Create configuration
config = create_similarity_config(
    layer="transformer.layer.5",
    similarity_metric="cosine",
    weight=1.0,
    progressive=False
)

# Initialize distiller
distiller = SimilarityTransfer(teacher, student, config)

# Training loop
for inputs, labels in dataloader:
    outputs = distiller(inputs, labels)
    loss = outputs['loss']
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Track metrics
    print(f"SAS Score: {outputs['sas_score']:.4f}")
```

### Progressive Training

```python
config = create_similarity_config(
    layers=["layer_2", "layer_4", "layer_6"],
    progressive=True,
    progressive_epochs=3
)

distiller = SimilarityTransfer(teacher, student, config)

for epoch in range(10):
    distiller.update_epoch(epoch)  # Updates active layers
    
    for batch in dataloader:
        metrics = distiller.train_step(batch, optimizer)
        print(f"Active layers: {len(distiller.current_layers)}")
```

---

## Test Results

```bash
$ python test_similarity_transfer.py
```

**Outputs**:
```
✅ Cosine Similarity: Loss = -0.1611, SAS = 0.9986
✅ Progressive Transfer: Layers grow from 1 → 2 → 3
✅ Graph-Based: Loss = -0.1430, SAS = 1.0000
✅ Euclidean Distance: Loss = -0.1647, SAS = 1.0000
✅ Training Step: Loss improved by 0.0402
✅ Comprehensive Metrics: All components tracked
```

**All tests passed!** ✅

---

## Integration with Multi-Stage Pipeline

### Registration
Automatically registered in `DistillerRegistry`:
```python
registry = DistillerRegistry()
# Available: ['kd', 'feature', 'similarity', 'attention', 'qat']
```

### Aliases
- `similarity` → SimilarityTransfer
- `similarity_transfer` → SimilarityTransfer

### Configuration File
Complete examples in: `configs/similarity_transfer.yaml`

---

## Metrics Tracked

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| `loss` | Total combined loss | `(-∞, ∞)` | Lower is better |
| `similarity_loss` | Relational structure loss | `[0, ∞)` | Lower is better |
| `kd_loss` | Knowledge distillation loss | `[0, ∞)` | Lower is better |
| `ce_loss` | Cross-entropy loss | `[0, ∞)` | Lower is better |
| `sas_score` | Structural Alignment Score | `[0, 1]` | Higher is better (1.0 = perfect) |
| `sas_layer_X` | Per-layer SAS | `[0, 1]` | Higher is better |
| `sim_loss_layer_X` | Per-layer similarity | `[0, ∞)` | Lower is better |

---

## When to Use

### ✅ Use Similarity Transfer When:
1. **Relational knowledge matters** - Classification depends on sample relationships
2. **Teacher captures structure** - Teacher learned meaningful semantic geometry
3. **Multi-stage pipeline** - As Stage 3 after KD + Feature distillation
4. **Sufficient batch size** - Need multiple samples to compute relationships (batch_size ≥ 8)
5. **Hierarchical learning** - Want progressive shallow → deep transfer

### ❌ Don't Use When:
1. **Very small batches** - batch_size < 4 (not enough relationships)
2. **Independent samples** - No meaningful relationships between samples
3. **Simple tasks** - Basic classification where individual features suffice
4. **Limited compute** - Pairwise similarity adds O(batch_size²) complexity

---

## Comparison with Other Distillers

| Distiller | Transfers | Granularity | Computational Cost |
|-----------|-----------|-------------|-------------------|
| **KD-Hinton** | Logits | Class probabilities | Low |
| **Feature** | Intermediate features | Layer activations | Medium |
| **Similarity** | Sample relationships | Pairwise structure | Medium-High |
| **Attention** | Attention maps | Token interactions | Medium |

**Unique to Similarity Transfer**:
- Only distiller that preserves **relational structure**
- Captures "geometric soul" beyond individual features
- Progressive layer transfer capability
- Cross-modality alignment for multimodal models

---

## Advanced Features

### 1. **Multi-Layer Similarity**
Transfer relationships from multiple layers simultaneously:
```yaml
layers: ["transformer.layer.2", "transformer.layer.4", "transformer.layer.6"]
```

### 2. **Temperature Scaling**
Control similarity sharpness:
```yaml
temperature: 4.0  # Higher = softer similarities
```

### 3. **Loss Weighting**
Balance similarity vs KD:
```yaml
weight: 0.7      # Similarity weight
kd_weight: 0.3   # KD weight
```

### 4. **Normalization**
Normalize features before similarity:
```yaml
normalize: true  # Unit sphere projection
```

---

## Future Enhancements

### Planned Features
1. **Attention-Weighted Similarity** - Weight relationships by attention scores
2. **Temporal Similarity** - For sequential/time-series data
3. **Hierarchical Clustering** - Group similar samples during training
4. **Contrastive Objectives** - Combine with contrastive learning
5. **Visualization Tools** - Plot similarity matrices and SAS evolution

### Research Directions
1. **Optimal Layer Selection** - Automatically determine best layers
2. **Dynamic Thresholding** - Learn graph threshold during training
3. **Multi-Modal Extensions** - Better cross-modality alignment
4. **Efficiency Improvements** - Approximate similarities for large batches

---

## References

### Theoretical Foundation
- **Similarity-Preserving Knowledge Distillation** (Tung & Mori, 2019)
- **Relational Knowledge Distillation** (Park et al., 2019)
- **Structural Knowledge Distillation** (Liu et al., 2019)

### Inspiration
- **CLIP** (Radford et al., 2021) - Cross-modality alignment
- **Graph Neural Networks** - Relational learning
- **Metric Learning** - Pairwise distance preservation

---

## Citation

If you use Similarity Transfer in your research:

```bibtex
@software{zynthe_similarity_transfer_2025,
  title = {Similarity Transfer: Relational Knowledge Distillation},
  author = {Zynthe Knowledge Distillation Toolkit},
  year = {2025},
  url = {https://github.com/zynthe/knowledge-distillation-toolkit}
}
```

---

## Contact & Support

**Issues**: File on GitHub Issues  
**Questions**: Discussions tab  
**Contributions**: Pull requests welcome!

---

## Summary

**Similarity Transfer** is a powerful Stage 3 distiller that captures the "geometric soul" of teacher understanding through relational structure preservation. With support for progressive training, multiple similarity metrics, and seamless multi-stage integration, it's an essential tool for advanced knowledge distillation.

**Key Innovations**:
- 🧬 Preserves sample relationships, not just features
- 📈 Progressive layer transfer for stable learning
- 🌉 Cross-modality alignment for multimodal models
- 🕸️ Graph-based sparse similarity
- 📊 SAS metric for relationship quality tracking

**The Geometric Soul of KD**: Capturing what lies beyond individual predictions—the structural essence of understanding.
