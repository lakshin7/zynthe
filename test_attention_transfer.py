#!/usr/bin/env python3
"""
Quick test script for Advanced Attention Transfer features.
Tests all components: Extractor, Matcher, Loss Composer, and full pipeline.
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from core.distillers.attention_transfer import (
    AttentionExtractor,
    AttentionMatcher,
    AttentionLossComposer,
    AttentionTransferDistiller
)

print("="*70)
print("🧪 Testing Advanced Attention Transfer Components")
print("="*70)

# ============================================================================
# 1. Test Attention Extractor
# ============================================================================
print("\n1️⃣ Testing Attention Extractor...")

class DummyTransformer(nn.Module):
    """Minimal transformer for testing."""
    def __init__(self, hidden_size=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
            for _ in range(num_layers)
        ])
        self.embedding = nn.Linear(128, hidden_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x

# Create model
model = DummyTransformer(hidden_size=256, num_layers=4)

# Create extractor
extractor = AttentionExtractor(
    model=model,
    layer_names=["layers.2", "layers.3"],
    model_type="transformer"
)

# Test forward pass
dummy_input = torch.randn(2, 16, 128)  # [batch, seq_len, input_dim]
_ = model(dummy_input)

# Extract attention maps
attention_maps = extractor.extract_attention_maps()
print(f"   ✅ Extracted {len(attention_maps)} attention maps")
print(f"   📊 Feature maps captured: {list(extractor.feature_maps.keys())}")

extractor.clear()
extractor.remove_hooks()

# ============================================================================
# 2. Test Attention Matcher
# ============================================================================
print("\n2️⃣ Testing Attention Matcher...")

matcher = AttentionMatcher(
    normalization="softmax",
    layer_mapping={"layer_4": "layer_2"}
)

# Create sample attention maps with different sizes
teacher_attn = torch.randn(2, 32, 32)  # [batch, H, W]
student_attn = torch.randn(2, 16, 16)  # Different size

# Test resize
resized = matcher.resize(student_attn, teacher_attn)
print(f"   ✅ Resized student from {student_attn.shape} to {resized.shape}")

# Test normalization
normalized = matcher.normalize(teacher_attn)
print(f"   ✅ Normalized attention: sum={normalized.sum().item():.4f}")

# Test layer matching
teacher_attns = {"layer_1": torch.randn(2, 8, 8), "layer_2": torch.randn(2, 16, 16)}
student_attns = {"layer_1": torch.randn(2, 4, 4), "layer_2": torch.randn(2, 8, 8)}

matched_pairs = matcher.match_layers(teacher_attns, student_attns)
print(f"   ✅ Matched {len(matched_pairs)} layer pairs")

# ============================================================================
# 3. Test Attention Loss Composer
# ============================================================================
print("\n3️⃣ Testing Attention Loss Composer...")

composer = AttentionLossComposer(
    loss_types=["l2", "kl", "contrastive", "relational"],
    weights=[0.4, 0.3, 0.2, 0.1],
    temperature=2.0
)

teacher_attn = torch.randn(4, 64)  # [batch, features]
student_attn = torch.randn(4, 64)

# Test individual losses
l2_loss = composer.l2_loss(student_attn, teacher_attn)
kl_loss = composer.kl_loss(student_attn, teacher_attn)
contrastive_loss = composer.contrastive_loss(student_attn, teacher_attn)
relational_loss = composer.relational_loss(student_attn, teacher_attn)

print(f"   ✅ L2 Loss: {l2_loss.item():.4f}")
print(f"   ✅ KL Loss: {kl_loss.item():.4f}")
print(f"   ✅ Contrastive Loss: {contrastive_loss.item():.4f}")
print(f"   ✅ Relational Loss: {relational_loss.item():.4f}")

# Test combined loss
total_loss = composer.compute(student_attn, teacher_attn)
print(f"   ✅ Combined Loss: {total_loss.item():.4f}")

# ============================================================================
# 4. Test Full Distiller
# ============================================================================
print("\n4️⃣ Testing Full AttentionTransferDistiller...")

class DummyBERT(nn.Module):
    """Minimal BERT-like model for testing."""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.embedding = nn.Linear(128, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4),
            num_layers=4
        )
        self.classifier = nn.Linear(hidden_size, 2)
    
    def forward(self, x, output_attentions=False, output_hidden_states=False, **kwargs):
        x = self.embedding(x)
        x = self.encoder(x)
        logits = self.classifier(x.mean(dim=1))
        
        # Mock outputs
        class Output:
            def __init__(self, logits, hidden):
                self.logits = logits
                self.last_hidden_state = hidden
                self.attentions = None
                self.hidden_states = None
        
        return Output(logits, x)

teacher = DummyBERT(hidden_size=256)
student = DummyBERT(hidden_size=128)

# Test from_config factory method
config = {
    "attention_transfer": {
        "enabled": True,
        "type": ["spatial", "relational"],
        "weight": 0.25,
        "temperature": 2.0,
        "normalization": "softmax",
        "loss_types": ["l2", "kl"],
        "loss_weights": [0.6, 0.4],
        "use_attention_rollout": True,
        "use_dual_matching": True,
        "use_cross_layer_flow": False,
        "use_temporal_attention": False
    }
}

distiller = AttentionTransferDistiller.from_config(teacher, student, config)
print(f"   ✅ Created distiller with mode: {distiller.mode}")
print(f"   ✅ Loss types: {distiller.loss_composer.loss_types}")
print(f"   ✅ Alpha: {distiller.alpha}")

# Test forward pass
dummy_input = torch.randn(2, 16, 128)
teacher.train()
student.train()

loss = distiller.forward(dummy_input, return_loss=True)
print(f"   ✅ Forward pass successful, loss: {loss.item():.4f}")

# Test evaluation mode
teacher.eval()
student.eval()

output = distiller.forward(dummy_input, return_loss=False)
print(f"   ✅ Evaluation mode successful, output shape: {output.logits.shape}")

# ============================================================================
# 5. Test Advanced Methods
# ============================================================================
print("\n5️⃣ Testing Advanced Methods...")

# Test Attention Rollout
attention_weights = [
    torch.randn(2, 4, 16, 16) for _ in range(3)  # [batch, heads, seq, seq]
]
rollout = distiller.attention_rollout(attention_weights, residual=True)
print(f"   ✅ Attention Rollout: {rollout.shape}")

# Test Cross-layer Attention Flow
teacher_attentions = [torch.randn(2, 4, 16, 16) for _ in range(4)]
student_attentions = [torch.randn(2, 4, 16, 16) for _ in range(2)]
flow_loss = distiller.cross_layer_attention_flow(teacher_attentions, student_attentions)
print(f"   ✅ Cross-layer Flow Loss: {flow_loss.item():.4f}")

# Test Dual Attention Matching
teacher_feats = {
    "feature_map": torch.randn(2, 256, 8, 8),
    "attn_matrix": torch.randn(2, 4, 16, 16)
}
student_feats = {
    "feature_map": torch.randn(2, 128, 4, 4),
    "attn_matrix": torch.randn(2, 4, 8, 8)
}
dual_loss = distiller.dual_attention_matching(teacher_feats, student_feats)
print(f"   ✅ Dual Matching Loss: {dual_loss.item():.4f}")

# Test Temporal Attention Transfer
teacher_temporal = torch.randn(2, 8, 16, 16)  # [batch, time, H, W]
student_temporal = torch.randn(2, 4, 8, 8)
temporal_loss = distiller.temporal_attention_transfer(teacher_temporal, student_temporal)
print(f"   ✅ Temporal Loss: {temporal_loss.item():.4f}")

# ============================================================================
# 6. Test Evaluation Metrics
# ============================================================================
print("\n6️⃣ Testing Evaluation Metrics...")

teacher_attns = {
    "layer_0": torch.randn(2, 16, 16),
    "layer_1": torch.randn(2, 32, 32)
}
student_attns = {
    "layer_0": torch.randn(2, 8, 8),
    "layer_1": torch.randn(2, 16, 16)
}

alignment_scores = distiller.compute_attention_alignment_score(teacher_attns, student_attns)
print(f"   ✅ Cosine Similarity: {alignment_scores['cosine_similarity']:.4f}")
print(f"   ✅ L2 Distance: {alignment_scores['l2_distance']:.4f}")
print(f"   ✅ KL Divergence: {alignment_scores['kl_divergence']:.4f}")
print(f"   ✅ Correlation: {alignment_scores['correlation']:.4f}")

# Test Interpretability Score
student_attention = torch.randn(2, 64)
gradients = torch.randn(2, 64)
interp_score = distiller.compute_interpretability_score(student_attention, gradients)
print(f"   ✅ Interpretability Score: {interp_score:.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✅ All tests passed successfully!")
print("="*70)
print("\n📝 Component Summary:")
print("   - AttentionExtractor: Hooks and extraction ✓")
print("   - AttentionMatcher: Resize and normalization ✓")
print("   - AttentionLossComposer: Multi-loss computation ✓")
print("   - AttentionTransferDistiller: Full pipeline ✓")
print("   - Advanced Methods: Rollout, Flow, Dual, Temporal ✓")
print("   - Evaluation Metrics: Alignment and interpretability ✓")
print("\n🚀 Ready to use Advanced Attention Transfer!")
print("="*70)
