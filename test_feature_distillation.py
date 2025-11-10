"""
Test Suite for Feature Distillation (All 4 Stages)
===================================================

Tests all components:
- FeatureAdapter (dimension alignment)
- FeatureMetrics (L2, CKA, Cosine, Gram, FSP, AB, CRD)
- FeatureLossComposer (multi-metric composition)
- LayerAligner (spatial/channel alignment)
- FeatureDistiller (full integration)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.distillers.feature_distiller import (
    FeatureAdapter,
    FeatureMetrics,
    FeatureLossComposer,
    LayerAligner,
    FeatureDistiller
)


# ============================================================================
# Test Models
# ============================================================================

class SimpleTeacher(nn.Module):
    """Teacher with 3 layers (256 -> 512 -> 1024 channels)."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 256, 3, padding=1)
        self.layer2 = nn.Conv2d(256, 512, 3, padding=1)
        self.layer3 = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleStudent(nn.Module):
    """Student with 3 layers (128 -> 256 -> 512 channels)."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 128, 3, padding=1)
        self.layer2 = nn.Conv2d(128, 256, 3, padding=1)
        self.layer3 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# Test Functions
# ============================================================================

def test_feature_adapter():
    """Test automatic dimension adapter."""
    print("\n" + "="*70)
    print("TEST 1: FeatureAdapter (Auto Dimension Alignment)")
    print("="*70)
    
    # Test 1x1 conv adapter
    adapter = FeatureAdapter(in_channels=128, out_channels=256, adapter_type='1x1conv')
    x = torch.randn(4, 128, 32, 32)
    out = adapter(x)
    
    assert out.shape == (4, 256, 32, 32), f"Expected (4, 256, 32, 32), got {out.shape}"
    print("✓ 1x1 Conv Adapter: (4, 128, 32, 32) -> (4, 256, 32, 32)")
    
    # Test linear adapter
    adapter_linear = FeatureAdapter(in_channels=128, out_channels=256, adapter_type='linear')
    x_flat = torch.randn(4, 128)
    out_linear = adapter_linear(x_flat)
    
    assert out_linear.shape == (4, 256), f"Expected (4, 256), got {out_linear.shape}"
    print("✓ Linear Adapter: (4, 128) -> (4, 256)")
    
    print("\n✅ FeatureAdapter tests passed!")


def test_feature_metrics():
    """Test all feature similarity metrics (Stage 1-4)."""
    print("\n" + "="*70)
    print("TEST 2: FeatureMetrics (All 4 Stages)")
    print("="*70)
    
    # Create dummy features
    f_t = torch.randn(4, 256, 16, 16)
    f_s = torch.randn(4, 256, 16, 16)
    
    # Stage 1: L2 distance
    l2_loss = FeatureMetrics.l2_distance(f_t, f_s)
    print(f"✓ Stage 1 - L2 Loss: {l2_loss.item():.4f}")
    
    # Stage 2: Cosine similarity
    cos_loss = FeatureMetrics.cosine_similarity_loss(f_t, f_s)
    print(f"✓ Stage 2 - Cosine Loss: {cos_loss.item():.4f}")
    
    # Stage 2: Gram matrix
    gram_loss = FeatureMetrics.gram_loss(f_t, f_s)
    print(f"✓ Stage 2 - Gram Loss: {gram_loss.item():.4f}")
    
    # Stage 2: CKA
    cka_loss = FeatureMetrics.centered_kernel_alignment(f_t, f_s)
    print(f"✓ Stage 2 - CKA Loss: {cka_loss.item():.4f}")
    
    # Stage 4: FSP Matrix
    f_t2 = torch.randn(4, 512, 16, 16)
    f_s2 = torch.randn(4, 512, 16, 16)
    fsp_loss = FeatureMetrics.fsp_loss((f_t, f_t2), (f_s, f_s2))
    print(f"✓ Stage 4 - FSP Loss: {fsp_loss.item():.4f}")
    
    # Stage 4: Activation Boundary
    ab_loss = FeatureMetrics.activation_boundary_loss(f_t, f_s)
    print(f"✓ Stage 4 - AB Loss: {ab_loss.item():.4f}")
    
    # Stage 4: Contrastive
    cont_loss = FeatureMetrics.contrastive_loss(f_t, f_s, temperature=0.07)
    print(f"✓ Stage 4 - Contrastive Loss: {cont_loss.item():.4f}")
    
    print("\n✅ All metric tests passed!")


def test_feature_loss_composer():
    """Test multi-metric loss composition."""
    print("\n" + "="*70)
    print("TEST 3: FeatureLossComposer (Multi-Metric)")
    print("="*70)
    
    # Create composer with multiple metrics
    composer = FeatureLossComposer(
        metrics=['l2', 'cka', 'cosine', 'gram'],
        weights={'l2': 1.0, 'cka': 0.5, 'cosine': 0.3, 'gram': 0.2}
    )
    
    f_t = torch.randn(4, 256, 16, 16)
    f_s = torch.randn(4, 256, 16, 16)
    
    total_loss, loss_dict = composer(f_t, f_s)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print("\nIndividual Components:")
    for key, value in loss_dict.items():
        print(f"  - {key}: {value:.4f}")
    
    # Test with attention weighting (Stage 3)
    composer_att = FeatureLossComposer(
        metrics=['l2', 'cosine'],
        use_attention=True
    )
    
    attention_t = torch.sigmoid(torch.randn(4, 1, 16, 16))
    attention_s = torch.sigmoid(torch.randn(4, 1, 16, 16))
    
    att_loss, att_loss_dict = composer_att(f_t, f_s, attention_t, attention_s)
    
    print(f"\n✓ Attention-Weighted Loss: {att_loss.item():.4f}")
    
    print("\n✅ FeatureLossComposer tests passed!")


def test_layer_aligner():
    """Test layer alignment with dimension mismatches."""
    print("\n" + "="*70)
    print("TEST 4: LayerAligner (Dimension Matching)")
    print("="*70)
    
    aligner = LayerAligner(auto_align=True, adapter_type='1x1conv')
    
    # Test 1: Spatial size mismatch
    f_t = torch.randn(4, 256, 32, 32)
    f_s = torch.randn(4, 256, 16, 16)
    
    f_t_aligned, f_s_aligned = aligner.align_features(f_t, f_s, 'test_layer_1')
    
    assert f_t_aligned.shape == f_s_aligned.shape, "Spatial alignment failed"
    print(f"✓ Spatial Alignment: {f_s.shape} -> {f_s_aligned.shape}")
    
    # Test 2: Channel mismatch
    f_t = torch.randn(4, 512, 16, 16)
    f_s = torch.randn(4, 256, 16, 16)
    
    f_t_aligned, f_s_aligned = aligner.align_features(f_t, f_s, 'test_layer_2')
    
    assert f_t_aligned.shape == f_s_aligned.shape, "Channel alignment failed"
    print(f"✓ Channel Alignment: {f_s.shape} -> {f_s_aligned.shape}")
    
    # Test 3: Both mismatches
    f_t = torch.randn(4, 1024, 32, 32)
    f_s = torch.randn(4, 256, 16, 16)
    
    f_t_aligned, f_s_aligned = aligner.align_features(f_t, f_s, 'test_layer_3')
    
    assert f_t_aligned.shape == f_s_aligned.shape, "Full alignment failed"
    print(f"✓ Full Alignment: {f_s.shape} -> {f_s_aligned.shape}")
    
    print("\n✅ LayerAligner tests passed!")


def test_feature_distiller():
    """Test full FeatureDistiller integration."""
    print("\n" + "="*70)
    print("TEST 5: FeatureDistiller (Full Integration)")
    print("="*70)
    
    # Create models
    teacher = SimpleTeacher()
    student = SimpleStudent()
    
    # Configuration
    config = {
        'feature_distillation': {
            'enabled': True,
            'layers': [
                {'teacher': 'layer1', 'student': 'layer1', 'weight': 0.5},
                {'teacher': 'layer2', 'student': 'layer2', 'weight': 1.0},
                {'teacher': 'layer3', 'student': 'layer3', 'weight': 1.5},
            ],
            'metrics': ['l2', 'cka', 'cosine'],
            'metric_weights': {'l2': 1.0, 'cka': 0.5, 'cosine': 0.3},
            'auto_align': True,
            'adapter_type': '1x1conv',
            'fsp_pairs': [[0, 1], [1, 2]],
            'contrastive_temperature': 0.07
        }
    }
    
    # Create distiller
    distiller = FeatureDistiller(teacher, student, config=config)
    print(f"✓ Distiller initialized on device: {distiller.device}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32).to(distiller.device)
    student_out, teacher_out, t_feat, s_feat = distiller.forward(
        dummy_input, return_features=True
    )
    
    print(f"✓ Forward pass successful")
    print(f"  - Student output: {student_out.shape}")
    print(f"  - Teacher output: {teacher_out.shape}")
    print(f"  - Teacher features: {len(t_feat)} layers")
    print(f"  - Student features: {len(s_feat)} layers")
    
    # Test loss computation
    dummy_targets = torch.randint(0, 10, (4,)).to(distiller.device)
    
    # Wrap outputs to have logits attribute (like transformer models)
    class OutputWrapper:
        def __init__(self, logits):
            self.logits = logits
    
    student_out_wrapped = OutputWrapper(student_out)
    
    total_loss, loss_dict = distiller.compute_loss(
        student_out_wrapped,
        teacher_out,
        dummy_targets,
        student_features=s_feat,
        teacher_features=t_feat
    )
    
    print(f"\n✓ Loss computation successful")
    print(f"  - Total loss: {total_loss.item():.4f}")
    print("\n  Individual losses:")
    for key, value in loss_dict.items():
        print(f"    - {key}: {value:.4f}")
    
    # Test alignment metrics
    alignment_metrics = distiller.compute_feature_alignment_metrics(t_feat, s_feat)
    
    print(f"\n✓ Feature alignment metrics:")
    for key, value in alignment_metrics.items():
        print(f"    - {key}: {value:.4f}")
    
    # Test training step
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    distiller.optimizer = optimizer
    
    batch = (dummy_input, dummy_targets)
    loss_dict_train = distiller.training_step(batch)
    
    print(f"\n✓ Training step successful")
    print(f"  - Global step: {distiller.global_step}")
    
    print("\n✅ Full FeatureDistiller integration test passed!")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("FEATURE DISTILLATION TEST SUITE - ALL 4 STAGES")
    print("="*70)
    print("Testing: Adapter, Metrics, Composer, Aligner, Distiller")
    print("="*70)
    
    try:
        test_feature_adapter()
        test_feature_metrics()
        test_feature_loss_composer()
        test_layer_aligner()
        test_feature_distiller()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED! Feature Distillation is fully functional.")
        print("="*70)
        print("\nSupported Features:")
        print("  ✓ Stage 1: Vanilla L2 feature regression")
        print("  ✓ Stage 2: CKA, Cosine, Gram matrix matching")
        print("  ✓ Stage 3: Attention-augmented features")
        print("  ✓ Stage 4: FSP, AB, CRD (advanced paradigms)")
        print("  ✓ Auto dimension alignment (1x1 conv adapters)")
        print("  ✓ Multi-metric loss composition")
        print("  ✓ Feature alignment evaluation")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    run_all_tests()
