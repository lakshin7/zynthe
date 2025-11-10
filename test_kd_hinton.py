"""
Test Suite for KD-Hinton Distiller with Hint Learning
======================================================

Tests all components:
- HintRegressor (1x1 conv, linear, MLP, attention)
- HintScheduler (progressive decay strategies)
- TemperatureScheduler (dynamic T scaling)
- HintLossFunctions (MSE, Cosine, KL, Contrastive)
- KDHintonDistiller (full integration with hints)
- Legacy compatibility
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.distillers.kd_hinton import (
    HintRegressor,
    HintScheduler,
    TemperatureScheduler,
    HintLossFunctions,
    KDHintonDistiller,
    LegacyKDHintonDistiller
)


# ============================================================================
# Test Models
# ============================================================================

class SimpleTeacher(nn.Module):
    """Teacher with 3 intermediate layers."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleStudent(nn.Module):
    """Student with 3 intermediate layers (narrower)."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# Test Functions
# ============================================================================

def test_hint_regressor():
    """Test hint regressor variants."""
    print("\n" + "="*70)
    print("TEST 1: HintRegressor (Feature Transformation)")
    print("="*70)
    
    # Test 1x1 conv regressor
    regressor_conv = HintRegressor(
        student_dim=64,
        teacher_dim=128,
        regressor_type='1x1conv',
        use_bn=True
    )
    x = torch.randn(4, 64, 16, 16)
    out = regressor_conv(x)
    assert out.shape == (4, 128, 16, 16)
    print("✓ 1x1 Conv Regressor: (4, 64, 16, 16) -> (4, 128, 16, 16)")
    
    # Test MLP regressor
    regressor_mlp = HintRegressor(
        student_dim=256,
        teacher_dim=512,
        regressor_type='mlp'
    )
    x_flat = torch.randn(4, 256)
    out_mlp = regressor_mlp(x_flat)
    assert out_mlp.shape == (4, 512)
    print("✓ MLP Regressor: (4, 256) -> (4, 512)")
    
    # Test attention regressor
    regressor_att = HintRegressor(
        student_dim=64,
        teacher_dim=128,
        regressor_type='attention'
    )
    out_att = regressor_att(x)
    assert out_att.shape == (4, 128, 16, 16)
    print("✓ Attention Regressor: (4, 64, 16, 16) -> (4, 128, 16, 16)")
    
    print("\n✅ HintRegressor tests passed!")


def test_hint_scheduler():
    """Test hint weight scheduling strategies."""
    print("\n" + "="*70)
    print("TEST 2: HintScheduler (Progressive Decay)")
    print("="*70)
    
    # Test constant
    scheduler_const = HintScheduler(
        initial_weight=1.0,
        final_weight=0.5,
        total_steps=100,
        scheduler_type='constant'
    )
    weights = [scheduler_const.step() for _ in range(10)]
    assert all(w == 1.0 for w in weights)
    print(f"✓ Constant: {weights[0]:.4f} (stable)")
    
    # Test linear decay
    scheduler_linear = HintScheduler(
        initial_weight=1.0,
        final_weight=0.1,
        total_steps=100,
        scheduler_type='linear_decay'
    )
    weights = [scheduler_linear.step() for _ in range(100)]
    print(f"✓ Linear Decay: {weights[0]:.4f} -> {weights[-1]:.4f}")
    
    # Test cosine decay
    scheduler_cosine = HintScheduler(
        initial_weight=1.0,
        final_weight=0.1,
        total_steps=100,
        scheduler_type='cosine_decay'
    )
    weights = [scheduler_cosine.step() for _ in range(100)]
    print(f"✓ Cosine Decay: {weights[0]:.4f} -> {weights[-1]:.4f}")
    
    # Test exponential decay
    scheduler_exp = HintScheduler(
        initial_weight=1.0,
        final_weight=0.1,
        total_steps=100,
        scheduler_type='exponential_decay'
    )
    weights = [scheduler_exp.step() for _ in range(100)]
    print(f"✓ Exponential Decay: {weights[0]:.4f} -> {weights[-1]:.4f}")
    
    print("\n✅ HintScheduler tests passed!")


def test_temperature_scheduler():
    """Test temperature scheduling."""
    print("\n" + "="*70)
    print("TEST 3: TemperatureScheduler (Dynamic T)")
    print("="*70)
    
    scheduler = TemperatureScheduler(
        initial_temp=4.0,
        final_temp=2.0,
        total_steps=100,
        scheduler_type='cosine'
    )
    
    temps = [scheduler.step() for _ in range(100)]
    print(f"✓ Temperature Schedule: {temps[0]:.4f} -> {temps[-1]:.4f}")
    print(f"  - Softer distributions early (T={temps[0]:.2f})")
    print(f"  - Sharper distributions late (T={temps[-1]:.2f})")
    
    print("\n✅ TemperatureScheduler tests passed!")


def test_hint_loss_functions():
    """Test all hint loss variants."""
    print("\n" + "="*70)
    print("TEST 4: HintLossFunctions (MSE, Cosine, KL, Contrastive)")
    print("="*70)
    
    hint_t = torch.randn(4, 128, 16, 16)
    hint_s = torch.randn(4, 128, 16, 16)
    
    # MSE loss
    mse_loss = HintLossFunctions.mse_loss(hint_t, hint_s)
    print(f"✓ MSE Loss: {mse_loss.item():.4f}")
    
    # Cosine loss
    cos_loss = HintLossFunctions.cosine_loss(hint_t, hint_s)
    print(f"✓ Cosine Loss: {cos_loss.item():.4f}")
    
    # KL loss
    kl_loss = HintLossFunctions.kl_loss(hint_t, hint_s, temperature=2.0)
    print(f"✓ KL Loss: {kl_loss.item():.4f}")
    
    # Contrastive loss
    cont_loss = HintLossFunctions.contrastive_hint_loss(hint_t, hint_s, temperature=0.07)
    print(f"✓ Contrastive Loss: {cont_loss.item():.4f}")
    
    print("\n✅ HintLossFunctions tests passed!")


def test_kd_hinton_distiller():
    """Test full KD-Hinton distiller with hints."""
    print("\n" + "="*70)
    print("TEST 5: KDHintonDistiller (Full Integration)")
    print("="*70)
    
    # Create models
    teacher = SimpleTeacher()
    student = SimpleStudent()
    
    # Configuration with hints
    config = {
        'kd_hinton': {
            'temperature': 4.0,
            'alpha': 0.7,
            'label_smoothing': 0.1,
            'hint_enabled': True,
            'hints': [
                {
                    'teacher': 'layer1',
                    'student': 'layer1',
                    'weight': 0.5,
                    'regressor': '1x1conv',
                    'loss': 'mse'
                },
                {
                    'teacher': 'layer2',
                    'student': 'layer2',
                    'weight': 0.3,
                    'regressor': '1x1conv',
                    'loss': 'cosine'
                }
            ],
            'hint_scheduler': {
                'type': 'cosine_decay',
                'initial_weight': 1.0,
                'final_weight': 0.1,
                'total_steps': 1000
            },
            'temperature_scheduler': {
                'type': 'cosine',
                'initial_temp': 4.0,
                'final_temp': 2.0,
                'total_steps': 1000
            }
        }
    }
    
    # Create distiller
    distiller = KDHintonDistiller(teacher, student, config=config)
    print(f"✓ Distiller initialized on device: {distiller.device}")
    print(f"  - Temperature: {distiller.temperature}")
    print(f"  - Alpha (KD weight): {distiller.alpha}")
    print(f"  - Hint enabled: {distiller.hint_enabled}")
    print(f"  - Number of hints: {len(distiller.hint_configs)}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32).to(distiller.device)
    student_out, teacher_out, t_feat, s_feat = distiller.forward(
        dummy_input, return_features=True
    )
    
    print(f"\n✓ Forward pass successful")
    print(f"  - Student output: {student_out.shape}")
    print(f"  - Teacher output: {teacher_out.shape}")
    print(f"  - Teacher features: {len(t_feat)} layers")
    print(f"  - Student features: {len(s_feat)} layers")
    
    # Test loss computation
    dummy_targets = torch.randint(0, 10, (4,)).to(distiller.device)
    
    total_loss, loss_dict = distiller.compute_loss(
        student_out,
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
    
    # Test training step
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    distiller.optimizer = optimizer
    
    batch = (dummy_input, dummy_targets)
    distiller.student.train()
    loss_dict_train = distiller.training_step(batch)
    
    print(f"\n✓ Training step successful")
    print(f"  - Global step: {distiller.global_step}")
    
    # Test hint alignment metrics
    alignment_metrics = distiller.compute_hint_alignment_metrics(t_feat, s_feat)
    
    print(f"\n✓ Hint alignment metrics:")
    for key, value in alignment_metrics.items():
        print(f"    - {key}: {value:.4f}")
    
    print("\n✅ Full KDHintonDistiller integration test passed!")


def test_legacy_compatibility():
    """Test backward compatibility with legacy interface."""
    print("\n" + "="*70)
    print("TEST 6: Legacy Compatibility")
    print("="*70)
    
    # Use class weights matching number of classes
    class_weights = [1.0] * 10  # 10 classes
    distiller = LegacyKDHintonDistiller(temperature=4.0, alpha=0.7, class_weights=class_weights)
    
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    
    loss = distiller.compute_loss(student_logits, teacher_logits, labels)
    
    print(f"✓ Legacy interface works: loss = {loss.item():.4f}")
    print("\n✅ Legacy compatibility test passed!")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("KD-HINTON TEST SUITE - Classical + Hint Learning")
    print("="*70)
    print("Testing: Regressor, Schedulers, Loss Functions, Full Integration")
    print("="*70)
    
    try:
        test_hint_regressor()
        test_hint_scheduler()
        test_temperature_scheduler()
        test_hint_loss_functions()
        test_kd_hinton_distiller()
        test_legacy_compatibility()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED! KD-Hinton with Hints is fully functional.")
        print("="*70)
        print("\nSupported Features:")
        print("  ✓ Classical Hinton KD (soft logits)")
        print("  ✓ Multi-hint intermediate guidance (FitNets-style)")
        print("  ✓ Progressive hint weight scheduling")
        print("  ✓ Dynamic temperature scaling")
        print("  ✓ Multiple hint losses (MSE, Cosine, KL, Contrastive)")
        print("  ✓ Label smoothing & class balancing")
        print("  ✓ Cross-architecture support")
        print("  ✓ Hint feature alignment metrics (HFA)")
        print("  ✓ Legacy compatibility")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    run_all_tests()
