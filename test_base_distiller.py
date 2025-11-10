"""
Test script to verify BaseDistiller is working correctly.
"""

import torch
import torch.nn as nn
from core.distillers.base_distiller import BaseDistiller


# Simple test models
class SimpleTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


class SimpleStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


# Custom distiller that implements compute_loss
class SimpleDistiller(BaseDistiller):
    def compute_loss(self, student_outputs, teacher_outputs, targets=None, **kwargs):
        """Simple MSE loss between student and teacher outputs."""
        loss_kd = nn.functional.mse_loss(student_outputs, teacher_outputs)
        
        loss_dict = {'kd_loss': loss_kd.item(), 'total': loss_kd.item()}
        
        if targets is not None:
            loss_ce = nn.functional.cross_entropy(student_outputs, targets)
            total_loss = loss_ce + loss_kd
            loss_dict['ce_loss'] = loss_ce.item()
            loss_dict['total'] = total_loss.item()
            return total_loss, loss_dict
        
        return loss_kd, loss_dict


def test_base_distiller():
    print("=" * 70)
    print("Testing BaseDistiller")
    print("=" * 70)
    
    # Create models
    print("\n1. Creating teacher and student models...")
    teacher = SimpleTeacher()
    student = SimpleStudent()
    print("   ✓ Models created")
    
    # Create distiller
    print("\n2. Initializing distiller...")
    config = {
        'losses': [
            {'type': 'supervised', 'weight': 1.0},
            {'type': 'kd', 'weight': 0.5}
        ]
    }
    distiller = SimpleDistiller(teacher, student, config=config)
    print(f"   ✓ Distiller initialized on device: {distiller.device}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_input = torch.randn(4, 10).to(distiller.device)
    student_out, teacher_out = distiller.forward(dummy_input)
    print(f"   ✓ Forward pass successful")
    print(f"   - Student output shape: {student_out.shape}")
    print(f"   - Teacher output shape: {teacher_out.shape}")
    
    # Test with feature extraction
    print("\n4. Testing forward pass with feature extraction...")
    student_out, teacher_out, t_feat, s_feat = distiller.forward(
        dummy_input, return_features=True
    )
    print(f"   ✓ Feature extraction successful")
    print(f"   - Teacher features: {len(t_feat)} layers")
    print(f"   - Student features: {len(s_feat)} layers")
    
    # Test compute_loss
    print("\n5. Testing loss computation...")
    dummy_targets = torch.randint(0, 5, (4,)).to(distiller.device)
    total_loss, loss_dict = distiller.compute_loss(
        student_out, teacher_out, dummy_targets
    )
    print(f"   ✓ Loss computation successful")
    print(f"   - Total loss: {total_loss.item():.4f}")
    for name, value in loss_dict.items():
        print(f"   - {name}: {value:.4f}")
    
    # Test training step
    print("\n6. Testing training step...")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    distiller.optimizer = optimizer
    
    batch = (dummy_input, dummy_targets)
    loss_dict = distiller.training_step(batch)
    print(f"   ✓ Training step successful")
    print(f"   - Global step: {distiller.global_step}")
    for name, value in loss_dict.items():
        print(f"   - {name}: {value:.4f}")
    
    # Test evaluation step
    print("\n7. Testing evaluation step...")
    loss_dict = distiller.evaluation_step(batch)
    print(f"   ✓ Evaluation step successful")
    for name, value in loss_dict.items():
        print(f"   - {name}: {value:.4f}")
    
    # Test metrics logging
    print("\n8. Testing metrics logging...")
    distiller.log_metrics({'accuracy': 0.85, 'f1': 0.82}, phase='train')
    distiller.log_metrics({'accuracy': 0.87, 'f1': 0.84}, phase='val')
    summary = distiller.get_metrics_summary()
    print(f"   ✓ Metrics logging successful")
    print(f"   - Tracked metrics: {list(distiller.metrics.keys())}")
    
    # Test optimizer initialization
    print("\n9. Testing optimizer initialization...")
    config_opt = {
        'total_steps': 1000  # Pass through kwargs to scheduler
    }
    opt, sched = distiller._init_optimizers(
        lr=2e-5,
        optimizer_type='adamw',
        scheduler_type='cosine',
        **config_opt
    )
    print(f"   ✓ Optimizer initialization successful")
    print(f"   - Optimizer type: {type(opt).__name__}")
    print(f"   - Scheduler type: {type(sched).__name__ if sched else 'None'}")
    
    # Test teacher freezing
    print("\n10. Testing teacher freeze/unfreeze...")
    distiller.freeze_teacher()
    frozen_params = sum(1 for p in teacher.parameters() if not p.requires_grad)
    print(f"   ✓ Teacher frozen: {frozen_params}/{sum(1 for _ in teacher.parameters())} params")
    
    distiller.unfreeze_teacher()
    unfrozen_params = sum(1 for p in teacher.parameters() if p.requires_grad)
    print(f"   ✓ Teacher unfrozen: {unfrozen_params}/{sum(1 for _ in teacher.parameters())} params")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed! BaseDistiller is working correctly.")
    print("=" * 70)


if __name__ == '__main__':
    test_base_distiller()
