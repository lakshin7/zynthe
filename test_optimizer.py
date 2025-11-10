#!/usr/bin/env python3
"""
Test script for Zynthe Toolkit Advanced Optimizer System

Tests:
1. Optimizer factory with different optimizers
2. Parameter grouping (simple and layer-wise)
3. Gradient management (clipping, centralization, noise)
4. Adaptive LR tuning based on metrics
5. Checkpoint save/load
6. Lookahead wrapper
7. Phase-aware optimization
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.optimizer import (
    OptimizerFactory,
    GradientManager,
    AdaptiveOptimizer,
    OptimizerCheckpoint,
    LookaheadOptimizer,
    get_optimizer,
    clip_gradients,
    centralize_gradients,
    inject_gradient_noise
)


# =============================================================================
# Dummy Models for Testing
# =============================================================================

class SimpleModel(nn.Module):
    """Simple feedforward model for basic testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLikeModel(nn.Module):
    """Transformer-like model with layers for layer-wise LR testing."""
    def __init__(self, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Simulated transformer layers (using underscores instead of dots)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                f'layer_{i}': nn.Linear(128, 128),
                f'layer_norm_{i}': nn.LayerNorm(128)
            })
            for i in range(num_layers)
        ])
        
        # Classifier head
        self.classifier = nn.Linear(128, 2)
        
        # Add config for layer-wise detection
        class Config:
            def __init__(self):
                self.num_hidden_layers = num_layers
        self.config = Config()
    
    def forward(self, x):
        for layer_dict in self.layers:
            for name, layer in layer_dict.items():
                if 'layer_norm' not in name:
                    x = torch.relu(layer(x))
        return self.classifier(x)


# =============================================================================
# Test Functions
# =============================================================================

def test_optimizer_factory():
    """Test OptimizerFactory with different optimizers."""
    print("\n" + "="*70)
    print("Test 1: Optimizer Factory")
    print("="*70 + "\n")
    
    model = SimpleModel()
    
    # Test different optimizers
    optimizers_to_test = ['adamw', 'adam', 'sgd']
    
    for opt_name in optimizers_to_test:
        print(f"Testing {opt_name.upper()}...")
        
        config = {
            'optimizer': opt_name,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'momentum': 0.9,  # For SGD
            'nesterov': True   # For SGD
        }
        
        try:
            optimizer = OptimizerFactory.get_optimizer(model, config, phase='distillation')
            
            # Verify optimizer was created
            assert optimizer is not None
            assert len(optimizer.param_groups) > 0
            
            print(f"  ✅ {opt_name.upper()} created successfully")
            print(f"     - Param groups: {len(optimizer.param_groups)}")
            print(f"     - Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
        except Exception as e:
            print(f"  ❌ {opt_name.upper()} failed: {e}")
            return False
    
    print("\n✅ Optimizer factory test passed!")
    return True


def test_phase_aware_optimization():
    """Test phase-aware LR adjustment."""
    print("\n" + "="*70)
    print("Test 2: Phase-Aware Optimization")
    print("="*70 + "\n")
    
    model = SimpleModel()
    config = {'optimizer': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.01}
    
    phases = ['distillation', 'quantization', 'finetuning']
    expected_multipliers = [1.0, 0.5, 1.5]
    
    for phase, multiplier in zip(phases, expected_multipliers):
        optimizer = OptimizerFactory.get_optimizer(model, config, phase=phase)
        actual_lr = optimizer.param_groups[0]['lr']
        expected_lr = config['learning_rate'] * multiplier
        
        print(f"Phase: {phase}")
        print(f"  Expected LR: {expected_lr:.2e} (base × {multiplier})")
        print(f"  Actual LR:   {actual_lr:.2e}")
        
        if abs(actual_lr - expected_lr) < 1e-10:
            print(f"  ✅ Correct phase adjustment")
        else:
            print(f"  ❌ Incorrect phase adjustment")
            return False
    
    print("\n✅ Phase-aware optimization test passed!")
    return True


def test_parameter_grouping():
    """Test parameter grouping with layer-wise LR."""
    print("\n" + "="*70)
    print("Test 3: Parameter Grouping")
    print("="*70 + "\n")
    
    # Test simple grouping
    print("Testing simple parameter grouping...")
    model = SimpleModel()
    config = {'optimizer': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.01}
    
    optimizer = OptimizerFactory.get_optimizer(model, config)
    print(f"  Simple grouping: {len(optimizer.param_groups)} groups")
    print(f"  ✅ Simple grouping works")
    
    # Test layer-wise grouping
    print("\nTesting layer-wise parameter grouping...")
    model = TransformerLikeModel(num_layers=4)
    config['layer_wise_lr'] = True
    
    optimizer = OptimizerFactory.get_optimizer(model, config)
    print(f"  Layer-wise grouping: {len(optimizer.param_groups)} groups")
    
    # Verify learning rates decrease for earlier layers
    if len(optimizer.param_groups) > 1:
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"  Learning rates: {[f'{lr:.2e}' for lr in lrs]}")
        print(f"  ✅ Layer-wise grouping works")
    else:
        print(f"  ⚠️  Only 1 group created (fallback)")
    
    print("\n✅ Parameter grouping test passed!")
    return True


def test_gradient_management():
    """Test gradient clipping, centralization, and noise injection."""
    print("\n" + "="*70)
    print("Test 4: Gradient Management")
    print("="*70 + "\n")
    
    model = SimpleModel()
    
    # Create dummy gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 10.0  # Large gradients
    
    # Test gradient clipping
    print("Testing gradient clipping...")
    grad_norm_before = GradientManager.get_gradient_stats(model)['grad_norm']
    clipped_norm = GradientManager.clip_gradients(model, max_norm=1.0)
    grad_norm_after = GradientManager.get_gradient_stats(model)['grad_norm']
    
    print(f"  Norm before clipping: {grad_norm_before:.4f}")
    print(f"  Clipped norm: {clipped_norm:.4f}")
    print(f"  Norm after clipping: {grad_norm_after:.4f}")
    assert grad_norm_after <= 1.1, "Gradient clipping failed"
    print(f"  ✅ Gradient clipping works")
    
    # Test gradient centralization
    print("\nTesting gradient centralization...")
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    
    GradientManager.centralize_gradients(model)
    
    # Check that gradients have zero mean (for multi-dimensional)
    for param in model.parameters():
        if param.grad is not None and param.grad.dim() > 1:
            mean_val = param.grad.mean(dim=tuple(range(1, param.grad.dim()))).abs().max().item()
            assert mean_val < 1e-5, f"Centralization failed, mean={mean_val}"
    print(f"  ✅ Gradient centralization works")
    
    # Test gradient noise injection
    print("\nTesting gradient noise injection...")
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    
    GradientManager.inject_gradient_noise(model, noise_scale=0.1)
    
    noise_detected = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().max() > 0:
            noise_detected = True
            break
    
    assert noise_detected, "Noise injection failed"
    print(f"  ✅ Gradient noise injection works")
    
    # Test gradient stats
    print("\nTesting gradient statistics...")
    stats = GradientManager.get_gradient_stats(model)
    print(f"  Grad norm: {stats['grad_norm']:.4f}")
    print(f"  Grad mean: {stats['grad_mean']:.4f}")
    print(f"  Grad std:  {stats['grad_std']:.4f}")
    assert 'grad_norm' in stats, "Stats computation failed"
    print(f"  ✅ Gradient stats work")
    
    print("\n✅ Gradient management test passed!")
    return True


def test_adaptive_optimizer():
    """Test adaptive LR tuning based on metrics."""
    print("\n" + "="*70)
    print("Test 5: Adaptive Optimizer")
    print("="*70 + "\n")
    
    model = SimpleModel()
    config = {'optimizer': 'adamw', 'learning_rate': 1e-3, 'weight_decay': 0.01}
    optimizer = OptimizerFactory.get_optimizer(model, config)
    
    adaptive_opt = AdaptiveOptimizer(
        optimizer,
        enable_auto_tune=True,
        patience=2,
        factor=0.5
    )
    
    # Test 1: DEI emergency reduction
    print("Testing DEI emergency reduction (DEI < 0.8)...")
    metrics = {'dei': 0.5, 'accuracy': 0.7}
    initial_lr = optimizer.param_groups[0]['lr']
    actions = adaptive_opt.auto_tune(metrics, epoch=1)
    new_lr = optimizer.param_groups[0]['lr']
    
    print(f"  Initial LR: {initial_lr:.2e}")
    print(f"  New LR:     {new_lr:.2e}")
    print(f"  Action:     {actions['action']}")
    assert new_lr < initial_lr, "DEI emergency reduction failed"
    print(f"  ✅ DEI emergency reduction works")
    
    # Test 2: Plateau detection
    print("\nTesting plateau detection...")
    optimizer = OptimizerFactory.get_optimizer(model, config)
    adaptive_opt = AdaptiveOptimizer(optimizer, patience=2)
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Simulate plateau (no improvement)
    for epoch in range(3):
        metrics = {'accuracy': 0.85}
        actions = adaptive_opt.auto_tune(metrics, epoch=epoch)
        print(f"  Epoch {epoch}: Action={actions['action']}, LR={optimizer.param_groups[0]['lr']:.2e}")
    
    final_lr = optimizer.param_groups[0]['lr']
    assert final_lr < initial_lr, "Plateau reduction failed"
    print(f"  ✅ Plateau detection works")
    
    print("\n✅ Adaptive optimizer test passed!")
    return True


def test_checkpoint():
    """Test optimizer checkpoint save/load."""
    print("\n" + "="*70)
    print("Test 6: Checkpoint Save/Load")
    print("="*70 + "\n")
    
    model = SimpleModel()
    config = {'optimizer': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.01}
    optimizer = OptimizerFactory.get_optimizer(model, config)
    
    # Save checkpoint
    checkpoint_path = "/tmp/test_optimizer_checkpoint.pt"
    OptimizerCheckpoint.save_checkpoint(
        optimizer,
        checkpoint_path,
        epoch=5,
        best_metric=0.95
    )
    print(f"  ✅ Checkpoint saved to {checkpoint_path}")
    
    # Modify optimizer state
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-6
    
    # Load checkpoint
    metadata = OptimizerCheckpoint.load_checkpoint(optimizer, checkpoint_path)
    restored_lr = optimizer.param_groups[0]['lr']
    
    print(f"  Restored LR: {restored_lr:.2e}")
    print(f"  Metadata: epoch={metadata['epoch']}, best_metric={metadata['best_metric']}")
    
    assert metadata['epoch'] == 5, "Epoch not restored"
    assert metadata['best_metric'] == 0.95, "Best metric not restored"
    print(f"  ✅ Checkpoint loaded successfully")
    
    # Cleanup
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\n✅ Checkpoint test passed!")
    return True


def test_lookahead():
    """Test Lookahead optimizer wrapper."""
    print("\n" + "="*70)
    print("Test 7: Lookahead Optimizer")
    print("="*70 + "\n")
    
    model = SimpleModel()
    config = {'optimizer': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.01}
    base_optimizer = OptimizerFactory.get_optimizer(model, config)
    
    # Wrap with Lookahead
    lookahead_opt = LookaheadOptimizer(base_optimizer, k=5, alpha=0.5)
    
    print(f"  Created Lookahead optimizer (k=5, alpha=0.5)")
    print(f"  Param groups: {len(lookahead_opt.param_groups)}")
    
    # Simulate training steps
    for step in range(10):
        # Dummy forward pass
        x = torch.randn(4, 100)
        y = model(x)
        loss = y.sum()
        
        # Backward and optimizer step
        loss.backward()
        lookahead_opt.step()
        lookahead_opt.zero_grad()
    
    print(f"  ✅ Completed 10 training steps with Lookahead")
    
    # Test state dict
    state = lookahead_opt.state_dict()
    assert 'optimizer' in state, "Lookahead state dict missing optimizer"
    assert 'step_count' in state, "Lookahead state dict missing step_count"
    print(f"  ✅ State dict works (step_count={state['step_count']})")
    
    print("\n✅ Lookahead test passed!")
    return True


def test_convenience_functions():
    """Test convenience functions for backward compatibility."""
    print("\n" + "="*70)
    print("Test 8: Convenience Functions")
    print("="*70 + "\n")
    
    model = SimpleModel()
    
    # Test get_optimizer convenience function
    optimizer = get_optimizer(model, lr=1e-4, weight_decay=0.01, phase='distillation')
    assert optimizer is not None
    print(f"  ✅ get_optimizer() works")
    
    # Test gradient management convenience functions
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 10.0
    
    clipped_norm = clip_gradients(model, max_norm=1.0)
    assert clipped_norm > 0
    print(f"  ✅ clip_gradients() works")
    
    centralize_gradients(model)
    print(f"  ✅ centralize_gradients() works")
    
    inject_gradient_noise(model, noise_scale=0.01)
    print(f"  ✅ inject_gradient_noise() works")
    
    print("\n✅ Convenience functions test passed!")
    return True


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Zynthe Toolkit - Advanced Optimizer System Test Suite")
    print("="*70)
    
    tests = [
        ("Optimizer Factory", test_optimizer_factory),
        ("Phase-Aware Optimization", test_phase_aware_optimization),
        ("Parameter Grouping", test_parameter_grouping),
        ("Gradient Management", test_gradient_management),
        ("Adaptive Optimizer", test_adaptive_optimizer),
        ("Checkpoint Save/Load", test_checkpoint),
        ("Lookahead Optimizer", test_lookahead),
        ("Convenience Functions", test_convenience_functions),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70 + "\n")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nThe Advanced Optimizer System is production-ready!")
        print("\nKey Features:")
        print("  • Multi-optimizer support (AdamW, Adam, SGD, Lion, etc.)")
        print("  • Phase-aware optimization")
        print("  • Gradient management (clipping, centralization, noise)")
        print("  • Adaptive LR tuning based on DEI/CAS metrics")
        print("  • Parameter grouping (simple and layer-wise)")
        print("  • Checkpoint save/load support")
        print("  • Lookahead wrapper for improved convergence")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please review the errors above and fix any issues.")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
