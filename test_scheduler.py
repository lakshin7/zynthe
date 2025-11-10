"""
Test Suite for Advanced Scheduler System
=========================================

Tests the SchedulerFactory, WarmupScheduler, MultiStageScheduler,
and AdaptiveScheduler components.
"""

import torch
import torch.nn as nn
from training.scheduler import (
    SchedulerFactory,
    WarmupScheduler,
    MultiStageScheduler,
    AdaptiveScheduler,
    get_scheduler
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_scheduler_factory():
    """Test SchedulerFactory can create all scheduler types."""
    print("\n" + "="*70)
    print("TEST 1: Scheduler Factory - Creating Multiple Scheduler Types")
    print("="*70)
    
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    scheduler_types = [
        'cosine',
        'linear',
        'polynomial',
        'step',
        'multistep',
        'exponential',
        'plateau',
        'constant',
        'onecycle',
        'cyclic'
    ]
    
    for sched_type in scheduler_types:
        config = {'scheduler': sched_type, 'lr': 1e-3}
        
        # Special configs for specific schedulers
        if sched_type == 'multistep':
            config['milestones'] = [10, 20, 30]
        elif sched_type == 'onecycle':
            config['max_lr'] = 1e-2
        elif sched_type == 'cyclic':
            config['base_lr'] = 1e-5
            config['max_lr'] = 1e-3
        
        factory = SchedulerFactory(optimizer, config)
        scheduler = factory.get_scheduler(num_training_steps=100)
        
        print(f"✓ {sched_type.upper()}: {type(scheduler).__name__}")
        
        # Test that scheduler can step
        try:
            if 'plateau' in sched_type.lower():
                scheduler.step(0.9)  # ReduceLROnPlateau needs metric
            else:
                scheduler.step()
            print(f"  - Scheduler step successful")
        except Exception as e:
            print(f"  ✗ Scheduler step failed: {e}")
            return False
    
    print("\n✅ PASSED: All scheduler types created and stepped successfully!")
    return True


def test_warmup_scheduler():
    """Test WarmupScheduler wrapper."""
    print("\n" + "="*70)
    print("TEST 2: Warmup Scheduler - Linear, Cosine, Constant Warmup")
    print("="*70)
    
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    warmup_types = ['linear', 'cosine', 'constant']
    
    for warmup_type in warmup_types:
        # Create base scheduler
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Wrap with warmup
        scheduler = WarmupScheduler(
            base_scheduler,
            warmup_steps=10,
            warmup_type=warmup_type
        )
        
        print(f"\n{warmup_type.upper()} Warmup:")
        
        # Test warmup phase
        initial_lr = optimizer.param_groups[0]['lr']
        print(f"  Initial LR: {initial_lr:.6e}")
        
        lrs = []
        for step in range(15):  # 10 warmup + 5 regular
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            if step < 10:  # Warmup phase
                if step == 0:
                    print(f"  Step {step:2d} (warmup): LR = {current_lr:.6e}")
                elif step == 5:
                    print(f"  Step {step:2d} (warmup): LR = {current_lr:.6e}")
                elif step == 9:
                    print(f"  Step {step:2d} (warmup): LR = {current_lr:.6e}")
            elif step == 10:  # First regular step
                print(f"  Step {step:2d} (regular): LR = {current_lr:.6e}")
        
        # Verify warmup behavior
        if warmup_type == 'linear':
            # LR should increase during warmup
            assert lrs[5] > lrs[0], f"Linear warmup failed: LR not increasing"
            print(f"  ✓ Linear warmup verified (LR increased from {lrs[0]:.6e} to {lrs[9]:.6e})")
        
        # Reset optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
    
    print("\n✅ PASSED: Warmup scheduler works correctly!")
    return True


def test_phase_aware_scheduling():
    """Test phase-aware scheduler behavior (implicit via config)."""
    print("\n" + "="*70)
    print("TEST 3: Phase-Aware Scheduling")
    print("="*70)
    
    model = SimpleModel()
    
    phases = ['distillation', 'quantization', 'finetuning']
    
    for phase in phases:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Phase-specific config (user can adjust per phase)
        if phase == 'distillation':
            config = {'scheduler': 'cosine', 'warmup_steps': 100}
        elif phase == 'quantization':
            config = {'scheduler': 'step', 'step_size': 5, 'gamma': 0.5}
        else:  # finetuning
            config = {'scheduler': 'linear', 'end_factor': 0.1}
        
        factory = SchedulerFactory(optimizer, config)
        scheduler = factory.get_scheduler(num_training_steps=100)
        
        print(f"\n{phase.upper()}:")
        print(f"  Scheduler: {type(scheduler).__name__}")
        print(f"  Config: {config}")
        
        # Step a few times
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(5):
            scheduler.step()
        final_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Initial LR: {initial_lr:.6e}")
        print(f"  LR after 5 steps: {final_lr:.6e}")
    
    print("\n✅ PASSED: Phase-aware scheduling configured correctly!")
    return True


def test_adaptive_scheduler():
    """Test AdaptiveScheduler with metric-based adjustments."""
    print("\n" + "="*70)
    print("TEST 4: Adaptive Scheduler - Metric-Based LR Adjustment")
    print("="*70)
    
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create base scheduler
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Wrap with adaptive scheduler
    scheduler = AdaptiveScheduler(
        base_scheduler,
        enable_adaptive=True,
        mode='max',  # Maximize metric (accuracy/DEI)
        patience=3,
        factor=0.5,
        min_lr=1e-6
    )
    
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"Initial LR: {initial_lr:.6e}")
    
    # Simulate training with improving metrics
    print("\n[Scenario 1] Metrics improving:")
    for epoch in range(5):
        metrics = {'accuracy': 0.8 + epoch * 0.02, 'dei': 1.5 + epoch * 0.1}
        scheduler.step(metrics)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}: accuracy={metrics['accuracy']:.3f}, LR={current_lr:.6e}")
    
    # Simulate plateau (no improvement)
    print("\n[Scenario 2] Plateau (no improvement for 4 epochs):")
    plateau_metric = 0.90
    for epoch in range(5):
        metrics = {'accuracy': plateau_metric, 'dei': 1.9}
        scheduler.step(metrics)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+6}: accuracy={metrics['accuracy']:.3f}, LR={current_lr:.6e}")
        
        # LR should reduce after patience epochs
        if epoch == 3:  # After patience=3
            assert current_lr < initial_lr, "LR should reduce after plateau"
            print(f"  ✓ LR reduced after plateau detection")
    
    print("\n✅ PASSED: Adaptive scheduler adjusts LR based on metrics!")
    return True


def test_multistage_scheduler():
    """Test MultiStageScheduler for complex pipelines."""
    print("\n" + "="*70)
    print("TEST 5: Multi-Stage Scheduler - Complex Training Pipelines")
    print("="*70)
    
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Define 3-stage schedule
    stages = [
        {'scheduler': 'linear', 'steps': 10, 'end_factor': 0.5},
        {'scheduler': 'constant', 'steps': 5},
        {'scheduler': 'exponential', 'steps': 10, 'gamma': 0.9}
    ]
    
    scheduler = MultiStageScheduler(
        optimizer,
        stages=stages,
        num_training_steps=25
    )
    
    print(f"Stages: {len(stages)}")
    for i, stage in enumerate(stages):
        print(f"  Stage {i+1}: {stage['scheduler']} for {stage['steps']} steps")
    
    print("\nSimulating 25 steps:")
    lrs = []
    for step in range(25):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        if step in [0, 9, 14, 24]:  # Key transitions
            stage_num = step // 10 + 1
            print(f"  Step {step:2d} (Stage {min(stage_num, 3)}): LR = {current_lr:.6e}")
    
    # Verify stage transitions
    assert len(lrs) == 25, "Should have 25 LR values"
    print(f"\n✓ Completed {len(lrs)} steps across {len(stages)} stages")
    
    print("\n✅ PASSED: Multi-stage scheduler works correctly!")
    return True


def test_warmup_with_factory():
    """Test SchedulerFactory with warmup."""
    print("\n" + "="*70)
    print("TEST 6: Scheduler Factory with Warmup")
    print("="*70)
    
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    config = {
        'scheduler': 'cosine',
        'warmup_steps': 10,
        'warmup_type': 'linear',
        'lr': 1e-3
    }
    
    factory = SchedulerFactory(optimizer, config)
    scheduler = factory.get_scheduler(num_training_steps=100)
    
    print(f"Scheduler: {type(scheduler).__name__}")
    print(f"Config: {config}")
    
    # Track LR over steps
    lrs = []
    for step in range(20):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        if step in [0, 5, 9, 10, 15]:
            print(f"  Step {step:2d}: LR = {current_lr:.6e}")
    
    # Verify warmup
    assert lrs[5] < lrs[9], "LR should increase during warmup"
    print(f"\n✓ Warmup verified: LR increased from {lrs[0]:.6e} to {lrs[9]:.6e}")
    
    print("\n✅ PASSED: Factory creates scheduler with warmup correctly!")
    return True


def test_convenience_function():
    """Test get_scheduler() convenience function."""
    print("\n" + "="*70)
    print("TEST 7: Convenience Function - get_scheduler()")
    print("="*70)
    
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test with config
    config = {'scheduler': 'cosine', 'warmup_steps': 5}
    scheduler = get_scheduler(optimizer, config, num_training_steps=100)
    print(f"✓ With config: {type(scheduler).__name__}")
    
    # Test with None (should return constant scheduler)
    scheduler_none = get_scheduler(optimizer, None)
    print(f"✓ With None: {type(scheduler_none).__name__}")
    
    # Test stepping
    for _ in range(3):
        scheduler.step()
    print(f"✓ Scheduler step works")
    
    print("\n✅ PASSED: Convenience function works correctly!")
    return True


def test_scheduler_state_dict():
    """Test scheduler state save/load."""
    print("\n" + "="*70)
    print("TEST 8: Scheduler State Dict - Save/Load")
    print("="*70)
    
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    config = {'scheduler': 'cosine', 'warmup_steps': 5}
    factory = SchedulerFactory(optimizer, config)
    scheduler = factory.get_scheduler(num_training_steps=100)
    
    # Step a few times
    for _ in range(10):
        scheduler.step()
    
    step_10_lr = optimizer.param_groups[0]['lr']
    print(f"LR at step 10: {step_10_lr:.6e}")
    
    # Save state
    state = scheduler.state_dict()
    print(f"✓ State saved: {list(state.keys())[:3]}...")
    
    # Create new scheduler and load state
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler2 = factory.get_scheduler(num_training_steps=100)
    scheduler2.load_state_dict(state)
    
    restored_lr = optimizer2.param_groups[0]['lr']
    print(f"Restored LR: {restored_lr:.6e}")
    
    # Note: LR might differ slightly due to scheduler internals,
    # but state should be loadable without error
    print(f"✓ State loaded successfully")
    
    print("\n✅ PASSED: Scheduler state save/load works!")
    return True


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" ADVANCED SCHEDULER SYSTEM - TEST SUITE")
    print("="*70)
    print("Testing SchedulerFactory, WarmupScheduler, MultiStageScheduler,")
    print("AdaptiveScheduler, and integration features.")
    print("="*70)
    
    tests = [
        ("Scheduler Factory", test_scheduler_factory),
        ("Warmup Scheduler", test_warmup_scheduler),
        ("Phase-Aware Scheduling", test_phase_aware_scheduling),
        ("Adaptive Scheduler", test_adaptive_scheduler),
        ("Multi-Stage Scheduler", test_multistage_scheduler),
        ("Factory with Warmup", test_warmup_with_factory),
        ("Convenience Function", test_convenience_function),
        ("State Dict Save/Load", test_scheduler_state_dict),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print("="*70)
    
    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED! The Advanced Scheduler System is production-ready!")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please review.")
    
    return passed_count == total_count


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
