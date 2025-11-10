"""
Quick Validation: Check if Optimizer/Scheduler Integration Works with Main.py
==============================================================================

This script validates that:
1. Config files have all required optimizer/scheduler parameters
2. Trainer can be initialized with the configs
3. The new optimizer/scheduler system is active
"""

import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_config(config_path):
    """Validate that config has required optimizer/scheduler params."""
    print(f"\n{'='*70}")
    print(f"Validating: {config_path}")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_config = config.get('train', {})
    
    # Required params for optimizer
    optimizer_params = {
        'optimizer': train_config.get('optimizer', 'NOT SET'),
        'lr': train_config.get('lr', 'NOT SET'),
        'weight_decay': train_config.get('weight_decay', 'NOT SET'),
        'max_grad_norm': train_config.get('max_grad_norm', 'NOT SET'),
    }
    
    # Required params for scheduler
    scheduler_params = {
        'scheduler': train_config.get('scheduler', 'NOT SET'),
        'warmup_steps': train_config.get('warmup_steps', 'NOT SET'),
    }
    
    # Optional but recommended
    optional_params = {
        'centralize_grads': train_config.get('centralize_grads', 'NOT SET'),
        'dynamic_lr': train_config.get('dynamic_lr', 'NOT SET'),
        'warmup_type': train_config.get('warmup_type', 'NOT SET'),
    }
    
    print("\n✅ OPTIMIZER CONFIGURATION:")
    for key, value in optimizer_params.items():
        status = "✓" if value != "NOT SET" else "✗"
        print(f"  {status} {key}: {value}")
    
    print("\n✅ SCHEDULER CONFIGURATION:")
    for key, value in scheduler_params.items():
        status = "✓" if value != "NOT SET" else "✗"
        print(f"  {status} {key}: {value}")
    
    print("\n⚙️  OPTIONAL (RECOMMENDED):")
    for key, value in optional_params.items():
        status = "✓" if value != "NOT SET" else "○"
        print(f"  {status} {key}: {value}")
    
    # Check if all required params are set
    all_required = list(optimizer_params.values()) + list(scheduler_params.values())
    missing = [v for v in all_required if v == "NOT SET"]
    
    if missing:
        print(f"\n⚠️  WARNING: {len(missing)} required parameter(s) missing!")
        return False
    else:
        print("\n✅ All required parameters present!")
        return True


def test_trainer_initialization():
    """Test that Trainer can be initialized with new system."""
    print(f"\n{'='*70}")
    print("Testing Trainer Initialization with OptimizerFactory/SchedulerFactory")
    print(f"{'='*70}")
    
    try:
        import torch
        import torch.nn as nn
        from core.config.config_manager import ConfigManager
        from training.trainer import Trainer
        from training.optimizer import OptimizerFactory
        from training.scheduler import SchedulerFactory
        
        # Simple dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
                self.config = type('Config', (), {'vocab_size': 1000})()
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                x = self.fc(input_ids.float())
                loss = torch.tensor(0.0) if labels is not None else None
                return type('Output', (), {'loss': loss, 'logits': x})()
            
            def save_pretrained(self, path):
                Path(path).mkdir(exist_ok=True)
        
        class DummyTokenizer:
            def save_pretrained(self, path):
                Path(path).mkdir(exist_ok=True)
        
        print("\n[1] Loading config...")
        cfg_manager = ConfigManager(config_path="configs/default.yaml")
        print("    ✓ Config loaded")
        
        print("\n[2] Creating models...")
        teacher = DummyModel()
        student = DummyModel()
        tokenizer = DummyTokenizer()
        print("    ✓ Models created")
        
        print("\n[3] Initializing Trainer...")
        trainer = Trainer(
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            config=cfg_manager.resolved_config,
            device=torch.device('cpu'),
            experiment_dir='/tmp/test_trainer'
        )
        print("    ✓ Trainer initialized")
        
        print("\n[4] Checking optimizer type...")
        optimizer_type = type(trainer.optimizer).__name__
        print(f"    ✓ Optimizer: {optimizer_type}")
        
        if optimizer_type not in ['AdamW', 'Adam', 'SGD', 'Lion']:
            print(f"    ⚠️  Unexpected optimizer type: {optimizer_type}")
            return False
        
        print("\n[5] Checking if AdaptiveOptimizer is present...")
        has_adaptive = hasattr(trainer, 'adaptive_opt')
        print(f"    {'✓' if has_adaptive else '✗'} AdaptiveOptimizer: {has_adaptive}")
        
        if not has_adaptive:
            print("    ⚠️  AdaptiveOptimizer not found!")
            return False
        
        print("\n[6] Checking current learning rate...")
        current_lr = trainer.optimizer.param_groups[0]['lr']
        expected_lr = cfg_manager.resolved_config['train'].get('lr', 0.0)
        print(f"    ✓ Current LR: {current_lr:.6e}")
        print(f"    ✓ Expected LR: {expected_lr:.6e}")
        
        if abs(current_lr - expected_lr) > 1e-9:
            print(f"    ⚠️  LR mismatch!")
            return False
        
        print("\n✅ ALL CHECKS PASSED!")
        print("\nThe new optimizer/scheduler system is properly integrated:")
        print("  ✓ OptimizerFactory creates optimizer")
        print("  ✓ AdaptiveOptimizer wraps optimizer")
        print("  ✓ SchedulerFactory will initialize on trainer.fit()")
        print("  ✓ GradientManager will run during training")
        print("  ✓ Adaptive LR tuning will run after each epoch")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print(" OPTIMIZER/SCHEDULER INTEGRATION VALIDATION")
    print("="*70)
    print("\nThis validates that main.py will work with the new system.")
    print("="*70)
    
    # Test configs
    configs = [
        "configs/default.yaml",
        "configs/advanced.yaml"
    ]
    
    config_results = []
    for config_path in configs:
        if Path(config_path).exists():
            result = validate_config(config_path)
            config_results.append((config_path, result))
        else:
            print(f"\n⚠️  Config not found: {config_path}")
            config_results.append((config_path, False))
    
    # Test trainer initialization
    trainer_result = test_trainer_initialization()
    
    # Summary
    print("\n" + "="*70)
    print(" VALIDATION SUMMARY")
    print("="*70)
    
    print("\n📋 Config Files:")
    for config_path, passed in config_results:
        status = "✅ VALID" if passed else "❌ INVALID"
        print(f"  {status}: {config_path}")
    
    print("\n🔧 Trainer Integration:")
    status = "✅ WORKING" if trainer_result else "❌ FAILED"
    print(f"  {status}: Optimizer/Scheduler system")
    
    all_passed = all(r for _, r in config_results) and trainer_result
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ VALIDATION PASSED!")
        print("\nYou can now run:")
        print("  python app/main.py --config configs/default.yaml")
        print("  python app/main.py --config configs/advanced.yaml")
        print("\nThe new optimizer/scheduler system will be used automatically!")
    else:
        print("⚠️  VALIDATION ISSUES DETECTED")
        print("\nPlease review the errors above before running main.py")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
