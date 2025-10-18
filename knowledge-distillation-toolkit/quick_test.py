#!/usr/bin/env python3
"""
Quick Test - Verify Trainer Fix
Run this to test that all fixes are working before starting full training.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def test_environment():
    """Test 1: Environment and MPS."""
    print("\n" + "="*70)
    print("TEST 1: Environment Check")
    print("="*70)
    
    import torch
    print(f"✅ Python: {sys.version.split()[0]}")
    print(f"✅ PyTorch: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) available")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA available")
        return "cuda"
    else:
        print("✅ Using CPU")
        return "cpu"


def test_model_loader():
    """Test 2: Model loader with label mappings."""
    print("\n" + "="*70)
    print("TEST 2: Model Loader Fix")
    print("="*70)
    
    try:
        from core.models.model_loader import load_models
        from core.config.config_manager import ConfigManager
        
        cfg = ConfigManager('configs/retrain_teacher.yaml')
        print("Loading models on CPU (this takes ~30 seconds)...")
        teacher, student, tokenizer = load_models(cfg, device='cpu')
        
        # Verify label mappings
        assert teacher.config.label2id == {'negative': 0, 'positive': 1}
        assert student.config.label2id == {'negative': 0, 'positive': 1}
        
        print(f"✅ Teacher labels: {teacher.config.label2id}")
        print(f"✅ Student labels: {student.config.label2id}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_trainer():
    """Test 3: Trainer has teacher fine-tuning."""
    print("\n" + "="*70)
    print("TEST 3: Trainer Fix")
    print("="*70)
    
    try:
        from training.trainer import Trainer
        import inspect
        
        # Check methods exist
        assert hasattr(Trainer, 'finetune_teacher')
        print("✅ finetune_teacher method exists")
        
        # Check __init__ has teacher_optimizer
        init_code = inspect.getsource(Trainer.__init__)
        assert 'teacher_optimizer' in init_code
        print("✅ teacher_optimizer created in __init__")
        
        # Check fit calls finetune_teacher
        fit_code = inspect.getsource(Trainer.fit)
        assert 'finetune_teacher' in fit_code
        assert 'PHASE 1' in fit_code and 'PHASE 2' in fit_code
        print("✅ Two-phase training: teacher + distillation")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_config():
    """Test 4: Config has teacher settings."""
    print("\n" + "="*70)
    print("TEST 4: Configuration")
    print("="*70)
    
    try:
        import yaml
        with open('configs/retrain_teacher.yaml') as f:
            cfg = yaml.safe_load(f)
        
        assert cfg['train'].get('teacher_epochs') == 2
        assert cfg['train'].get('finetune_teacher') == True
        
        print(f"✅ Teacher epochs: {cfg['train']['teacher_epochs']}")
        print(f"✅ Finetune teacher: {cfg['train']['finetune_teacher']}")
        print(f"✅ Distillation epochs: {cfg['train']['epochs']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*22 + "Quick Fix Test" + " "*32 + "║")
    print("╚" + "="*68 + "╝")
    
    results = []
    device = test_environment()
    results.append(("Environment", device is not None))
    results.append(("Model Loader", test_model_loader()))
    results.append(("Trainer Fix", test_trainer()))
    results.append(("Configuration", test_config()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results:
        print(f"{'✅' if passed else '❌'} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nTo start training:")
        print("  ./run_training.sh")
        print("\nOr use Python directly:")
        print("  python3 train_with_fix.py")
    else:
        print("\n⚠️  Some tests failed.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
