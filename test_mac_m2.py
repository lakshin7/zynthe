#!/usr/bin/env python3
"""
Quick validation test for the knowledge distillation toolkit on Mac M2.
This script validates that all components work together correctly.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from core.config.config_manager import ConfigManager
        print("✅ ConfigManager imported successfully")
        
        from core.models.model_loader import load_models, model_summary
        print("✅ Model loader imported successfully")
        
        from transformers import AutoTokenizer
        print("✅ Transformers imported successfully")
        
        # Make functions available globally for other tests
        globals()['ConfigManager'] = ConfigManager
        globals()['load_models'] = load_models
        globals()['model_summary'] = model_summary
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_device():
    """Test device detection and MPS availability."""
    print("\n🖥️  Testing device detection...")
    
    print(f"PyTorch version: {torch.__version__}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) is available")
        device = torch.device("mps")
        
        # Test basic MPS operation
        try:
            x = torch.randn(2, 3).to(device)
            y = torch.matmul(x, x.T)
            print(f"✅ MPS basic operation test passed: {y.shape}")
            return "mps"
        except Exception as e:
            print(f"⚠️  MPS available but operation failed: {e}")
            return "cpu"
    else:
        print("ℹ️  MPS not available, using CPU")
        return "cpu"

def test_config():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from core.config.config_manager import ConfigManager
        
        # Test with our Mac M2 config
        config_path = "configs/mac_m2_test.yaml"
        if not Path(config_path).exists():
            config_path = "configs/default.yaml"
            
        cm = ConfigManager(config_path=config_path)
        print(f"✅ Configuration loaded from {config_path}")
        
        device = cm.device()
        print(f"✅ Device detected: {device}")
        
        config = cm.resolved_config
        print(f"✅ Model config: {config.get('model', {}).get('name')} → {config.get('model', {}).get('student_name')}")
        
        return cm
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None

def test_model_loading(config_manager, device_str):
    """Test model loading (lightweight models only)."""
    print("\n🤖 Testing model loading...")
    
    try:
        # Create a test ConfigManager with very small models
        import tempfile
        import yaml
        
        test_config_data = {
            "model": {
                "name": "prajjwal1/bert-tiny",       # Only 4M parameters
                "student_name": "prajjwal1/bert-mini", # Only 11M parameters  
                "type": "transformer",
                "tokenizer_name": "prajjwal1/bert-tiny"
            },
            "train": {"epochs": 1, "batch_size": 4, "lr": 5e-5},
            "data": {"train_path": "dummy", "val_path": "dummy"},
            "distillation": {"method": "kd_hinton"}
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(test_config_data, f)
            temp_config_path = f.name
        
        try:
            # Create ConfigManager with test config
            from core.config.config_manager import ConfigManager
            from core.models.model_loader import load_models, model_summary
            
            test_cm = ConfigManager(config_path=temp_config_path)
            
            print("Loading tiny models for testing...")
            teacher, student, tokenizer = load_models(test_cm, device=device_str)
            
            print("✅ Teacher model loaded:")
            teacher_summary = model_summary(teacher)
            print(f"   - Name: {teacher_summary['name']}")
            print(f"   - Parameters: {teacher_summary['parameters']:,}")
            print(f"   - Device: {teacher_summary['device']}")
            
            print("✅ Student model loaded:")
            student_summary = model_summary(student)
            print(f"   - Name: {student_summary['name']}")
            print(f"   - Parameters: {student_summary['parameters']:,}")
            print(f"   - Device: {student_summary['device']}")
            
            print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
            
            return teacher, student, tokenizer
            
        finally:
            # Clean up temp file
            import os
            os.unlink(temp_config_path)
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_forward_pass(teacher, student, tokenizer, device_str):
    """Test forward pass with sample data."""
    print("\n🚀 Testing forward pass...")
    
    try:
        device = torch.device(device_str)
        
        # Create sample input
        sample_text = ["This is a great movie!", "This movie is terrible."]
        inputs = tokenizer(
            sample_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        )
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        print(f"✅ Sample input shape: {input_ids.shape}")
        
        # Test teacher forward pass
        with torch.no_grad():
            teacher_output = teacher(input_ids=input_ids, attention_mask=attention_mask)
            print(f"✅ Teacher forward pass: output shape {teacher_output.logits.shape}")
        
        # Test student forward pass
        with torch.no_grad():
            student_output = student(input_ids=input_ids, attention_mask=attention_mask)
            print(f"✅ Student forward pass: output shape {student_output.logits.shape}")
            
        # Test basic distillation loss computation
        teacher_logits = teacher_output.logits
        student_logits = student_output.logits
        
        temperature = 2.0
        alpha = 0.5
        
        # Soft targets from teacher
        soft_teacher = torch.softmax(teacher_logits / temperature, dim=1)
        soft_student = torch.log_softmax(student_logits / temperature, dim=1)
        
        # KL divergence loss
        kd_loss = torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean')
        print(f"✅ KD loss computed: {kd_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🧪 Knowledge Distillation Toolkit - Mac M2 Validation")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please install requirements:")
        print("pip install -r requirements.txt")
        return False
    
    # Test 2: Device detection
    device_str = test_device()
    
    # Test 3: Configuration
    config_manager = test_config()
    if config_manager is None:
        return False
    
    # Test 4: Model loading (with small models)
    teacher, student, tokenizer = test_model_loading(config_manager, device_str)
    if teacher is None:
        print("\n⚠️  Model loading failed. This might be due to:")
        print("1. Network connection (models need to download first time)")
        print("2. Disk space (models require ~100MB)")
        print("3. Missing transformers package")
        return False
    
    # Test 5: Forward pass
    if not test_forward_pass(teacher, student, tokenizer, device_str):
        return False
    
    # Summary
    print("\n🎉 All tests passed! Your Mac M2 setup is ready for knowledge distillation.")
    print("\n📝 Next steps:")
    print("1. Run your first experiment:")
    print("   python app/main.py --config configs/mac_m2_test.yaml")
    print("\n2. For production training:")
    print("   python app/main.py --config configs/default.yaml")
    print("\n3. Check the documentation:")
    print("   - docs/quickstart.md")
    print("   - docs/msme_playbook.md")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)