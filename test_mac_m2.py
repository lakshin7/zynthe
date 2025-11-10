"""#!/usr/bin/env python3

Quick Mac M2 Test - Verify Fix Works"""

#Tests the trainer fix on Apple Silicon without running full training.Quick validation test for the knowledge distillation toolkit on Mac M2.

"""This script validates that all components work together correctly.

"""

import sys

from pathlib import Pathimport sys

sys.path.insert(0, str(Path(__file__).parent))import torch

from pathlib import Path

import torch

import inspect# Add project root to path

project_root = Path(__file__).parent

def test_environment():sys.path.insert(0, str(project_root))

    """Test 1: Check environment and MPS availability."""

    print("\n" + "="*70)def test_imports():

    print("TEST 1: Environment Check")    """Test that all required modules can be imported."""

    print("="*70)    print("🔍 Testing imports...")

        

    print(f"Python: {sys.version}")    try:

    print(f"PyTorch: {torch.__version__}")        from core.config.config_manager import ConfigManager

            print("✅ ConfigManager imported successfully")

    if torch.backends.mps.is_available():        

        print("✅ MPS (Apple Silicon) available")        from core.models.model_loader import load_models, model_summary

        device = "mps"        print("✅ Model loader imported successfully")

    elif torch.cuda.is_available():        

        print("✅ CUDA available")        from transformers import AutoTokenizer

        device = "cuda"        print("✅ Transformers imported successfully")

    else:        

        print("⚠️  Using CPU")        # Make functions available globally for other tests

        device = "cpu"        globals()['ConfigManager'] = ConfigManager

            globals()['load_models'] = load_models

    print(f"Device: {device}")        globals()['model_summary'] = model_summary

    return True, device        

        return True

    except ImportError as e:

def test_model_loader():        print(f"❌ Import error: {e}")

    """Test 2: Verify model loader has label mappings."""        return False

    print("\n" + "="*70)

    print("TEST 2: Model Loader Fix (Label Mappings)")def test_device():

    print("="*70)    """Test device detection and MPS availability."""

        print("\n🖥️  Testing device detection...")

    try:    

        from core.models.model_loader import load_models    print(f"PyTorch version: {torch.__version__}")

        from core.config.config_manager import ConfigManager    

            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():

        cfg = ConfigManager('configs/retrain_teacher.yaml')        print("✅ MPS (Apple Silicon GPU) is available")

        print("Loading models (this may take a minute)...")        device = torch.device("mps")

        teacher, student, tokenizer = load_models(cfg, device='cpu')        

                # Test basic MPS operation

        # Check teacher        try:

        assert hasattr(teacher.config, 'label2id'), "Teacher missing label2id"            x = torch.randn(2, 3).to(device)

        assert hasattr(teacher.config, 'id2label'), "Teacher missing id2label"            y = torch.matmul(x, x.T)

        assert teacher.config.label2id == {'negative': 0, 'positive': 1}, \            print(f"✅ MPS basic operation test passed: {y.shape}")

            f"Wrong label2id: {teacher.config.label2id}"            return "mps"

                except Exception as e:

        print(f"✅ Teacher label2id: {teacher.config.label2id}")            print(f"⚠️  MPS available but operation failed: {e}")

        print(f"✅ Teacher id2label: {teacher.config.id2label}")            return "cpu"

            else:

        # Check student        print("ℹ️  MPS not available, using CPU")

        assert hasattr(student.config, 'label2id'), "Student missing label2id"        return "cpu"

        assert student.config.label2id == {'negative': 0, 'positive': 1}

        def test_config():

        print(f"✅ Student label2id: {student.config.label2id}")    """Test configuration loading."""

        print(f"✅ Student id2label: {student.config.id2label}")    print("\n⚙️  Testing configuration...")

            

        return True    try:

    except Exception as e:        from core.config.config_manager import ConfigManager

        print(f"❌ Error: {e}")        

        return False        # Test with our Mac M2 config

        config_path = "configs/mac_m2_test.yaml"

        if not Path(config_path).exists():

def test_trainer_fix():            config_path = "configs/default.yaml"

    """Test 3: Verify trainer has teacher fine-tuning."""            

    print("\n" + "="*70)        cm = ConfigManager(config_path=config_path)

    print("TEST 3: Trainer Fix (Teacher Fine-tuning)")        print(f"✅ Configuration loaded from {config_path}")

    print("="*70)        

            device = cm.device()

    try:        print(f"✅ Device detected: {device}")

        from training.trainer import Trainer        

                config = cm.resolved_config

        # Check finetune_teacher method exists        print(f"✅ Model config: {config.get('model', {}).get('name')} → {config.get('model', {}).get('student_name')}")

        assert hasattr(Trainer, 'finetune_teacher'), "Missing finetune_teacher method"        

        print("✅ Trainer has finetune_teacher method")        return cm

            except Exception as e:

        # Check __init__ creates teacher_optimizer        print(f"❌ Configuration test failed: {e}")

        init_source = inspect.getsource(Trainer.__init__)        return None

        assert 'teacher_optimizer' in init_source, "Missing teacher_optimizer in __init__"

        print("✅ Trainer creates teacher_optimizer")def test_model_loading(config_manager, device_str):

            """Test model loading (lightweight models only)."""

        # Check fit calls finetune_teacher    print("\n🤖 Testing model loading...")

        fit_source = inspect.getsource(Trainer.fit)    

        assert 'finetune_teacher' in fit_source, "fit() doesn't call finetune_teacher"    try:

        print("✅ Trainer.fit() calls finetune_teacher")        # Create a test ConfigManager with very small models

                import tempfile

        # Check for PHASE comments        import yaml

        if "PHASE 1" in fit_source and "PHASE 2" in fit_source:        

            print("✅ Two-phase training (teacher + distillation)")        test_config_data = {

                    "model": {

        return True                "name": "prajjwal1/bert-tiny",       # Only 4M parameters

    except Exception as e:                "student_name": "prajjwal1/bert-mini", # Only 11M parameters  

        print(f"❌ Error: {e}")                "type": "transformer",

        import traceback                "tokenizer_name": "prajjwal1/bert-tiny"

        traceback.print_exc()            },

        return False            "train": {"epochs": 1, "batch_size": 4, "lr": 5e-5},

            "data": {"train_path": "dummy", "val_path": "dummy"},

            "distillation": {"method": "kd_hinton"}

def test_config():        }

    """Test 4: Verify config has teacher settings."""        

    print("\n" + "="*70)        # Create temporary config file

    print("TEST 4: Configuration")        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:

    print("="*70)            yaml.safe_dump(test_config_data, f)

                temp_config_path = f.name

    try:        

        import yaml        try:

        with open('configs/retrain_teacher.yaml') as f:            # Create ConfigManager with test config

            cfg = yaml.safe_load(f)            from core.config.config_manager import ConfigManager

                    from core.models.model_loader import load_models, model_summary

        teacher_epochs = cfg['train'].get('teacher_epochs')            

        finetune_teacher = cfg['train'].get('finetune_teacher')            test_cm = ConfigManager(config_path=temp_config_path)

                    

        assert teacher_epochs is not None, "Missing teacher_epochs in config"            print("Loading tiny models for testing...")

        assert finetune_teacher is not None, "Missing finetune_teacher in config"            teacher, student, tokenizer = load_models(test_cm, device=device_str)

                    

        print(f"✅ Teacher epochs: {teacher_epochs}")            print("✅ Teacher model loaded:")

        print(f"✅ Finetune teacher: {finetune_teacher}")            teacher_summary = model_summary(teacher)

        print(f"✅ Distillation epochs: {cfg['train']['epochs']}")            print(f"   - Name: {teacher_summary['name']}")

        print(f"✅ Batch size: {cfg['train']['batch_size']}")            print(f"   - Parameters: {teacher_summary['parameters']:,}")

                    print(f"   - Device: {teacher_summary['device']}")

        return True            

    except Exception as e:            print("✅ Student model loaded:")

        print(f"❌ Error: {e}")            student_summary = model_summary(student)

        return False            print(f"   - Name: {student_summary['name']}")

            print(f"   - Parameters: {student_summary['parameters']:,}")

            print(f"   - Device: {student_summary['device']}")

def test_tools():            

    """Test 5: Verify diagnostic tools exist."""            print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")

    print("\n" + "="*70)            

    print("TEST 5: Diagnostic Tools")            return teacher, student, tokenizer

    print("="*70)            

            finally:

    tools = [            # Clean up temp file

        'tools/diagnose_teacher.py',            import os

        'tools/repair_teacher_labels.py',            os.unlink(temp_config_path)

    ]        

        except Exception as e:

    all_exist = True        print(f"❌ Model loading failed: {e}")

    for tool in tools:        import traceback

        if Path(tool).exists():        traceback.print_exc()

            print(f"✅ {tool}")        return None, None, None

        else:

            print(f"❌ {tool} not found")def test_forward_pass(teacher, student, tokenizer, device_str):

            all_exist = False    """Test forward pass with sample data."""

        print("\n🚀 Testing forward pass...")

    return all_exist    

    try:

        device = torch.device(device_str)

def main():        

    print("\n" + "╔" + "="*68 + "╗")        # Create sample input

    print("║" + " "*20 + "Mac M2 Quick Test" + " "*30 + "║")        sample_text = ["This is a great movie!", "This movie is terrible."]

    print("║" + " "*15 + "Verify Trainer Fix Works" + " "*25 + "║")        inputs = tokenizer(

    print("╚" + "="*68 + "╝")            sample_text, 

                return_tensors="pt", 

    results = []            padding=True, 

                truncation=True, 

    # Run tests            max_length=64

    results.append(("Environment", test_environment()[0]))        )

    results.append(("Model Loader", test_model_loader()))        

    results.append(("Trainer Fix", test_trainer_fix()))        # Move inputs to device

    results.append(("Configuration", test_config()))        input_ids = inputs["input_ids"].to(device)

    results.append(("Tools", test_tools()))        attention_mask = inputs["attention_mask"].to(device)

            

    # Summary        print(f"✅ Sample input shape: {input_ids.shape}")

    print("\n" + "="*70)        

    print("TEST SUMMARY")        # Test teacher forward pass

    print("="*70)        with torch.no_grad():

                teacher_output = teacher(input_ids=input_ids, attention_mask=attention_mask)

    passed = sum(1 for _, result in results if result)            print(f"✅ Teacher forward pass: output shape {teacher_output.logits.shape}")

    total = len(results)        

            # Test student forward pass

    for name, result in results:        with torch.no_grad():

        status = "✅ PASS" if result else "❌ FAIL"            student_output = student(input_ids=input_ids, attention_mask=attention_mask)

        print(f"{status}: {name}")            print(f"✅ Student forward pass: output shape {student_output.logits.shape}")

                

    print(f"\nResults: {passed}/{total} tests passed")        # Test basic distillation loss computation

            teacher_logits = teacher_output.logits

    if passed == total:        student_logits = student_output.logits

        print("\n🎉 All tests passed! Ready to train.")        

        print("\nNext steps:")        temperature = 2.0

        print("  1. Run training: ./run_training.sh")        alpha = 0.5

        print("  2. Or full test: ./test_pipeline.sh")        

        return 0        # Soft targets from teacher

    else:        soft_teacher = torch.softmax(teacher_logits / temperature, dim=1)

        print("\n⚠️  Some tests failed. Check the output above.")        soft_student = torch.log_softmax(student_logits / temperature, dim=1)

        return 1        

        # KL divergence loss

        kd_loss = torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean')

if __name__ == "__main__":        print(f"✅ KD loss computed: {kd_loss.item():.4f}")

    sys.exit(main())        

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