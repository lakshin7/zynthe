#!/usr/bin/env python3
"""
Quick test script to validate ConfigManager improvements.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def test_config_manager():
    """Test the ConfigManager with various scenarios."""
    from core.config.config_manager import ConfigManager
    
    print("=" * 70)
    print("Testing ConfigManager")
    print("=" * 70)
    
    # Test 1: Load default config
    print("\n[Test 1] Loading default configuration...")
    try:
        cfg = ConfigManager()
        print("✅ Default config loaded successfully")
        print(f"   Device: {cfg.device()}")
        print(f"   Experiment ID: {cfg.experiment_id}")
        print(f"   Defaults path: {cfg.defaults_path}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 2: Access configuration values
    print("\n[Test 2] Accessing configuration values...")
    try:
        epochs = cfg.get("train", {}).get("epochs")
        lr = cfg.get("train", {}).get("lr")
        batch_size = cfg.get("train", {}).get("batch_size")
        early_stop = cfg.get("train", {}).get("early_stop_patience")
        
        print(f"✅ Config values accessible")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Batch size: {batch_size}")
        print(f"   Early stop patience: {early_stop}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 3: Use get_nested method
    print("\n[Test 3] Testing get_nested method...")
    try:
        model_name = cfg.get_nested("model", "name")
        student_name = cfg.get_nested("model", "student_name")
        tokenizer = cfg.get_nested("model", "tokenizer_name")
        
        print(f"✅ Nested access works")
        print(f"   Teacher model: {model_name}")
        print(f"   Student model: {student_name}")
        print(f"   Tokenizer: {tokenizer}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 4: Check quantization config
    print("\n[Test 4] Checking quantization configuration...")
    try:
        quant_enable = cfg.get_nested("quantization", "enable")
        quant_mode = cfg.get_nested("quantization", "mode")
        similarity = cfg.get("similarity_transfer")
        
        print(f"✅ Extended config accessible")
        print(f"   Quantization enabled: {quant_enable}")
        print(f"   Quantization mode: {quant_mode}")
        print(f"   Similarity transfer: {similarity}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 5: Check runtime resolution
    print("\n[Test 5] Checking runtime resolution...")
    try:
        runtime = cfg.get_runtime()
        print(f"✅ Runtime config accessible")
        print(f"   Device: {runtime.get('device')}")
        print(f"   Seed: {runtime.get('seed')}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 6: Experiment info
    print("\n[Test 6] Getting experiment info...")
    try:
        info = cfg.experiment_info()
        print(f"✅ Experiment info accessible")
        print(f"   Experiment ID: {info['id']}")
        print(f"   Experiment dir: {info['dir']}")
        print(f"   Checkpoints dir: {info['paths'].get('checkpoints')}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 7: String representation
    print("\n[Test 7] Testing __repr__...")
    try:
        repr_str = repr(cfg)
        print(f"✅ String representation works")
        print(f"{repr_str}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 8: Load custom config
    print("\n[Test 8] Loading custom config (default.yaml)...")
    try:
        cfg2 = ConfigManager(config_path="configs/default.yaml")
        print(f"✅ Custom config loaded successfully")
        print(f"   Teacher: {cfg2.get_nested('model', 'name')}")
        print(f"   Student: {cfg2.get_nested('model', 'student_name')}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_config_manager()
    sys.exit(0 if success else 1)
