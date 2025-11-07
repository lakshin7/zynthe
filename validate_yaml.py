#!/usr/bin/env python3
"""
Simple YAML validation script to check default.yaml structure.
"""

import yaml
import os

def validate_yaml():
    """Validate the default.yaml configuration file."""
    
    print("=" * 70)
    print("Validating default.yaml Configuration")
    print("=" * 70)
    
    yaml_path = "configs/default.yaml"
    
    # Check if file exists
    if not os.path.exists(yaml_path):
        print(f"❌ File not found: {yaml_path}")
        return False
    
    print(f"\n✅ File exists: {yaml_path}")
    
    # Try to parse YAML
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ YAML syntax is valid")
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        return False
    
    # Check required sections
    required_sections = ["train", "model", "distillation", "data", "device"]
    print(f"\nChecking required sections...")
    
    for section in required_sections:
        if section in config:
            print(f"  ✅ Section '{section}' present")
        else:
            print(f"  ❌ Section '{section}' missing")
            return False
    
    # Check required keys in train section
    print(f"\nChecking 'train' section keys...")
    train_keys = ["epochs", "batch_size", "lr", "grad_accum_steps", 
                  "mixed_precision", "early_stop_patience"]
    
    for key in train_keys:
        if key in config.get("train", {}):
            value = config["train"][key]
            print(f"  ✅ train.{key} = {value}")
        else:
            print(f"  ❌ train.{key} missing")
            return False
    
    # Check required keys in model section
    print(f"\nChecking 'model' section keys...")
    model_keys = ["name", "student_name", "type", "tokenizer_name", "max_length"]
    
    for key in model_keys:
        if key in config.get("model", {}):
            value = config["model"][key]
            print(f"  ✅ model.{key} = {value}")
        else:
            print(f"  ❌ model.{key} missing")
            return False
    
    # Check distillation section
    print(f"\nChecking 'distillation' section keys...")
    distill_keys = ["method", "temperature", "alpha"]
    
    for key in distill_keys:
        if key in config.get("distillation", {}):
            value = config["distillation"][key]
            print(f"  ✅ distillation.{key} = {value}")
        else:
            print(f"  ❌ distillation.{key} missing")
            return False
    
    # Check data section
    print(f"\nChecking 'data' section keys...")
    data_keys = ["train_path", "val_path"]
    
    for key in data_keys:
        if key in config.get("data", {}):
            value = config["data"][key]
            print(f"  ✅ data.{key} = {value}")
        else:
            print(f"  ❌ data.{key} missing")
            return False
    
    # Check optional sections
    print(f"\nChecking optional sections...")
    optional_sections = ["quantization", "explainability"]
    
    for section in optional_sections:
        if section in config:
            print(f"  ✅ Optional section '{section}' present")
        else:
            print(f"  ⚠️  Optional section '{section}' not present (OK)")
    
    # Check quantization if present
    if "quantization" in config:
        print(f"\nChecking 'quantization' section...")
        if "enable" in config["quantization"]:
            print(f"  ✅ quantization.enable = {config['quantization']['enable']}")
        if "mode" in config["quantization"]:
            print(f"  ✅ quantization.mode = {config['quantization']['mode']}")
    
    # Check similarity_transfer
    if "similarity_transfer" in config:
        print(f"\n✅ similarity_transfer = {config['similarity_transfer']}")
    
    # Check seed
    if "seed" in config:
        print(f"✅ seed = {config['seed']}")
    
    # Check output_root
    if "output_root" in config:
        print(f"✅ output_root = {config['output_root']}")
    
    print("\n" + "=" * 70)
    print("✅ YAML configuration is valid and complete!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    import sys
    success = validate_yaml()
    sys.exit(0 if success else 1)
