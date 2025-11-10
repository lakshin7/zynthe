#!/usr/bin/env python3
"""
Quick test for model validator
"""
import sys
sys.path.insert(0, '/Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit')

from core.preflight.model_validator import validate_models

print("🧪 Testing Model Validator...\n")

# Test 1: Valid pair
print("Test 1: Validating BERT → DistilBERT (should pass)")
print("-" * 60)
result = validate_models('bert-base-uncased', 'distilbert-base-uncased')
print(f"✓ Pair compatible: {result['pair_compatible']}")
print(f"✓ Teacher exists: {result['teacher']['exists']}")
print(f"✓ Student exists: {result['student']['exists']}")
print(f"✓ Teacher device compatible: {result['teacher']['device_compatible']}")
print(f"✓ Student device compatible: {result['student']['device_compatible']}")
if 'compression_ratio' in result:
    print(f"✓ Compression ratio: {result['compression_ratio']}")
print()

# Test 2: Invalid model
print("Test 2: Validating non-existent model (should fail)")
print("-" * 60)
result = validate_models('non-existent-model-12345', 'distilbert-base-uncased')
print(f"✗ Pair compatible: {result['pair_compatible']}")
print(f"✗ Teacher errors: {result['teacher']['errors']}")
print(f"✓ Alternatives suggested: {len(result['teacher']['alternatives'])} options")
print()

# Test 3: Device info
print("Test 3: Checking device capabilities")
print("-" * 60)
from core.preflight.model_validator import ModelValidator
validator = ModelValidator()
print(f"✓ Available device: {validator.available_device}")
print()

print("🎉 All tests completed!")
