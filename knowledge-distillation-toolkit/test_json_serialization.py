"""
Test script to verify JSON serialization fix for numpy arrays
"""

import numpy as np
import json
from pathlib import Path

print("=" * 60)
print("JSON SERIALIZATION TEST")
print("=" * 60)

# Define the conversion function (copy from main.py)
def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-JSON-serializable objects to JSON-serializable types.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        # Handle all numpy scalar types
        return obj.item()
    else:
        return obj

# Create test data with various numpy types (simulating evaluation metrics)
test_data = {
    'teacher': {
        'accuracy': np.float64(0.912),
        'f1': np.float32(0.895),
        'loss': np.array([0.245]),
        'predictions': np.array([0, 1, 1, 0, 1])
    },
    'student': {
        'accuracy': np.float64(0.891),
        'f1': np.float32(0.878),
        'loss': np.array([0.312])
    },
    'extended_metrics': {
        'kl_divergence': np.float64(0.0234),
        'js_divergence': np.array([[0.012, 0.015]]),
        'prediction_agreement': np.float32(0.87),
        'confidence_correlation': np.array([0.91, 0.88, 0.93])
    },
    'dei': {
        'dei': 0.856,
        'accuracy_retention': 1.0731,
        'compression_ratio': 0.48
    },
    'cas': {
        'cas': -0.2124,
        'speedup': np.float32(2.1)
    }
}

print("\n1. Original data types (checking for numpy types):")
print(f"   teacher.accuracy: {type(test_data['teacher']['accuracy'])}")
print(f"   student.loss: {type(test_data['student']['loss'])}")
print(f"   extended_metrics.kl_divergence: {type(test_data['extended_metrics']['kl_divergence'])}")

print("\n2. Converting to serializable format...")
serializable_data = convert_to_serializable(test_data)

print("\n3. Converted data types:")
print(f"   teacher.accuracy: {type(serializable_data['teacher']['accuracy'])} = {serializable_data['teacher']['accuracy']}")
print(f"   student.loss: {type(serializable_data['student']['loss'])} = {serializable_data['student']['loss']}")
print(f"   extended_metrics.confidence_correlation: {type(serializable_data['extended_metrics']['confidence_correlation'])}")

print("\n4. Testing JSON serialization...")
try:
    json_str = json.dumps(serializable_data, indent=2)
    print("   ✅ JSON serialization successful!")
    
    # Try to save to file
    test_file = Path("test_output.json")
    with open(test_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"   ✅ Saved to {test_file}")
    
    # Verify we can load it back
    with open(test_file, 'r') as f:
        loaded_data = json.load(f)
    print("   ✅ Successfully loaded back from file")
    print(f"   ✅ Loaded teacher accuracy: {loaded_data['teacher']['accuracy']}")
    
    # Clean up
    test_file.unlink()
    print("   ✅ Test file cleaned up")
    
except TypeError as e:
    print(f"   ❌ JSON serialization failed: {e}")
    import sys
    sys.exit(1)

print("\n5. Sample JSON output:")
print(json_str[:600] + "\n...")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nThe numpy array serialization fix is working correctly!")
print("Training evaluation results will now save without errors.")

