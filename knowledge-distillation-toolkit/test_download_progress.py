"""
Quick test script for download progress monitoring
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.utils.download_monitor import install_progress_hooks, check_model_cached

print("=" * 60)
print("DOWNLOAD PROGRESS MONITORING TEST")
print("=" * 60)

# Install hooks
print("\n1. Installing progress hooks...")
install_progress_hooks()
print("   ✓ Hooks installed")

# Check cache status
print("\n2. Checking model cache status...")
models_to_check = [
    "distilbert-base-uncased",
    "google/mobilebert-uncased",
    "prajjwal1/bert-tiny",
]

for model_id in models_to_check:
    cached = check_model_cached(model_id)
    status = "✓ CACHED" if cached else "✗ NOT CACHED (will download)"
    print(f"   {model_id}: {status}")

# Test with a small model
print("\n3. Testing download progress with small model...")
print("   Model: prajjwal1/bert-tiny (~17MB)")
print("   This will download if not cached, or load from cache if available")
print("\n" + "=" * 60)

try:
    from transformers import AutoModel
    import torch
    
    print("\n[TEST] Loading prajjwal1/bert-tiny...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    print("\n[TEST] ✓ Model loaded successfully!")
    print(f"[TEST] Model type: {type(model).__name__}")
    print(f"[TEST] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"\n[TEST] ✗ Failed to load model: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
