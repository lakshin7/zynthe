"""
Teacher Model Diagnostic Tool
Identifies issues with teacher model performance and label alignment.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import argparse
from collections import Counter


def diagnose_model(model_path: str, val_path: str = "data/imdb_val.jsonl", max_samples: int = 100):
    """
    Diagnose teacher model issues.
    
    Args:
        model_path: Path to model directory
        val_path: Path to validation JSONL file
        max_samples: Number of samples to test
    """
    print(f"\n{'='*60}")
    print(f"🔍 TEACHER MODEL DIAGNOSTICS")
    print(f"{'='*60}\n")
    
    # Device setup
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    print(f"\n📂 Model Path: {model_path}")
    print(f"📊 Dataset: {val_path}")
    print(f"🔢 Testing {max_samples} samples\n")
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check config
    print(f"\n{'='*60}")
    print("📋 MODEL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model Type: {model.config.model_type}")
    print(f"Num Labels: {model.config.num_labels}")
    print(f"Has label2id: {hasattr(model.config, 'label2id') and model.config.label2id is not None}")
    print(f"Has id2label: {hasattr(model.config, 'id2label') and model.config.id2label is not None}")
    
    if hasattr(model.config, 'label2id') and model.config.label2id:
        print(f"label2id: {model.config.label2id}")
    if hasattr(model.config, 'id2label') and model.config.id2label:
        print(f"id2label: {model.config.id2label}")
    
    # Load validation data
    print(f"\n{'='*60}")
    print("📖 LOADING DATASET")
    print(f"{'='*60}")
    samples = []
    with open(val_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line))
    
    label_dist = Counter([s['label'] for s in samples])
    print(f"Loaded {len(samples)} samples")
    print(f"Label Distribution: {dict(label_dist)}")
    
    # Run predictions
    print(f"\n{'='*60}")
    print("🎯 RUNNING PREDICTIONS")
    print(f"{'='*60}")
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for sample in samples:
            text = sample['text']
            label = sample['label']
            
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            
            predictions.append(pred)
            true_labels.append(label)
    
    # Analyze predictions
    pred_dist = Counter(predictions)
    print(f"\nPrediction Distribution: {dict(pred_dist)}")
    print(f"True Label Distribution: {dict(label_dist)}")
    
    # Compute accuracy
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(predictions)
    
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    # Check for issues
    print(f"\n{'='*60}")
    print("🔍 DIAGNOSTIC FINDINGS")
    print(f"{'='*60}\n")
    
    issues_found = False
    
    # Issue 1: Missing label mappings
    if not (hasattr(model.config, 'label2id') and model.config.label2id):
        print("❌ ISSUE: Missing label2id mapping in config")
        print("   This can cause the classifier head to be misaligned with dataset labels.")
        issues_found = True
    
    if not (hasattr(model.config, 'id2label') and model.config.id2label):
        print("❌ ISSUE: Missing id2label mapping in config")
        issues_found = True
    
    # Issue 2: Chance-level performance
    if accuracy < 0.55:
        print(f"❌ ISSUE: Accuracy ({accuracy:.4f}) is near chance level (0.50)")
        print("   Possible causes:")
        print("   - Model classification head not properly trained")
        print("   - Label mapping mismatch between model and dataset")
        print("   - Tokenizer mismatch during training/evaluation")
        issues_found = True
    
    # Issue 3: Prediction bias
    pred_ratio = pred_dist.get(0, 0) / pred_dist.get(1, 1) if pred_dist.get(1, 0) > 0 else float('inf')
    if pred_ratio > 3 or pred_ratio < 0.33:
        print(f"⚠️  WARNING: Prediction bias detected (ratio: {pred_ratio:.2f})")
        print(f"   Model predicts class 0: {pred_dist.get(0, 0)} times")
        print(f"   Model predicts class 1: {pred_dist.get(1, 0)} times")
        issues_found = True
    
    # Issue 4: Sample predictions
    print(f"\n{'='*60}")
    print("📝 SAMPLE PREDICTIONS")
    print(f"{'='*60}\n")
    
    for i in range(min(5, len(samples))):
        sample = samples[i]
        pred = predictions[i]
        true_label = true_labels[i]
        
        match = "✓" if pred == true_label else "✗"
        print(f"{match} Sample {i+1}:")
        print(f"   Text: {sample['text'][:100]}...")
        print(f"   True Label: {true_label}, Predicted: {pred}\n")
    
    if not issues_found:
        print("\n✅ No obvious issues detected!")
    else:
        print(f"\n{'='*60}")
        print("💡 RECOMMENDED ACTIONS")
        print(f"{'='*60}\n")
        print("1. Use the repair utility to add label mappings:")
        print(f"   python tools/repair_teacher_labels.py --model {model_path}\n")
        print("2. If accuracy remains low, consider:")
        print("   - Retraining the teacher model from scratch")
        print("   - Verifying the teacher checkpoint is from a completed training run")
        print("   - Checking if the saved model is actually fine-tuned (not base model)\n")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'issues_found': issues_found
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose teacher model issues")
    parser.add_argument("--model", required=True, help="Path to teacher model directory")
    parser.add_argument("--val", default="data/imdb_val.jsonl", help="Path to validation data")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to test")
    
    args = parser.parse_args()
    
    diagnose_model(args.model, args.val, args.samples)
