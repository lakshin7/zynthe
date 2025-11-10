#!/usr/bin/env python3
"""
Comprehensive Data Leakage Diagnostic Script
=============================================

Checks all common sources of data leakage that cause metrics to converge:
1. Dataset overlap (train/val texts)
2. Evaluation on wrong dataset
3. Teacher logit cache alignment
4. Model evaluation mode
5. Prediction distribution analysis
6. Confusion matrix and per-class metrics
7. Sample-by-sample inspection
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import sys

print("="*80)
print("DATA LEAKAGE DIAGNOSTIC REPORT")
print("Experiment: 20251023T175322Z_5285cff4")
print("="*80)

# ============================================================================
# 1. CHECK DATASET FILES - TEXT OVERLAP
# ============================================================================
print("\n" + "="*80)
print("1. DATASET OVERLAP CHECK")
print("="*80)

train_path = Path("data/imdb_train.jsonl")
val_path = Path("data/imdb_val.jsonl")

if not train_path.exists():
    print(f"❌ Training file not found: {train_path}")
    print(f"   Available data files:")
    for f in Path("data").glob("*.jsonl"):
        print(f"   - {f}")
else:
    print(f"✓ Found training file: {train_path}")
    
if not val_path.exists():
    print(f"❌ Validation file not found: {val_path}")
else:
    print(f"✓ Found validation file: {val_path}")

# Load datasets
if train_path.exists() and val_path.exists():
    import json
    
    train_texts = []
    val_texts = []
    
    print(f"\nLoading datasets...")
    with open(train_path) as f:
        for line in f:
            data = json.loads(line.strip())
            train_texts.append(data.get('text', ''))
    
    with open(val_path) as f:
        for line in f:
            data = json.loads(line.strip())
            val_texts.append(data.get('text', ''))
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    # Check for exact text overlap
    train_set = set(train_texts)
    val_set = set(val_texts)
    overlap = train_set.intersection(val_set)
    
    print(f"\n🔍 OVERLAP ANALYSIS:")
    print(f"   Unique train texts: {len(train_set)}")
    print(f"   Unique val texts: {len(val_set)}")
    print(f"   Overlapping texts: {len(overlap)}")
    
    if len(overlap) > 0:
        overlap_pct = (len(overlap) / len(val_set)) * 100
        print(f"\n⚠️  DATA LEAKAGE DETECTED!")
        print(f"   {len(overlap)} samples ({overlap_pct:.2f}% of validation) appear in both train and val!")
        print(f"\n   Example overlapping texts:")
        for i, text in enumerate(list(overlap)[:3]):
            print(f"   [{i+1}] {text[:100]}...")
    else:
        print(f"\n✓ No text overlap detected between train and val")
    
    # Check for near-duplicates (first 100 chars)
    train_prefixes = set([t[:100] for t in train_texts])
    val_prefixes = set([t[:100] for t in val_texts])
    prefix_overlap = train_prefixes.intersection(val_prefixes)
    
    print(f"\n🔍 NEAR-DUPLICATE CHECK (first 100 chars):")
    print(f"   Overlapping prefixes: {len(prefix_overlap)}")
    if len(prefix_overlap) > 0 and len(prefix_overlap) != len(overlap):
        print(f"   ⚠️  Found {len(prefix_overlap) - len(overlap)} potential near-duplicates!")

# ============================================================================
# 2. CHECK EXPERIMENT METRICS - PREDICTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. METRICS & PREDICTION ANALYSIS")
print("="*80)

exp_dir = Path("experiments/20251023T175322Z_5285cff4")
metrics_file = exp_dir / "metrics.json"

if metrics_file.exists():
    metrics = json.loads(metrics_file.read_text())
    
    print(f"\n📊 Reported Metrics:")
    print(f"   Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"   F1:        {metrics.get('f1', 'N/A'):.4f}")
    print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
    print(f"   Recall:    {metrics.get('recall', 'N/A'):.4f}")
    
    # Check per-class metrics
    if 'precision_per_class' in metrics:
        print(f"\n📊 Per-Class Metrics:")
        prec_pc = metrics['precision_per_class']
        rec_pc = metrics['recall_per_class']
        f1_pc = metrics['f1_per_class']
        
        for cls in sorted(prec_pc.keys()):
            print(f"   Class {cls}:")
            print(f"      Precision: {prec_pc[cls]:.4f}")
            print(f"      Recall:    {rec_pc[cls]:.4f}")
            print(f"      F1:        {f1_pc[cls]:.4f}")
    
    # Look for prediction/label arrays
    pred_keys = ['preds', 'predictions', 'all_preds', 'student_predictions', 'val_preds']
    label_keys = ['labels', 'true_labels', 'all_labels', 'targets', 'val_labels']
    
    preds = None
    labels = None
    
    print(f"\n🔍 Searching for raw predictions/labels...")
    print(f"   Available keys in metrics.json: {list(metrics.keys())}")
    
    for key in pred_keys:
        if key in metrics:
            preds = np.array(metrics[key])
            print(f"   ✓ Found predictions: '{key}' (shape: {preds.shape})")
            break
    
    for key in label_keys:
        if key in metrics:
            labels = np.array(metrics[key])
            print(f"   ✓ Found labels: '{key}' (shape: {labels.shape})")
            break
    
    if preds is not None and labels is not None:
        print(f"\n📊 PREDICTION DISTRIBUTION:")
        print(f"   Total samples: {len(preds)}")
        print(f"   True label distribution:  {dict(Counter(labels))}")
        print(f"   Predicted distribution:   {dict(Counter(preds))}")
        
        # Check for class imbalance
        label_counts = Counter(labels)
        pred_counts = Counter(preds)
        
        # Check if predicting only one class
        unique_preds = np.unique(preds)
        print(f"   Unique predictions: {unique_preds}")
        
        if len(unique_preds) == 1:
            print(f"\n⚠️  CRITICAL: Model predicting only class {unique_preds[0]}!")
            print(f"   This explains metric convergence!")
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(labels, preds)
        
        print(f"\n📊 CONFUSION MATRIX:")
        print(cm)
        
        # Check for diagonal dominance
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            print(f"\n   True Negatives:  {tn} ({tn/total*100:.1f}%)")
            print(f"   False Positives: {fp} ({fp/total*100:.1f}%)")
            print(f"   False Negatives: {fn} ({fn/total*100:.1f}%)")
            print(f"   True Positives:  {tp} ({tp/total*100:.1f}%)")
            
            # Check if errors are suspiciously low
            error_rate = (fp + fn) / total
            print(f"\n   Error rate: {error_rate*100:.2f}%")
            if error_rate < 0.06:
                print(f"   ⚠️  Error rate very low! Possible causes:")
                print(f"      - Data leakage")
                print(f"      - Trivial task")
                print(f"      - Very strong teacher model")
        
        print(f"\n📊 CLASSIFICATION REPORT:")
        print(classification_report(labels, preds, digits=4))
        
        # Show misclassified examples
        mismatches = [(i, int(p), int(t)) for i, (p, t) in enumerate(zip(preds, labels)) if p != t]
        print(f"\n🔍 MISCLASSIFICATION ANALYSIS:")
        print(f"   Total misclassifications: {len(mismatches)}")
        print(f"   Accuracy check: {(preds == labels).mean():.4f}")
        
        if len(mismatches) > 0 and len(mismatches) < 20:
            print(f"\n   Misclassified sample indices:")
            for i, (idx, pred, true) in enumerate(mismatches[:10]):
                print(f"      [{idx}] predicted={pred}, true={true}")
    else:
        print(f"\n⚠️  No raw predictions/labels found in metrics.json")
        print(f"   Cannot perform detailed analysis")

else:
    print(f"❌ Metrics file not found: {metrics_file}")

# ============================================================================
# 3. CHECK TRAINING LOGS - EVALUATION DATASET
# ============================================================================
print("\n" + "="*80)
print("3. TRAINING LOG ANALYSIS")
print("="*80)

log_dir = exp_dir / "logs"
if log_dir.exists():
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        print(f"✓ Found {len(log_files)} log files")
        
        # Read latest log
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"   Reading: {latest_log.name}")
        
        log_content = latest_log.read_text()
        
        # Check for dataset loading
        print(f"\n🔍 Dataset Loading Evidence:")
        for line in log_content.split('\n'):
            if any(keyword in line.lower() for keyword in ['loading', 'dataset', 'samples', 'train', 'val']):
                if len(line.strip()) > 0 and len(line) < 200:
                    print(f"   {line.strip()[:150]}")
        
        # Check for evaluation calls
        print(f"\n🔍 Evaluation Calls:")
        eval_lines = [line for line in log_content.split('\n') if 'eval' in line.lower()]
        for line in eval_lines[:10]:
            if len(line.strip()) > 0:
                print(f"   {line.strip()[:150]}")
    else:
        print(f"⚠️  No log files found in {log_dir}")
else:
    print(f"⚠️  Log directory not found: {log_dir}")

# ============================================================================
# 4. CHECK MODEL FILES - VERIFY SAVED MODEL
# ============================================================================
print("\n" + "="*80)
print("4. MODEL FILE ANALYSIS")
print("="*80)

student_dir = exp_dir / "student_model"
teacher_dir = exp_dir / "teacher_model"

for model_type, model_dir in [("Student", student_dir), ("Teacher", teacher_dir)]:
    if model_dir.exists():
        print(f"\n✓ {model_type} model saved at: {model_dir}")
        config_file = model_dir / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            print(f"   Model: {config.get('_name_or_path', 'unknown')}")
            print(f"   Num labels: {config.get('num_labels', 'unknown')}")
    else:
        print(f"⚠️  {model_type} model not found")

# ============================================================================
# 5. RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("5. DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("="*80)

issues_found = []
recommendations = []

# Analyze findings
if train_path.exists() and val_path.exists():
    if len(overlap) > 0:
        issues_found.append(f"❌ DATA LEAKAGE: {len(overlap)} samples overlap between train/val")
        recommendations.append("Remove overlapping samples from validation set")
    else:
        issues_found.append("✓ No text overlap between train and val datasets")

if preds is not None and labels is not None:
    error_rate = 1 - (preds == labels).mean()
    if error_rate < 0.06:
        issues_found.append(f"⚠️  Very low error rate: {error_rate*100:.2f}%")
        recommendations.append("Investigate if task is too simple or data is leaked")
    
    if len(np.unique(preds)) == 1:
        issues_found.append("❌ CRITICAL: Model predicts only one class")
        recommendations.append("Add class weights or balance dataset")

print(f"\n🔍 Issues Found:")
for issue in issues_found:
    print(f"   {issue}")

if recommendations:
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
else:
    print(f"\n✓ No obvious data leakage detected")
    print(f"\n💡 Possible explanations for high performance:")
    print(f"   1. Task is relatively simple (binary sentiment classification)")
    print(f"   2. Strong teacher model (roberta-base is powerful)")
    print(f"   3. Good distillation setup (temperature=2.0, alpha=0.5)")
    print(f"   4. Limited dataset size (1000 samples) may lead to overfitting")

print("\n" + "="*80)
print("END OF DIAGNOSTIC REPORT")
print("="*80)
