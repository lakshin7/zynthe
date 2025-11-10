#!/usr/bin/env python3
"""
Test Script: Data Validation, HuggingFace Integration, and Training Health
===========================================================================

This script demonstrates the new features:
1. Automatic data leakage detection
2. HuggingFace dataset loading
3. Overfitting/underfitting detection
4. Training health monitoring

Run this to test the new functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.utils.data_validator import DataValidator, DataLeakageDetector, OverfitUnderfitDetector
from core.utils.hf_dataset_loader import HuggingFaceDatasetLoader
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)


def test_data_leakage_detection():
    """Test data leakage detector with current dataset."""
    print("\n" + "="*80)
    print("TEST 1: DATA LEAKAGE DETECTION")
    print("="*80)
    
    # Load existing datasets
    train_path = Path("data/imdb_train.jsonl")
    val_path = Path("data/imdb_val.jsonl")
    
    if not train_path.exists() or not val_path.exists():
        print("⚠️  Dataset files not found. Skipping test.")
        return
    
    # Load data
    train_data = []
    val_data = []
    
    with open(train_path) as f:
        for line in f:
            train_data.append(json.loads(line))
    
    with open(val_path) as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Validate
    results = DataValidator.validate_dataset_split(
        train_data,
        val_data,
        text_key='text',
        label_key='label'
    )
    
    # Save report
    report_path = Path("data_validation_test_report.json")
    DataValidator.save_validation_report(results, report_path)
    
    print(f"\n✓ Validation complete. Report saved to: {report_path}")
    
    if results['validation_passed']:
        print("✓ Dataset validation PASSED")
    else:
        print("❌ Dataset validation FAILED")
        print(f"   Errors: {results['errors']}")


def test_overfitting_detection():
    """Test overfitting/underfitting detector."""
    print("\n" + "="*80)
    print("TEST 2: OVERFITTING/UNDERFITTING DETECTION")
    print("="*80)
    
    # Simulate training curves
    
    # Scenario 1: Healthy training
    print("\nScenario 1: Healthy Training")
    train_losses = [0.6, 0.5, 0.4, 0.35, 0.32, 0.30]
    val_losses = [0.65, 0.55, 0.47, 0.42, 0.40, 0.39]
    
    analysis = OverfitUnderfitDetector.analyze_training_curves(train_losses, val_losses)
    print(f"  Status: {analysis['status']}")
    print(f"  Confidence: {analysis['confidence']*100:.1f}%")
    
    # Scenario 2: Overfitting
    print("\nScenario 2: Overfitting")
    train_losses = [0.6, 0.4, 0.25, 0.15, 0.08, 0.04]
    val_losses = [0.62, 0.48, 0.42, 0.50, 0.58, 0.65]
    
    analysis = OverfitUnderfitDetector.analyze_training_curves(train_losses, val_losses)
    print(f"  Status: {analysis['status']}")
    print(f"  Confidence: {analysis['confidence']*100:.1f}%")
    print(f"  Recommendations:")
    for rec in analysis['recommendations']:
        print(f"    - {rec}")
    
    # Scenario 3: Underfitting
    print("\nScenario 3: Underfitting")
    train_losses = [0.8, 0.75, 0.73, 0.72, 0.71, 0.70]
    val_losses = [0.82, 0.77, 0.75, 0.74, 0.73, 0.72]
    
    analysis = OverfitUnderfitDetector.analyze_training_curves(train_losses, val_losses)
    print(f"  Status: {analysis['status']}")
    print(f"  Confidence: {analysis['confidence']*100:.1f}%")
    print(f"  Recommendations:")
    for rec in analysis['recommendations']:
        print(f"    - {rec}")
    
    # Scenario 4: Suspicious (data leakage)
    print("\nScenario 4: Suspicious (Possible Data Leakage)")
    train_losses = [0.6, 0.5, 0.4, 0.35, 0.32, 0.30]
    val_losses = [0.55, 0.45, 0.35, 0.30, 0.27, 0.25]  # Val < Train!
    
    analysis = OverfitUnderfitDetector.analyze_training_curves(train_losses, val_losses)
    print(f"  Status: {analysis['status']}")
    print(f"  Recommendations:")
    for rec in analysis['recommendations']:
        print(f"    - {rec}")


def test_hf_dataset_loader():
    """Test HuggingFace dataset loader."""
    print("\n" + "="*80)
    print("TEST 3: HUGGINGFACE DATASET LOADER")
    print("="*80)
    
    # List available datasets
    print("\nAvailable datasets in catalog:")
    catalog = HuggingFaceDatasetLoader.list_available_datasets()
    for category, datasets in catalog.items():
        print(f"\n{category.upper()}:")
        for dataset in datasets:
            info = HuggingFaceDatasetLoader.get_dataset_info(dataset)
            print(f"  - {dataset}: {info['task']}")
    
    # Test loading a small dataset
    print("\n" + "-"*80)
    print("Testing dataset loading (SST-2 - 100 samples)...")
    print("-"*80)
    
    try:
        output_dir = Path("data/test_hf_dataset")
        paths = HuggingFaceDatasetLoader.prepare_dataset(
            'glue/sst2',
            output_dir=output_dir,
            max_samples=100,  # Small sample for testing
            text_col='sentence',
            label_col='label'
        )
        
        print(f"\n✓ Dataset downloaded and prepared:")
        for split_name, path in paths.items():
            print(f"  {split_name}: {path}")
        
        # Validate the new dataset
        print(f"\nValidating HuggingFace dataset...")
        
        train_data = []
        val_data = []
        
        if 'train' in paths and paths['train'].exists():
            with open(paths['train']) as f:
                for line in f:
                    train_data.append(json.loads(line))
        
        if 'val' in paths and paths['val'].exists():
            with open(paths['val']) as f:
                for line in f:
                    val_data.append(json.loads(line))
        
        if train_data and val_data:
            results = DataValidator.validate_dataset_split(
                train_data,
                val_data,
                text_key='text',
                label_key='label'
            )
            
            if results['validation_passed']:
                print("✓ HuggingFace dataset validation PASSED")
            else:
                print("❌ HuggingFace dataset validation FAILED")
    
    except Exception as e:
        print(f"⚠️  Could not download dataset (may need internet): {e}")
        print("   This is expected if offline or HuggingFace is unavailable")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ZYNTHE DATA VALIDATION & HEALTH MONITORING TEST SUITE")
    print("="*80)
    
    test_data_leakage_detection()
    test_overfitting_detection()
    test_hf_dataset_loader()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Check 'data_validation_test_report.json' for validation results")
    print("2. Try training with automatic validation enabled")
    print("3. Monitor 'training_health.json' during training")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
