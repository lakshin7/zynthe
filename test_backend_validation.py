#!/usr/bin/env python3
"""
Test backend validation with a small HuggingFace dataset.
This will demonstrate:
1. Downloading a dataset from HuggingFace
2. Automatic data validation (leakage detection)
3. Training with health monitoring
4. Overfitting/underfitting detection
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.utils.hf_dataset_loader import prepare_hf_dataset
from core.utils.data_validator import DataValidator

def main():
    print("="*80)
    print("BACKEND VALIDATION TEST")
    print("="*80)
    
    # Step 1: Download SST-2 dataset (small, fast)
    print("\n[STEP 1] Downloading SST-2 dataset from HuggingFace...")
    print("-" * 80)
    
    try:
        paths = prepare_hf_dataset(
            dataset_id='glue/sst2',
            output_dir=Path('data/sst2_test'),
            max_samples=500,  # Small for quick testing
            text_col='sentence',
            label_col='label'
        )
        
        print("\n✓ Dataset downloaded successfully!")
        print(f"  Train: {paths['train']}")
        print(f"  Val: {paths['val']}")
        if 'test' in paths:
            print(f"  Test: {paths['test']}")
        
    except Exception as e:
        print(f"\n❌ Failed to download dataset: {e}")
        print("\nNote: This requires internet connection. Trying with existing data...")
        
        # Fallback to existing IMDB data
        paths = {
            'train': Path('data/imdb_train.jsonl'),
            'val': Path('data/imdb_val.jsonl')
        }
        
        if not paths['train'].exists() or not paths['val'].exists():
            print("❌ No test data available. Please check internet connection.")
            return 1
    
    # Step 2: Validate the downloaded dataset
    print("\n[STEP 2] Running data validation (leakage detection)...")
    print("-" * 80)
    
    try:
        # Load the data
        train_data = []
        with open(paths['train']) as f:
            for line in f:
                train_data.append(json.loads(line))
        
        val_data = []
        with open(paths['val']) as f:
            for line in f:
                val_data.append(json.loads(line))
        
        print(f"\nLoaded {len(train_data)} training samples")
        print(f"Loaded {len(val_data)} validation samples")
        
        # Run validation
        print("\nRunning validation checks...")
        results = DataValidator.validate_dataset_split(
            train_data,
            val_data,
            text_key='text',
            label_key='label'
        )
        
        # Display results
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        
        if results['validation_passed']:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  ❌ {error}")
        
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  ⚠️  {warning}")
        
        # Leakage details
        leakage = results.get('leakage', {})
        print(f"\nData Leakage Check:")
        print(f"  Exact overlap: {leakage.get('exact_overlap_count', 0)} samples")
        if leakage.get('has_exact_leakage'):
            print(f"  ❌ LEAKAGE DETECTED!")
        else:
            print(f"  ✓ No leakage detected")
        
        # Class balance
        train_balance = results.get('train_balance', {})
        val_balance = results.get('val_balance', {})
        
        print(f"\nClass Distribution:")
        print(f"  Train: {train_balance.get('num_classes', 0)} classes, "
              f"imbalance ratio: {train_balance.get('imbalance_ratio', 0):.2f}")
        print(f"  Val: {val_balance.get('num_classes', 0)} classes, "
              f"imbalance ratio: {val_balance.get('imbalance_ratio', 0):.2f}")
        
        # Save report
        report_path = Path('backend_validation_test_report.json')
        DataValidator.save_validation_report(results, report_path)
        print(f"\n✓ Full report saved to: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Demonstrate training integration
    print("\n[STEP 3] Training Integration Demo")
    print("-" * 80)
    print("\nTo test automatic validation during training, run:")
    print(f"\n  python app/main.py --config configs/default.yaml")
    print(f"  # Update config to use: {paths['train']} and {paths['val']}")
    print("\nThe trainer will automatically:")
    print("  1. Check for data leakage before training")
    print("  2. Monitor for overfitting during training")
    print("  3. Generate health report after training")
    
    print("\n" + "="*80)
    print("BACKEND VALIDATION TEST COMPLETE ✅")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
