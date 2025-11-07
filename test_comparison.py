#!/usr/bin/env python3
"""
Test the teacher-student comparison feature on existing experiment results.
This script can run the comparison on already-trained models without retraining.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from evaluation.model_comparison import ModelComparator
from data.dataloaders import create_dataloaders
from transformers import AutoTokenizer


def test_comparison_on_experiment(experiment_dir: str):
    """
    Test comparison on an existing experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/20251018T100839Z_9b3dfc41)
    """
    exp_path = Path(experiment_dir)
    teacher_path = exp_path / "teacher_model"
    student_path = exp_path / "student_model"
    comparison_dir = exp_path / "comparison"
    
    print("\n" + "="*70)
    print("🧪 TESTING TEACHER-STUDENT COMPARISON")
    print("="*70)
    print(f"\nExperiment: {exp_path}")
    print(f"Teacher:    {teacher_path}")
    print(f"Student:    {student_path}")
    print(f"Output:     {comparison_dir}")
    
    # Check if models exist
    if not teacher_path.exists():
        print(f"\n❌ Teacher model not found at: {teacher_path}")
        print("Please run training first to generate the teacher model.")
        return False
    
    if not student_path.exists():
        print(f"\n❌ Student model not found at: {student_path}")
        print("Please run training first to generate the student model.")
        return False
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"\n🍎 Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"\n🚀 Using CUDA")
    else:
        device = "cpu"
        print(f"\n💻 Using CPU")
    
    try:
        # Initialize comparator
        print("\n" + "-"*70)
        print("📦 Loading models...")
        comparator = ModelComparator(
            teacher_path=str(teacher_path),
            student_path=str(student_path),
            device=device,
            use_same_tokenizer=True
        )
        
        # Load validation dataset
        print("\n" + "-"*70)
        print("📚 Loading validation dataset...")
        
        # Try to load tokenizer from student model
        tokenizer = AutoTokenizer.from_pretrained(str(student_path))
        
        # Create dataloader
        config = {
            "data": {
                "train_path": "data/imdb_train.jsonl",
                "val_path": "data/imdb_val.jsonl"
            },
            "train": {
                "batch_size": 8
            },
            "model": {
                "max_length": 128
            }
        }
        
        _, val_loader = create_dataloaders(config, tokenizer)
        print(f"✅ Loaded {len(val_loader.dataset)} validation samples")
        
        # Run comparison
        print("\n" + "-"*70)
        print("🔬 Running model comparison...")
        teacher_results, student_results = comparator.compare_models(val_loader)
        
        # Generate visualizations
        print("\n" + "-"*70)
        print("📈 Generating visualizations...")
        comparator.visualize_comparison(
            teacher_results,
            student_results,
            save_dir=str(comparison_dir),
            show_plots=False
        )
        
        # Save results
        print("\n" + "-"*70)
        print("💾 Saving results...")
        comparator.save_results(
            teacher_results,
            student_results,
            save_dir=str(comparison_dir)
        )
        
        # Generate report
        print("\n" + "-"*70)
        print("📄 Generating report...")
        comparator.generate_report(
            teacher_results,
            student_results,
            save_dir=str(comparison_dir)
        )
        
        # Export additional artifacts
        print("\n" + "-"*70)
        print("📦 Exporting additional artifacts...")
        comparator.export_expected_artifacts(
            teacher_results,
            student_results,
            save_dir=str(comparison_dir)
        )
        
        # Print summary
        print("\n" + "="*70)
        print("✅ COMPARISON COMPLETE!")
        print("="*70)
        print(f"\n📊 Results Summary:")
        print(f"   Teacher Accuracy:  {teacher_results['accuracy']:.4f}")
        print(f"   Student Accuracy:  {student_results['accuracy']:.4f}")
        print(f"   Accuracy Drop:     {(teacher_results['accuracy'] - student_results['accuracy']):.4f}")
        print(f"   Compression Ratio: {comparator.compression_ratio:.2f}x")
        print(f"   Teacher Params:    {comparator.teacher_params:,}")
        print(f"   Student Params:    {comparator.student_params:,}")
        
        print(f"\n📁 Results saved to: {comparison_dir}")
        print("\nGenerated files:")
        print("   ✓ comparison_results.json")
        print("   ✓ COMPARISON_REPORT.md")
        print("   ✓ metrics_comparison.png")
        print("   ✓ confusion_matrices_comparison.png")
        print("   ✓ per_class_comparison.png")
        print("   ✓ efficiency_comparison.png")
        print("   ✓ comparison_table.png")
        print("   ✓ latency_results.csv")
        print("   ✓ compression_summary.txt")
        print("   ✓ teacher_metrics.json")
        print("   ✓ student_metrics.json")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test teacher-student comparison on existing experiment"
    )
    parser.add_argument(
        "--exp",
        "--experiment",
        dest="experiment_dir",
        type=str,
        default="experiments/20251018T100839Z_9b3dfc41",
        help="Path to experiment directory"
    )
    
    args = parser.parse_args()
    
    success = test_comparison_on_experiment(args.experiment_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
