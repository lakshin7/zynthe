"""
Compare Teacher and Student Models
Run comprehensive comparison between trained teacher and student models.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.dataloaders import get_imdb_dataloaders
from evaluation.model_comparison import ModelComparator


def main():
    """Run teacher vs student comparison."""
    
    # Configuration
    EXPERIMENT_DIR = "experiments/20250926T052444Z_ba9f0508"  # Latest experiment
    TEACHER_PATH = f"{EXPERIMENT_DIR}/teacher_model"
    STUDENT_PATH = f"{EXPERIMENT_DIR}/student_model"
    OUTPUT_DIR = f"{EXPERIMENT_DIR}/comparison"
    
    # Check if paths exist
    if not Path(TEACHER_PATH).exists():
        print(f"❌ Teacher model not found at: {TEACHER_PATH}")
        print("💡 Make sure the teacher model is saved during training.")
        return
    
    if not Path(STUDENT_PATH).exists():
        print(f"❌ Student model not found at: {STUDENT_PATH}")
        print("💡 Make sure the student model is saved during training.")
        return
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    print("\n" + "="*60)
    print("🎯 TEACHER vs STUDENT MODEL COMPARISON")
    print("="*60)
    print(f"\nExperiment: {EXPERIMENT_DIR}")
    print(f"Teacher:    {TEACHER_PATH}")
    print(f"Student:    {STUDENT_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Device:     {device}")
    
    # Initialize comparator
    print("\n" + "-"*60)
    comparator = ModelComparator(
        teacher_path=TEACHER_PATH,
        student_path=STUDENT_PATH,
        device=device,
        use_same_tokenizer=True
    )
    
    # Load dataset
    print("\n" + "-"*60)
    print("📚 Loading evaluation dataset...")
    
    try:
        train_loader, val_loader = get_imdb_dataloaders(
            train_path="data/imdb_train.jsonl",
            val_path="data/imdb_val.jsonl",
            tokenizer=comparator.tokenizer,
            batch_size=8,
            max_length=128
        )
        print(f"✅ Loaded validation set: {len(val_loader.dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Run comparison
    print("\n" + "-"*60)
    teacher_results, student_results = comparator.compare_models(val_loader)
    
    # Generate visualizations
    print("\n" + "-"*60)
    comparator.visualize_comparison(
        teacher_results,
        student_results,
        save_dir=OUTPUT_DIR,
        show_plots=False
    )
    
    # Save results
    print("\n" + "-"*60)
    comparator.save_results(
        teacher_results,
        student_results,
        save_dir=OUTPUT_DIR
    )
    
    # Generate report
    print("\n" + "-"*60)
    comparator.generate_report(
        teacher_results,
        student_results,
        save_dir=OUTPUT_DIR
    )
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 COMPARISON COMPLETE!")
    print("="*60)
    
    print("\n📊 Summary:")
    print(f"   Teacher Accuracy:  {teacher_results['accuracy']:.4f}")
    print(f"   Student Accuracy:  {student_results['accuracy']:.4f}")
    print(f"   Accuracy Drop:     {(teacher_results['accuracy'] - student_results['accuracy'])*100:.2f}%")
    print(f"   Compression Ratio: {student_results['compression_ratio']:.2f}x")
    print(f"   Size Reduction:    {(1 - 1/student_results['compression_ratio'])*100:.1f}%")
    
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
    print("\n🔍 Check these files:")
    print(f"   📈 {OUTPUT_DIR}/metrics_comparison.png")
    print(f"   🎯 {OUTPUT_DIR}/confusion_matrices_comparison.png")
    print(f"   📊 {OUTPUT_DIR}/per_class_comparison.png")
    print(f"   ⚡ {OUTPUT_DIR}/efficiency_comparison.png")
    print(f"   📋 {OUTPUT_DIR}/comparison_table.png")
    print(f"   💾 {OUTPUT_DIR}/comparison_results.json")
    print(f"   📝 {OUTPUT_DIR}/COMPARISON_REPORT.md")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
