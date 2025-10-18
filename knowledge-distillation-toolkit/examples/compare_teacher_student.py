"""
Compare Teacher and Student Models
Run comprehensive comparison between trained teacher and student models.
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.dataloaders import get_imdb_dataloaders, JsonlDataset
from torch.utils.data import DataLoader
from evaluation.model_comparison import ModelComparator


def main():
    """Run teacher vs student comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", dest="experiment_dir", type=str, required=True,
                        help="Path to experiment directory (e.g., experiments/20251015T064334Z_79be4fb2)")
    parser.add_argument(
        "--tokenizer-mode",
        choices=["separate", "student", "teacher"],
        default="separate",
        help="Which tokenizer(s) to use for evaluation: separate (recommended), or force student/teacher for both."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    EXPERIMENT_DIR = args.experiment_dir
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
        use_same_tokenizer=False  # always load both to allow flexible evaluation
    )
    
    # Load dataset
    print("\n" + "-"*60)
    print("📚 Loading evaluation dataset...")
    
    VAL_PATH = "data/imdb_val.jsonl"
    try:
        if args.tokenizer_mode == "separate":
            print("🔀 Using separate tokenizers for teacher and student (recommended).")
            teacher_val_loader = DataLoader(
                JsonlDataset(VAL_PATH, comparator.teacher_tokenizer, max_length=args.max_length),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            student_val_loader = DataLoader(
                JsonlDataset(VAL_PATH, comparator.student_tokenizer, max_length=args.max_length),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            print(f"✅ Loaded validation set: {len(student_val_loader.dataset)} samples")
        else:
            chosen = comparator.student_tokenizer if args.tokenizer_mode == "student" else comparator.teacher_tokenizer
            print(f"🔁 Using {args.tokenizer_mode} tokenizer for BOTH models (may skew results).")
            _, val_loader = get_imdb_dataloaders(
                train_path="data/imdb_train.jsonl",
                val_path=VAL_PATH,
                tokenizer=chosen,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            print(f"✅ Loaded validation set: {len(val_loader.dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Run comparison
    print("\n" + "-"*60)
    if args.tokenizer_mode == "separate":
        teacher_results, student_results = comparator.compare_models_dual_loaders(teacher_val_loader, student_val_loader)
    else:
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

    # Export expected artifacts (latency CSV, metrics JSON, compression summary, final_report.pdf)
    comparator.export_expected_artifacts(
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
