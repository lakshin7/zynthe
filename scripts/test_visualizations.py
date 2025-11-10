#!/usr/bin/env python3
"""
Test Visualization Pipeline - Direct Trainer Invocation
Bypasses MultiStageDistiller to use the enhanced Trainer with micro-series and confusion matrix generation.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root
import os
os.chdir(project_root)

def run_trainer_test():
    """Run distillation using enhanced Trainer with all visualizations."""
    from core.config.config_manager import ConfigManager
    from core.models.model_loader import load_models
    from data.dataloaders import create_dataloaders
    from training.trainer import Trainer
    import torch
    
    print("\n" + "="*70)
    print("🎨 VISUALIZATION PIPELINE TEST")
    print("="*70 + "\n")
    
    # Load config
    config_path = "configs/test_visualization.yaml"
    print(f"Loading config: {config_path}")
    cm = ConfigManager(config_path)
    cfg = cm.resolved_config
    
    # Load models
    print("\n📥 Loading models...")
    teacher, student, tokenizer = load_models(cm, cm.device())
    print(f"✓ Teacher: {teacher.config._name_or_path}")
    print(f"✓ Student: {student.config._name_or_path}")
    
    # Load dataloaders
    print("\n📚 Loading dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg, tokenizer)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\n⚙️  Initializing Trainer...")
    trainer = Trainer(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        config=cfg,
        device=cm.device(),
        experiment_dir=cm.experiment_dir
    )
    print(f"✓ Experiment dir: {cm.experiment_dir}")
    
    # Run training
    print("\n🚀 Starting training...")
    print("=" * 70)
    trainer.fit(train_loader, val_loader)
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\nExperiment directory: {cm.experiment_dir}")
    print("\nExpected artifacts:")
    print("  📊 training_curves.png (student)")
    print("  📊 teacher_training_curves.png")
    print("  📈 student_epoch*_micro.png")
    print("  📈 student_epoch*_eval_micro.png")
    print("  📈 teacher_epoch*_train_micro.png")
    print("  📈 teacher_epoch*_eval_micro.png")
    print("  🎯 student_confusion/confusion_matrix.png")
    print("  🎯 teacher_confusion/confusion_matrix.png")
    print("  📋 extended_metrics.json")
    print("  📋 EXPERIMENT_SUMMARY.md")
    
    # Run artifact check
    print("\n" + "="*70)
    print("🔍 ARTIFACT VALIDATION")
    print("="*70 + "\n")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/check_artifacts.py", str(cm.experiment_dir)],
        capture_output=False
    )
    
    return result.returncode


if __name__ == "__main__":
    try:
        exit_code = run_trainer_test()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
