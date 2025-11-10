"""
Repair Teacher Model Labels
Adds or corrects label2id and id2label mappings in existing model checkpoints.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import shutil
from datetime import datetime


def repair_model_labels(
    model_path: str,
    label2id: dict = None,
    id2label: dict = None,
    backup: bool = True
):
    """
    Add or correct label mappings in a model's config.json.
    
    Args:
        model_path: Path to model directory
        label2id: Dictionary mapping label names to IDs (default: binary sentiment)
        id2label: Dictionary mapping IDs to label names (default: binary sentiment)
        backup: Whether to create a backup of the original config
    """
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        print(f"❌ Error: config.json not found at {config_path}")
        return False
    
    # Default to binary sentiment classification
    if label2id is None:
        label2id = {"negative": 0, "positive": 1}
    if id2label is None:
        id2label = {"0": "negative", "1": "positive"}
    
    print(f"\n{'='*60}")
    print("🔧 TEACHER MODEL LABEL REPAIR UTILITY")
    print(f"{'='*60}\n")
    print(f"Model Path: {model_path}")
    print(f"Config File: {config_path}")
    
    # Load existing config
    print("\n📖 Loading existing config...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"   Current label2id: {config.get('label2id', 'NOT SET')}")
    print(f"   Current id2label: {config.get('id2label', 'NOT SET')}")
    
    # Create backup if requested
    if backup:
        backup_path = model_path / f"config.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n💾 Creating backup: {backup_path.name}")
        shutil.copy(config_path, backup_path)
    
    # Update config
    print(f"\n🔄 Updating label mappings...")
    config['label2id'] = label2id
    config['id2label'] = id2label
    
    print(f"   New label2id: {label2id}")
    print(f"   New id2label: {id2label}")
    
    # Save updated config
    print(f"\n💾 Saving updated config...")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Successfully updated {config_path}")
    print(f"\n{'='*60}")
    print("⚠️  IMPORTANT NOTES")
    print(f"{'='*60}\n")
    print("1. This repair adds label mappings to the config ONLY.")
    print("2. If the model's classification head was trained with misaligned labels,")
    print("   you may need to retrain the model for full accuracy recovery.")
    print("3. The updated config will be used in future model loads.")
    print("4. Re-run the comparison to see if accuracy improves.\n")
    
    if backup:
        print(f"💡 To revert: mv {backup_path} {config_path}\n")
    
    return True


def repair_experiment(experiment_dir: str):
    """Repair both teacher and student models in an experiment directory."""
    experiment_dir = Path(experiment_dir)
    
    print(f"\n{'='*60}")
    print(f"🔧 REPAIRING EXPERIMENT: {experiment_dir.name}")
    print(f"{'='*60}\n")
    
    teacher_path = experiment_dir / "teacher_model"
    student_path = experiment_dir / "student_model"
    
    success = True
    
    if teacher_path.exists():
        print("🔄 Repairing teacher model...")
        success &= repair_model_labels(str(teacher_path))
    else:
        print(f"⚠️  Teacher model not found at {teacher_path}")
    
    if student_path.exists():
        print("\n🔄 Repairing student model...")
        success &= repair_model_labels(str(student_path))
    else:
        print(f"⚠️  Student model not found at {student_path}")
    
    if success:
        print(f"\n{'='*60}")
        print("✅ EXPERIMENT REPAIRED")
        print(f"{'='*60}\n")
        print("Next steps:")
        print(f"1. Re-run the comparison:")
        print(f"   python3 examples/compare_teacher_student.py --exp {experiment_dir} --tokenizer-mode separate\n")
        print(f"2. Check if teacher accuracy improves")
        print(f"3. If accuracy is still poor, the teacher may need retraining\n")
    
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair teacher model label mappings")
    parser.add_argument("--model", help="Path to model directory")
    parser.add_argument("--exp", help="Path to experiment directory (repairs both teacher and student)")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating config backup")
    parser.add_argument("--label2id", type=json.loads, 
                       help='Custom label2id mapping as JSON string (default: {"negative": 0, "positive": 1})')
    parser.add_argument("--id2label", type=json.loads,
                       help='Custom id2label mapping as JSON string (default: {"0": "negative", "1": "positive"})')
    
    args = parser.parse_args()
    
    if args.exp:
        repair_experiment(args.exp)
    elif args.model:
        repair_model_labels(
            args.model,
            label2id=args.label2id,
            id2label=args.id2label,
            backup=not args.no_backup
        )
    else:
        parser.print_help()
        print("\n❌ Error: Must specify either --model or --exp")
        sys.exit(1)
