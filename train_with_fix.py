#!/usr/bin/env python3

import logging
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config.config_manager import ConfigManager
from core.models.model_loader import load_models
from data.dataloaders import create_dataloaders
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_with_fix")
    
    print("\n" + "="*70)
    print("Knowledge Distillation Training - Fixed Trainer")
    print("="*70 + "\n")
    
    # Load configuration
    config_path = "configs/retrain_teacher.yaml"
    print(f"Loading configuration: {config_path}")
    cfg = ConfigManager(config_path=config_path)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"Device: {device}\n")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    experiment_id = os.urandom(4).hex()
    experiment_dir = Path("experiments") / f"{timestamp}_{experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}\n")
    
    # Load models
    print("="*70)
    print("Loading Models")
    print("="*70)
    teacher, student, tokenizer = load_models(cfg, device=device)
    
    print(f"\nTeacher: {cfg.model.teacher.name}")
    print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    print(f"\nStudent: {cfg.model.student.name}")
    print(f"  Parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    train_loader, val_loader = create_dataloaders(cfg, tokenizer)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\n" + "="*70)
    print("Initializing Trainer")
    print("="*70)
    trainer = Trainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        experiment_dir=experiment_dir
    )
    
    print(f"\nTrainer configuration:")
    print(f"  Teacher epochs: {cfg.train.get('teacher_epochs', 2)}")
    print(f"  Distillation epochs: {cfg.train.epochs}")
    
    # Train
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    try:
        trainer.fit()
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        
        # Save models
        teacher_path = experiment_dir / "teacher_model"
        student_path = experiment_dir / "student_model"
        
        trainer.teacher.save_pretrained(teacher_path)
        trainer.student.save_pretrained(student_path)
        tokenizer.save_pretrained(student_path)
        
        print(f"\nModels saved:")
        print(f"   Teacher: {teacher_path}")
        print(f"   Student: {student_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
        return 1
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
