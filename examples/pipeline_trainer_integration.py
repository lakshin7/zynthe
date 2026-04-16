"""
Example: Using Pipeline with Trainer
====================================

This example shows how to use the new pipeline system with the Trainer class.

Three approaches are demonstrated:
1. Providing a pre-built pipeline to Trainer
2. Using configuration to auto-build pipeline
3. Legacy distiller mode (backward compatible)
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

# Import Zynthe components
from core.pipelines import PipelineBuilder
from training.trainer import Trainer


# ============================================================================
# EXAMPLE 1: Provide Pre-built Pipeline to Trainer
# ============================================================================

def example_1_prebuild_pipeline():
    """Build pipeline first, then pass to Trainer."""
    
    # Load models
    teacher = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    student = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # BUILD PIPELINE using fluent API
    pipeline = PipelineBuilder() \
        .add_stage('logit_distillation', weight=0.7) \
            .add_distiller('kd_hinton', temperature=4.0, alpha=0.8) \
        .add_stage('feature_matching', weight=0.3) \
            .add_distiller('feature', layers=[2, 4]) \
        .with_mode('sequential') \
        .build(teacher, student, device)
    
    print(f"✅ Built pipeline: {pipeline}")
    
    # Configuration (rest of the config, no distillation section needed)
    config = {
        'train': {
            'num_epochs': 3,
            'learning_rate': 5e-5,
            'use_amp': True,
        }
    }
    
    # Create Trainer with pipeline parameter
    trainer = Trainer(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        config=config,
        device=device,
        experiment_dir="./experiments",
        pipeline=pipeline  # NEW: Pass pipeline directly
    )
    
    print("✅ Trainer created with pipeline")
    
    # Use trainer as normal
    # trainer.train(train_loader, val_loader)
    
    return trainer


# ============================================================================
# EXAMPLE 2: Configuration-based Pipeline (Auto-build)
# ============================================================================

def example_2_config_pipeline():
    """Let Trainer build pipeline from configuration."""
    
    # Load models
    teacher = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    student = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration with pipeline section
    config = {
        'train': {
            'num_epochs': 3,
            'learning_rate': 5e-5,
            'use_amp': True,
        },
        'distillation': {
            'pipeline': {
                'type': 'multi_stage',  # This triggers pipeline mode
                'mode': 'hybrid',
                'stages': [
                    {
                        'name': 'logit_stage',
                        'weight': 0.6,
                        'distillers': [
                            {
                                'type': 'kd_hinton',
                                'config': {
                                    'temperature': 4.0,
                                    'alpha': 0.7
                                }
                            }
                        ]
                    },
                    {
                        'name': 'feature_stage',
                        'weight': 0.4,
                        'mode': 'parallel',
                        'distillers': [
                            {
                                'type': 'feature',
                                'config': {
                                    'layers': [2, 4]
                                }
                            },
                            # Can add more distillers here
                        ]
                    }
                ]
            }
        }
    }
    
    # Create Trainer - it will auto-build pipeline from config
    trainer = Trainer(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        config=config,
        device=device,
        experiment_dir="./experiments",
        # No pipeline parameter - Trainer builds from config
    )
    
    print("✅ Trainer created with auto-built pipeline from config")
    print(f"   Pipeline: {trainer.pipeline}")
    
    # Use trainer as normal
    # trainer.train(train_loader, val_loader)
    
    return trainer


# ============================================================================
# EXAMPLE 3: Legacy Distiller Mode (Backward Compatible)
# ============================================================================

def example_3_legacy_distiller():
    """Traditional single distiller mode - fully backward compatible."""
    
    # Load models
    teacher = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    student = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Traditional configuration (no pipeline section)
    config = {
        'train': {
            'num_epochs': 3,
            'learning_rate': 5e-5,
            'use_amp': True,
        },
        'distillation': {
            'method': 'kd_hinton',  # Traditional single distiller
            'temperature': 4.0,
            'alpha': 0.7,
        }
    }
    
    # Create Trainer - uses legacy distiller mode
    trainer = Trainer(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        config=config,
        device=device,
        experiment_dir="./experiments",
    )
    
    print("✅ Trainer created with legacy distiller mode")
    print(f"   Distiller: {trainer.distiller}")
    
    # Use trainer as normal - no changes needed
    # trainer.train(train_loader, val_loader)
    
    return trainer


# ============================================================================
# EXAMPLE 4: Dynamic Pipeline with Conditions
# ============================================================================

def example_4_conditional_pipeline():
    """Advanced: Conditional routing in pipeline."""
    
    teacher = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    student = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build pipeline with conditional execution
    # (Note: conditions require custom functions, shown as example)
    def early_training_condition(batch, outputs):
        """Example condition: only run early stages."""
        # In practice, you'd check epoch number or other state
        return True
    
    pipeline = PipelineBuilder() \
        .add_stage('early_stage', weight=1.0) \
            .add_distiller('kd_hinton', temperature=4.0) \
        .add_stage('late_stage', weight=0.5) \
            .add_distiller('feature', layers=[4, 6]) \
        .with_mode('sequential') \
        .build(teacher, student, device)
    
    # Add condition programmatically (not in fluent API yet)
    # pipeline.stages[1].condition = lambda b, o: epoch > 5
    
    config = {
        'train': {
            'num_epochs': 10,
            'learning_rate': 5e-5,
        }
    }
    
    trainer = Trainer(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        config=config,
        device=device,
        experiment_dir="./experiments",
        pipeline=pipeline
    )
    
    print("✅ Trainer created with conditional pipeline")
    
    return trainer


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXAMPLE 1: Pre-built Pipeline")
    print("=" * 70)
    trainer1 = example_1_prebuild_pipeline()
    print()
    
    print("=" * 70)
    print("EXAMPLE 2: Configuration-based Pipeline")
    print("=" * 70)
    trainer2 = example_2_config_pipeline()
    print()
    
    print("=" * 70)
    print("EXAMPLE 3: Legacy Distiller Mode")
    print("=" * 70)
    trainer3 = example_3_legacy_distiller()
    print()
    
    print("=" * 70)
    print("EXAMPLE 4: Conditional Pipeline")
    print("=" * 70)
    trainer4 = example_4_conditional_pipeline()
    print()
    
    print("✅ All examples completed successfully!")
    print("\nKey Takeaways:")
    print("  1. New pipeline mode is fully integrated with Trainer")
    print("  2. Three ways to use it: pre-built, config-based, or legacy")
    print("  3. Backward compatible - existing code still works")
    print("  4. Flexible execution modes: sequential, parallel, hybrid")
