"""
Auto-Student Builder
===================

Main class for automatically generating student model architectures from teacher models.

Features:
- Automatic architecture sizing based on compression ratio
- Multiple sizing strategies (conservative, balanced, aggressive)
- Validation and memory feasibility checks
- Export to config format compatible with existing pipeline

Usage:
    builder = AutoStudentBuilder(teacher_name="bert-base-uncased")
    student_config = builder.generate(
        compression_ratio=0.5,
        strategy='balanced',
        validate=True
    )
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import yaml

from .heuristics import StudentSizingHeuristics
from .validator import StudentValidator

LOG = logging.getLogger(__name__)


class AutoStudentBuilder:
    """
    Automatically build student model architectures from teacher models.
    """
    
    def __init__(
        self,
        teacher_name: str,
        teacher_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize auto-student builder.
        
        Args:
            teacher_name: Name of teacher model (e.g., "bert-base-uncased")
            teacher_config: Optional teacher config dict (if not using known models)
            output_dir: Directory to save generated configs
        """
        if os.environ.get("ZYNTHE_ENABLE_AUTO_STUDENT", "0") != "1":
            raise RuntimeError(
                "AutoStudentBuilder is temporarily disabled in manual stabilization mode. "
                "Set ZYNTHE_ENABLE_AUTO_STUDENT=1 to re-enable explicitly."
            )

        self.teacher_name = teacher_name
        self.output_dir = Path(output_dir) if output_dir else Path("data/generated_students")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get or set teacher config
        if teacher_config:
            self.teacher_config = teacher_config
            LOG.info(f"Using provided teacher config: {teacher_config.get('total_params', 'unknown')} params")
        else:
            self.teacher_config = StudentSizingHeuristics.get_teacher_config(teacher_name)
            if not self.teacher_config:
                raise ValueError(
                    f"Unknown teacher model: {teacher_name}. "
                    f"Please provide teacher_config manually or use a known model."
                )
        
        LOG.info(f"Initialized AutoStudentBuilder for teacher: {teacher_name}")
        LOG.info(f"  Layers: {self.teacher_config['num_layers']}")
        LOG.info(f"  Hidden: {self.teacher_config['hidden_size']}")
        LOG.info(f"  Params: {self.teacher_config['total_params']:,}")
    
    def generate(
        self,
        compression_ratio: float = 0.5,
        strategy: str = 'balanced',
        validate: bool = True,
        auto_fix: bool = True,
        save: bool = False
    ) -> Dict[str, Any]:
        """
        Generate student architecture.
        
        Args:
            compression_ratio: Target size as fraction of teacher (0.5 = 50%)
            strategy: Sizing strategy ('conservative', 'balanced', 'aggressive')
            validate: Run validation checks
            auto_fix: Automatically fix validation issues
            save: Save config to file
            
        Returns:
            Student configuration dict
        """
        LOG.info(f"\n{'='*70}")
        LOG.info(f"Generating student architecture...")
        LOG.info(f"  Teacher: {self.teacher_name}")
        LOG.info(f"  Compression: {compression_ratio:.1%}")
        LOG.info(f"  Strategy: {strategy}")
        LOG.info(f"{'='*70}\n")
        
        # Generate student dimensions
        student_config = StudentSizingHeuristics.calculate_student_dimensions(
            self.teacher_config,
            compression_ratio=compression_ratio,
            strategy=strategy
        )
        
        # Add metadata
        student_config['teacher_name'] = self.teacher_name
        student_config['compression_ratio'] = compression_ratio
        student_config['strategy'] = strategy
        
        # Validate
        if validate:
            is_valid, issues = StudentValidator.validate(student_config, strict=False)
            
            if not is_valid:
                if auto_fix:
                    LOG.warning("Validation failed, attempting auto-fix...")
                    student_config = StudentValidator.suggest_fixes(student_config, issues)
                    
                    # Re-validate
                    is_valid, new_issues = StudentValidator.validate(student_config, strict=False)
                    if is_valid:
                        LOG.info("✓ Auto-fix successful!")
                    else:
                        LOG.error(f"✗ Auto-fix failed: {len(new_issues)} issues remain")
                        for issue in new_issues:
                            LOG.error(f"  - {issue}")
                else:
                    raise ValueError(f"Validation failed with {len(issues)} issues: {issues}")
        
        # Log final config
        LOG.info(f"\n{'='*70}")
        LOG.info(f"Generated Student Architecture:")
        LOG.info(f"{'='*70}")
        LOG.info(f"  Layers: {student_config['num_layers']} "
                f"({student_config['num_layers']/self.teacher_config['num_layers']:.1%} of teacher)")
        LOG.info(f"  Hidden Size: {student_config['hidden_size']} "
                f"({student_config['hidden_size']/self.teacher_config['hidden_size']:.1%} of teacher)")
        LOG.info(f"  Attention Heads: {student_config['num_attention_heads']}")
        LOG.info(f"  Intermediate Size: {student_config['intermediate_size']}")
        LOG.info(f"  Total Params: {student_config['total_params']:,} "
                f"({student_config['total_params']/self.teacher_config['total_params']:.1%} of teacher)")
        LOG.info(f"{'='*70}\n")
        
        # Save if requested
        if save:
            self.save_config(student_config)
        
        return student_config
    
    def generate_multiple(
        self,
        compression_ratios: List[float],
        strategies: Optional[List[str]] = None,
        save: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple student candidates.
        
        Args:
            compression_ratios: List of compression ratios to try
            strategies: List of strategies (or None for all)
            save: Save configs to files
            
        Returns:
            List of student configs
        """
        if strategies is None:
            strategies = ['conservative', 'balanced', 'aggressive']
        
        candidates = []
        
        LOG.info(f"\nGenerating {len(compression_ratios) * len(strategies)} candidates...")
        
        for ratio in compression_ratios:
            for strategy in strategies:
                try:
                    config = self.generate(
                        compression_ratio=ratio,
                        strategy=strategy,
                        validate=True,
                        auto_fix=True,
                        save=save
                    )
                    candidates.append(config)
                except Exception as e:
                    LOG.error(f"Failed to generate {strategy} @ {ratio:.1%}: {e}")
        
        LOG.info(f"\n✓ Successfully generated {len(candidates)} candidates")
        
        # Rank by compression ratio (closest to target)
        candidates.sort(key=lambda c: abs(c['total_params'] - self.teacher_config['total_params'] * compression_ratios[0]))
        
        return candidates
    
    def save_config(self, student_config: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Save student config to YAML file compatible with main.py.
        
        Args:
            student_config: Student architecture config
            filename: Custom filename (or auto-generate)
            
        Returns:
            Path to saved config file
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ratio = int(student_config['compression_ratio'] * 100)
            strategy = student_config['strategy']
            filename = f"student_{self.teacher_name}_{strategy}_{ratio}pct_{timestamp}.yaml"
        
        output_path = self.output_dir / filename
        
        # Create config in format compatible with main.py
        config = {
            'model': {
                'name': self.teacher_name,
                'student_name': f"auto_student_{student_config['strategy']}",
                'type': 'transformer',
                'tokenizer_name': self.teacher_name,
                'max_length': 128,
                
                # Student architecture (for manual model creation if needed)
                'student_architecture': {
                    'num_layers': student_config['num_layers'],
                    'hidden_size': student_config['hidden_size'],
                    'num_attention_heads': student_config['num_attention_heads'],
                    'intermediate_size': student_config['intermediate_size'],
                    'vocab_size': student_config['vocab_size'],
                }
            },
            'train': {
                'epochs': 3,
                'batch_size': 8,
                'lr': 2e-5,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'warmup_steps': 100,
            },
            'distillation': {
                'method': 'kd_hinton',
                'temperature': 2.0,
                'alpha': 0.5,
            },
            'metadata': {
                'generated_by': 'AutoStudentBuilder',
                'teacher': self.teacher_name,
                'compression_ratio': student_config['compression_ratio'],
                'strategy': student_config['strategy'],
                'teacher_params': self.teacher_config['total_params'],
                'student_params': student_config['total_params'],
                'compression_achieved': student_config['total_params'] / self.teacher_config['total_params'],
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        LOG.info(f"✓ Saved config to: {output_path}")
        
        # Also save raw architecture as JSON for easy reference
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(student_config, f, indent=2)
        LOG.info(f"✓ Saved architecture to: {json_path}")
        
        return output_path
    
    def estimate_training_time(self, student_config: Dict[str, Any], dataset_size: int = 10000) -> Dict[str, float]:
        """
        Estimate training time and resource usage.
        
        Args:
            student_config: Student architecture
            dataset_size: Number of training examples
            
        Returns:
            Dict with estimates (time, memory, etc.)
        """
        # Rough estimates based on parameter count
        params = student_config['total_params']
        
        # Estimate seconds per training step (very rough)
        # Smaller models train faster
        if params < 50_000_000:
            sec_per_step = 0.1
        elif params < 100_000_000:
            sec_per_step = 0.2
        else:
            sec_per_step = 0.3
        
        batch_size = 8
        steps_per_epoch = dataset_size // batch_size
        epochs = 3
        
        total_steps = steps_per_epoch * epochs
        total_time_min = (total_steps * sec_per_step) / 60
        
        # Memory estimate
        _, memory_gb = StudentValidator.check_memory_feasibility(
            student_config,
            batch_size=batch_size,
            seq_length=128,
            available_memory_gb=8.0
        )
        
        estimates = {
            'total_steps': total_steps,
            'estimated_time_minutes': total_time_min,
            'estimated_memory_gb': memory_gb,
            'batch_size': batch_size,
            'epochs': epochs,
        }
        
        LOG.info(f"\nTraining Estimates:")
        LOG.info(f"  Time: ~{total_time_min:.1f} minutes ({epochs} epochs)")
        LOG.info(f"  Memory: ~{memory_gb:.2f} GB")
        LOG.info(f"  Steps: {total_steps:,}")
        
        return estimates


__all__ = ['AutoStudentBuilder']
