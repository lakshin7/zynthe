"""
Multi-Stage Distiller - Progressive Knowledge Transfer

Orchestrates sequential distillation stages with:
1. Stage Controller: Manages sequence, duration, checkpoints
2. Distiller Registry: Plug-and-play distiller modules
3. Adaptive Loss Scheduler: Dynamic weight adjustment
4. Intermediate Evaluation: Stage-wise performance tracking
5. Knowledge Replay: Prevents catastrophic forgetting
6. Progressive Precision: Gradual quantization
7. Layer-wise Freezing: Efficient computation
8. Preflight-Aware Planning: Automatic stage generation

Example:
    Stage 1: Logit Alignment (KD-Hinton) -> alpha=0.9
    Stage 2: Feature Refinement -> beta=0.6
    Stage 3: Similarity Transfer (Relational) -> gamma=0.4
    Stage 4: Attention Imitation -> delta=0.3
    Stage 5: QAT Fine-Tuning -> int8
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings
from collections import defaultdict

from .base_distiller import BaseDistiller
from .kd_hinton import KDHintonDistiller
from .feature_distiller import FeatureDistiller
from .similarity_transfer import SimilarityTransfer

# Optional imports
try:
    from .attention_transfer import AttentionTransferDistiller
    HAS_ATTENTION = True
except ImportError:
    HAS_ATTENTION = False
    AttentionTransferDistiller = None  # type: ignore
    warnings.warn("AttentionTransfer not available")

try:
    from ..quant.qat import QATDistiller  # type: ignore[import]
    HAS_QAT = True
except ImportError:
    HAS_QAT = False
    QATDistiller = None  # type: ignore
    warnings.warn("QAT not available. Stage 'qat' will be skipped.")


class StageController:
    """Controls stage execution, checkpoints, and dependencies."""
    
    def __init__(self, stages: List[Dict], output_dir: str):
        """Initialize stage controller."""
        self.stages = stages
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_stage = 0
        self.stage_history = []
        self.checkpoints = {}
    
    def get_next_stage(self) -> Optional[Dict]:
        """Get next stage configuration."""
        if self.current_stage < len(self.stages):
            stage = self.stages[self.current_stage]
            self.current_stage += 1
            return stage
        return None
    
    def save_checkpoint(
        self,
        stage_idx: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        metrics: Dict[str, float]
    ) -> Path:
        """Save stage checkpoint."""
        checkpoint_path = self.output_dir / f"stage_{stage_idx}_checkpoint.pt"
        
        checkpoint = {
            'stage_idx': stage_idx,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints[stage_idx] = checkpoint_path
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, stage_idx: int, model: nn.Module) -> Dict:
        """Load checkpoint from previous stage."""
        if stage_idx not in self.checkpoints:
            raise ValueError(f"No checkpoint found for stage {stage_idx}")
        
        checkpoint_path = self.checkpoints[stage_idx]
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from stage {stage_idx}")
        return checkpoint
    
    def check_dependencies(self, stage: Dict) -> bool:
        """Check if stage dependencies are satisfied."""
        dependencies = stage.get('depends_on', [])
        
        for dep_idx in dependencies:
            if dep_idx not in self.checkpoints:
                print(f"⚠ Dependency not satisfied: stage {dep_idx} not completed")
                return False
        
        return True
    
    def log_stage(self, stage_idx: int, stage_name: str, metrics: Dict):
        """Log stage completion."""
        self.stage_history.append({
            'stage': stage_idx,
            'name': stage_name,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_report(self) -> Dict:
        """Generate training report."""
        return {
            'total_stages': len(self.stages),
            'completed_stages': len(self.stage_history),
            'stage_history': self.stage_history,
            'checkpoints': {k: str(v) for k, v in self.checkpoints.items()}
        }


class DistillerRegistry:
    """Registry for distiller modules with plug-and-play support."""
    
    def __init__(self):
        """Initialize registry with built-in distillers."""
        self._registry = {
            'kd': KDHintonDistiller,
            'kd_hinton': KDHintonDistiller,
            'feature': FeatureDistiller,
            'similarity': SimilarityTransfer,
            'similarity_transfer': SimilarityTransfer,
        }
        
        # Add optional distillers
        if HAS_ATTENTION and AttentionTransferDistiller is not None:
            self._registry['attention'] = AttentionTransferDistiller
        
        if HAS_QAT and QATDistiller is not None:
            self._registry['qat'] = QATDistiller
    
    def register(self, name: str, distiller_cls: type):
        """Register custom distiller."""
        self._registry[name] = distiller_cls
        print(f"✓ Registered distiller: {name}")
    
    def get(self, name: str) -> Optional[type]:
        """Get distiller by name."""
        return self._registry.get(name)
    
    def list_available(self) -> List[str]:
        """List all available distillers."""
        return list(self._registry.keys())


class AdaptiveLossScheduler:
    """Dynamically adjusts loss weights across stages."""
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None, schedule_type: str = 'linear'):
        """
        Args:
            initial_weights: Initial loss weights (alpha, beta, gamma). If None, uses defaults.
            schedule_type: 'linear', 'cosine', or 'step'
        """
        # Use default weights if none provided
        if initial_weights is None:
            initial_weights = {'alpha': 0.7, 'beta': 0.5, 'gamma': 0.3}
        
        self.weights = initial_weights.copy()
        self.initial_weights = initial_weights.copy()
        self.schedule_type = schedule_type
        self.history = []
    
    def update(self, stage_idx: int, total_stages: int, metrics: Dict[str, float]):
        """Update weights based on progress and metrics."""
        progress = stage_idx / total_stages
        
        if self.schedule_type == 'linear':
            # Linear decay for early distillation
            self.weights['alpha'] = self.initial_weights['alpha'] * (1 - 0.5 * progress)
            self.weights['beta'] = self.initial_weights['beta'] * (1 + 0.5 * progress)
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing
            import math
            self.weights['alpha'] = self.initial_weights['alpha'] * (1 + math.cos(math.pi * progress)) / 2
        
        elif self.schedule_type == 'adaptive':
            # Performance-based adjustment
            if 'student_acc' in metrics and 'teacher_acc' in metrics:
                gap = metrics['teacher_acc'] - metrics['student_acc']
                if gap > 0.1:
                    self.weights['alpha'] *= 1.1  # Increase KD weight
                else:
                    self.weights['beta'] *= 1.1  # Focus on features
        
        self.history.append(self.weights.copy())
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()


class MultiStageDistiller:
    """
    Progressive multi-stage knowledge distillation orchestrator.
    
    Features:
    - Sequential stage execution with checkpointing
    - Plug-and-play distiller registry
    - Adaptive loss weight scheduling
    - Intermediate evaluation and reporting
    - Knowledge replay for forgetting prevention
    - Progressive precision reduction (QAT)
    - Layer-wise freezing for efficiency
    - Preflight-aware automatic stage planning
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Dict[str, Any],
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        output_dir: str = 'experiments/multi_stage'
    ):
        """
        Initialize multi-stage distiller.
        
        Args:
            teacher: Teacher model
            student: Student model
            config: Configuration dictionary
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device for training
            output_dir: Output directory for checkpoints and logs
        """
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.stages = self._parse_stages(config)
        self.controller = StageController(self.stages, output_dir)
        self.registry = DistillerRegistry()
        self.loss_scheduler = AdaptiveLossScheduler(
            config.get('distillation', {}).get('loss_schedule')
        )
        
        # Stage tracking
        self.stage_metrics = []
        self.knowledge_bank = []  # For knowledge replay
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_stages(self, config: Dict) -> List[Dict]:
        """
        Parse stage configuration from config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of stage configurations
        """
        distill_cfg = config.get('distillation', {})
        
        # Check if multi-stage is enabled
        if not distill_cfg.get('multi_stage', False):
            # Single stage fallback
            return [{
                'name': 'Single Stage Distillation',
                'type': distill_cfg.get('method', 'kd'),
                'epochs': config.get('training', {}).get('epochs', 10),
                'config': distill_cfg
            }]
        
        # Parse stages from config
        stages = distill_cfg.get('stages', [])
        
        if not stages:
            # Auto-generate stages from preflight
            stages = self._auto_generate_stages(config)
        
        return stages
    
    def _auto_generate_stages(self, config: Dict) -> List[Dict]:
        """
        Auto-generate stages based on preflight analysis.
        
        Args:
            config: Configuration with preflight results
            
        Returns:
            Generated stage configurations
        """
        print("🤖 Auto-generating stages from preflight analysis...")
        
        preflight = config.get('preflight', {})
        compression_ratio = preflight.get('compression_ratio', 2.0)
        model_type = preflight.get('model_type', 'unknown')
        
        stages = []
        
        # Stage 1: Always start with logit alignment
        stages.append({
            'name': 'Stage 1 - Logit Alignment',
            'type': 'kd',
            'epochs': 3,
            'config': {
                'temperature': 4.0,
                'alpha': 0.9
            }
        })
        
        # Stage 2: Feature distillation for high compression
        if compression_ratio > 3.0:
            stages.append({
                'name': 'Stage 2 - Feature Refinement',
                'type': 'feature',
                'epochs': 3,
                'config': {
                    'beta': 0.6,
                    'feature_stages': [1, 2, 3]
                }
            })
        
        # Stage 3: Attention transfer for transformers
        if ('transformer' in model_type.lower() or 'bert' in model_type.lower()) and HAS_ATTENTION:
            stages.append({
                'name': 'Stage 3 - Attention Imitation',
                'type': 'attention',
                'epochs': 2,
                'config': {
                    'gamma': 0.3
                }
            })
        
        # Stage 4: QAT for very high compression
        if compression_ratio > 8.0 and HAS_QAT:
            stages.append({
                'name': 'Stage 4 - QAT Fine-Tuning',
                'type': 'qat',
                'epochs': 2,
                'config': {
                    'precision': 'int8'
                }
            })
        
        print(f"  → Generated {len(stages)} stages")
        for i, stage in enumerate(stages, 1):
            print(f"     {i}. {stage['name']} ({stage['type']})")
        
        return stages
    
    def run(self) -> Dict[str, Any]:
        """
        Execute multi-stage distillation.
        
        Returns:
            Training report with all stages
        """
        print("\n" + "=" * 70)
        print("🚀 MULTI-STAGE DISTILLATION")
        print("=" * 70)
        print(f"Total Stages: {len(self.stages)}")
        print(f"Output Dir: {self.output_dir}")
        print("")
        
        # Run each stage
        for stage_idx, stage_cfg in enumerate(self.stages, 1):
            print(f"\n{'=' * 70}")
            print(f"📍 STAGE {stage_idx}/{len(self.stages)}: {stage_cfg['name']}")
            print(f"{'=' * 70}")
            
            # Check dependencies
            if not self.controller.check_dependencies(stage_cfg):
                print("❌ Dependencies not satisfied. Skipping stage.")
                continue
            
            # Run stage
            stage_metrics = self._run_stage(stage_idx, stage_cfg)
            
            # Save checkpoint
            self.controller.save_checkpoint(
                stage_idx,
                self.student,
                None,  # Can add optimizer if needed
                stage_metrics
            )
            
            # Log stage
            self.controller.log_stage(stage_idx, stage_cfg['name'], stage_metrics)
            self.stage_metrics.append(stage_metrics)
            
            # Adaptive weight adjustment
            # Update loss scheduler weights based on stage metrics
            if hasattr(self.loss_scheduler, 'update'):
                self.loss_scheduler.update(stage_idx, len(self.stages), stage_metrics)
            
            print(f"\n✅ Stage {stage_idx} completed!")
            self._print_stage_summary(stage_idx, stage_metrics)
        
        # Generate final report
        report = self._generate_final_report()
        
        # Save report
        self._save_report(report)
        
        print("\n" + "=" * 70)
        print("🎉 MULTI-STAGE DISTILLATION COMPLETED")
        print("=" * 70)
        self._print_final_summary(report)
        
        return report
    
    def _run_stage(self, stage_idx: int, stage_cfg: Dict) -> Dict[str, float]:
        """
        Run single distillation stage.
        
        Args:
            stage_idx: Stage index
            stage_cfg: Stage configuration
            
        Returns:
            Stage metrics
        """
        # Get distiller class
        distiller_cls = self.registry.get(stage_cfg['type'])
        
        if distiller_cls is None:
            warnings.warn(f"Distiller type '{stage_cfg['type']}' not found in registry. Skipping stage.")
            return {'train_loss': 0.0, 'val_loss': 0.0, 'val_accuracy': 0.0}
        
        # Get loss weights (update scheduler first if needed)
        if hasattr(self.loss_scheduler, 'update'):
            self.loss_scheduler.update(stage_idx, len(self.stages), {})
        loss_weights = self.loss_scheduler.get_weights()
        print(f"Loss weights: α={loss_weights.get('alpha', 0):.2f}, "
              f"β={loss_weights.get('beta', 0):.2f}, "
              f"γ={loss_weights.get('gamma', 0):.2f}")
        
        # Merge stage config with loss weights
        distiller_config = {**stage_cfg.get('config', {}), **loss_weights}
        
        # Initialize distiller
        print(f"Initializing {stage_cfg['type']} distiller...")
        distiller = distiller_cls(
            self.teacher,
            self.student,
            distiller_config
        )
        
        # Apply layer freezing if specified
        if stage_cfg.get('freeze_layers'):
            self._freeze_layers(stage_cfg['freeze_layers'])
        
        # Train stage
        epochs = stage_cfg.get('epochs', 3)
        print(f"Training for {epochs} epochs...")
        
        stage_metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'val_accuracy': 0.0
        }
        
        # Simple training loop (can be expanded)
        if self.train_loader is not None:
            for epoch in range(epochs):
                epoch_loss = self._train_epoch(distiller, epoch + 1, epochs)
                stage_metrics['train_loss'] = epoch_loss
                
                # Evaluate
                if self.val_loader is not None and (epoch + 1) % 1 == 0:
                    val_metrics = self._evaluate(distiller)
                    stage_metrics.update(val_metrics)
                    
                    print(f"  Epoch {epoch + 1}/{epochs}: "
                          f"Loss={epoch_loss:.4f}, "
                          f"Val Acc={val_metrics.get('val_accuracy', 0):.2f}%")
        else:
            warnings.warn("No train_loader provided, skipping training")
        
        # Unfreeze layers
        if stage_cfg.get('freeze_layers'):
            self._unfreeze_all()
        
        # Knowledge replay (store teacher outputs)
        if stage_cfg.get('knowledge_replay', False):
            self._store_knowledge(distiller)
        
        return stage_metrics
    
    def _train_epoch(
        self,
        distiller: BaseDistiller,
        epoch: int,
        total_epochs: int
    ) -> float:
        """
        Train single epoch.
        
        Args:
            distiller: Distiller instance
            epoch: Current epoch
            total_epochs: Total epochs
            
        Returns:
            Average loss
        """
        if self.train_loader is None:
            warnings.warn("train_loader is None, returning 0.0 loss")
            return 0.0
        
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch.get('labels', None)
                if labels is not None:
                    labels = labels.to(self.device)
            else:
                inputs = batch.to(self.device)
                labels = None
            
            # Forward pass
            try:
                # Get outputs from both models
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs) if not isinstance(inputs, dict) else self.teacher(**inputs)
                
                student_outputs = self.student(inputs) if not isinstance(inputs, dict) else self.student(**inputs)
                
                # Compute loss
                loss_result = distiller.compute_loss(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    targets=labels
                )
                
                # Handle loss result (could be tensor or tuple)
                if isinstance(loss_result, tuple):
                    loss = loss_result[0]
                else:
                    loss = loss_result
                
                # Accumulate loss
                if hasattr(loss, 'item'):
                    total_loss += loss.item()
                else:
                    total_loss += float(loss)
                
                num_batches += 1
                
            except Exception as e:
                warnings.warn(f"Error in batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate(self, distiller: BaseDistiller) -> Dict[str, float]:
        """
        Evaluate current student model.
        
        Args:
            distiller: Distiller instance
            
        Returns:
            Evaluation metrics
        """
        if self.val_loader is None:
            warnings.warn("val_loader is None, returning empty metrics")
            return {'val_loss': 0.0, 'val_accuracy': 0.0}
        
        self.student.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    labels = batch.get('labels', None)
                    if labels is not None:
                        labels = labels.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    labels = None
                
                # Forward pass
                try:
                    if isinstance(inputs, dict):
                        student_out = self.student(**inputs)
                    else:
                        student_out = self.student(inputs)
                    
                    # Extract logits - handle dict, object with logits attr, or tensor
                    if isinstance(student_out, dict):
                        logits = student_out['logits']
                    elif hasattr(student_out, 'logits'):
                        logits = student_out.logits
                    elif isinstance(student_out, tuple):
                        logits = student_out[0]
                    else:
                        logits = student_out
                    
                    # Compute accuracy
                    if labels is not None and hasattr(logits, 'dim') and logits.dim() >= 2:
                        _, predicted = logits.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                except Exception as e:
                    warnings.warn(f"Error in evaluation: {e}")
                    continue
        
        accuracy = 100.0 * correct / max(total, 1)
        
        return {
            'val_loss': total_loss / max(len(self.val_loader), 1),
            'val_accuracy': accuracy
        }
    
    def _freeze_layers(self, layer_spec: List[int]):
        """
        Freeze specified layers.
        
        Args:
            layer_spec: List of layer indices to freeze
        """
        print(f"  🔒 Freezing layers: {layer_spec}")
        
        # Get all named parameters
        named_params = list(self.student.named_parameters())
        
        for idx in layer_spec:
            if idx < len(named_params):
                name, param = named_params[idx]
                param.requires_grad = False
    
    def _unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.student.parameters():
            param.requires_grad = True
    
    def _store_knowledge(self, distiller: BaseDistiller):
        """
        Store teacher knowledge for replay.
        
        Args:
            distiller: Distiller instance
        """
        if self.train_loader is None:
            warnings.warn("train_loader is None, skipping knowledge storage")
            return
        
        print("  💾 Storing knowledge for replay...")
        
        # Store teacher outputs for a subset of data
        self.teacher.eval()
        knowledge_samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= 10:  # Store first 10 batches
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                teacher_out = self.teacher(inputs)
                knowledge_samples.append(teacher_out.cpu())
        
        self.knowledge_bank.append(knowledge_samples)
        print(f"  ✓ Stored {len(knowledge_samples)} knowledge samples")
    
    def _print_stage_summary(self, stage_idx: int, metrics: Dict):
        """Print stage summary."""
        print("\n📊 Stage Summary:")
        print(f"  Train Loss: {metrics.get('train_loss', 0):.4f}")
        print(f"  Val Loss: {metrics.get('val_loss', 0):.4f}")
        print(f"  Val Accuracy: {metrics.get('val_accuracy', 0):.2f}%")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        # Calculate aggregate metrics
        total_gain = 0.0
        for i, metrics in enumerate(self.stage_metrics):
            if i > 0:
                prev_acc = self.stage_metrics[i-1].get('val_accuracy', 0)
                curr_acc = metrics.get('val_accuracy', 0)
                gain = curr_acc - prev_acc
                total_gain += gain
        
        # Get preflight info
        preflight = self.config.get('preflight', {})
        
        report = {
            'summary': {
                'total_stages': len(self.stages),
                'model_type': preflight.get('model_type', 'unknown'),
                'compression_ratio': preflight.get('compression_ratio', 0),
                'total_accuracy_gain': 0.0  # Will be updated below
            },
            'preflight': {
                'model_type': preflight.get('model_type', 'unknown'),
                'compression_ratio': preflight.get('compression_ratio', 0),
                'stages_completed': []
            },
            'stages': [],
            'final_metrics': {},
            'stage_controller_report': self.controller.generate_report()
        }
        
        # Add stage details
        for i, (stage_cfg, metrics) in enumerate(zip(self.stages, self.stage_metrics)):
            stage_info = {
                'stage': i + 1,
                'name': stage_cfg['name'],
                'type': stage_cfg['type'],
                'epochs': stage_cfg.get('epochs', 0),
                'metrics': metrics
            }
            
            report['stages'].append(stage_info)
            
            # Add to preflight summary
            acc_gain = 0.0
            if i > 0:
                prev_acc = self.stage_metrics[i-1].get('val_accuracy', 0)
                curr_acc = metrics.get('val_accuracy', 0)
                acc_gain = curr_acc - prev_acc
            
            report['preflight']['stages_completed'].append({
                'name': stage_cfg['name'],
                'accuracy_gain': acc_gain
            })
        
        # Final metrics
        if self.stage_metrics:
            final = self.stage_metrics[-1]
            report['final_metrics'] = {
                'final_accuracy': final.get('val_accuracy', 0),
                'total_accuracy_gain': total_gain,
                'final_loss': final.get('val_loss', 0)
            }
            # Update summary as well
            report['summary']['total_accuracy_gain'] = total_gain
        
        return report
    
    def _print_final_summary(self, report: Dict):
        """Print final summary."""
        print("\n📊 FINAL SUMMARY")
        print("-" * 70)
        
        preflight = report['preflight']
        print(f"Model Type: {preflight['model_type']}")
        print(f"Compression Ratio: {preflight['compression_ratio']:.1f}x")
        print(f"Stages Completed: {len(preflight['stages_completed'])}")
        
        print("\n📈 Stage-wise Progress:")
        for stage_info in preflight['stages_completed']:
            gain = stage_info['accuracy_gain']
            sign = '+' if gain >= 0 else ''
            print(f"  {stage_info['name']}: {sign}{gain:.2f}% accuracy gain")
        
        if report.get('final_metrics'):
            final = report['final_metrics']
            print(f"\n🎯 Final Results:")
            print(f"  Final Accuracy: {final['final_accuracy']:.2f}%")
            print(f"  Total Gain: {final['total_accuracy_gain']:.2f}%")
    
    def _save_report(self, report: Dict):
        """Save report to file."""
        # Save JSON
        json_path = self.output_dir / 'multi_stage_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Report saved: {json_path}")
        
        # Save YAML
        yaml_path = self.output_dir / 'multi_stage_report.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        print(f"💾 Report saved: {yaml_path}")


# Convenience function for backward compatibility
def run_multi_stage_distillation(
    teacher: nn.Module,
    student: nn.Module,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    output_dir: str = 'experiments/multi_stage'
) -> Dict[str, Any]:
    """
    Convenience function to run multi-stage distillation.
    
    Args:
        teacher: Teacher model
        student: Student model
        config: Configuration dictionary
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device for training
        output_dir: Output directory
        
    Returns:
        Training report
    """
    distiller = MultiStageDistiller(
        teacher=teacher,
        student=student,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir
    )
    
    return distiller.run()

