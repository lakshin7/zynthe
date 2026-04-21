"""
Zynthe Toolkit - Advanced Scheduler System
===========================================

Smart learning rate scheduler system with:
- Multi-scheduler support (Cosine, Linear, Polynomial, Step, Plateau, etc.)
- Phase-aware scheduling (Distillation, Quantization, Fine-tuning)
- Warmup support (linear, cosine, constant)
- Multi-stage scheduling for complex training pipelines
- Metric-based adaptive scheduling (ReduceLROnPlateau)
- Integration with evaluation metrics (DEI, CAS)
- Hybrid scheduler mode (Cosine + Adaptive blend)

Author: Zynthe Team
License: MIT
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    LinearLR,
    PolynomialLR,
    CyclicLR,
    OneCycleLR,
    LambdaLR
)
from typing import Dict, List, Optional, Any
import logging
import math

LOG = logging.getLogger(__name__)


# =============================================================================
# Scheduler Factory - Phase-Aware Scheduler Creation
# =============================================================================

class SchedulerFactory:
    """
    Factory for creating learning rate schedulers with phase-aware configurations.
    Supports: Cosine, Linear, Step, MultiStep, Exponential, Plateau, Warmup, etc.
    """
    
    SUPPORTED_SCHEDULERS = {
        'cosine': 'Cosine annealing with optional warmup',
        'linear': 'Linear decay',
        'polynomial': 'Polynomial decay',
        'step': 'Step decay at specific epochs',
        'multistep': 'Step decay at multiple milestones',
        'exponential': 'Exponential decay',
        'plateau': 'Reduce on metric plateau',
        'constant': 'Constant learning rate',
        'onecycle': 'OneCycle policy for super-convergence',
        'cyclic': 'Cyclic learning rate',
        'multistage': 'Multi-stage scheduling for complex pipelines'
    }
    
    def __init__(self, optimizer: Optimizer, config: Dict[str, Any]):
        """
        Args:
            optimizer: Optimizer instance
            config: Configuration dict with scheduler settings
        """
        self.optimizer = optimizer
        self.config = config
        
    def get_scheduler(self, num_training_steps: Optional[int] = None):
        """
        Create scheduler based on config.
        
        Args:
            num_training_steps: Total number of training steps (for step-based schedulers)
            
        Returns:
            Learning rate scheduler (or wrapper with warmup)
        """
        scheduler_type = self.config.get('scheduler', 'cosine').lower()
        warmup_steps = self.config.get('warmup_steps', 0)
        warmup_ratio = self.config.get('warmup_ratio', 0.0)
        
        # Calculate warmup steps from ratio if specified
        if warmup_ratio > 0 and num_training_steps:
            warmup_steps = int(num_training_steps * warmup_ratio)
        
        LOG.info(f"Creating {scheduler_type} scheduler")
        if warmup_steps > 0:
            LOG.info(f"Warmup enabled: {warmup_steps} steps")
        
        # Create base scheduler
        if scheduler_type == 'cosine':
            scheduler = self._create_cosine_scheduler(num_training_steps)
        elif scheduler_type == 'linear':
            scheduler = self._create_linear_scheduler(num_training_steps)
        elif scheduler_type == 'polynomial':
            scheduler = self._create_polynomial_scheduler(num_training_steps)
        elif scheduler_type == 'step':
            scheduler = self._create_step_scheduler()
        elif scheduler_type == 'multistep':
            scheduler = self._create_multistep_scheduler()
        elif scheduler_type == 'exponential':
            scheduler = self._create_exponential_scheduler()
        elif scheduler_type == 'plateau':
            scheduler = self._create_plateau_scheduler()
        elif scheduler_type == 'constant':
            scheduler = self._create_constant_scheduler()
        elif scheduler_type == 'onecycle':
            scheduler = self._create_onecycle_scheduler(num_training_steps)
        elif scheduler_type == 'cyclic':
            scheduler = self._create_cyclic_scheduler()
        elif scheduler_type == 'multistage':
            scheduler = self._create_multistage_scheduler(num_training_steps)
        else:
            LOG.warning(f"Unknown scheduler '{scheduler_type}', using cosine")
            scheduler = self._create_cosine_scheduler(num_training_steps)
        
        # Wrap with warmup if specified
        if warmup_steps > 0 and scheduler_type != 'onecycle':  # OneCycle has built-in warmup
            warmup_type = self.config.get('warmup_type', 'linear')
            scheduler = WarmupScheduler(
                scheduler,
                warmup_steps=warmup_steps,
                warmup_type=warmup_type
            )
        
        return scheduler
    
    def _create_cosine_scheduler(self, num_training_steps: Optional[int]):
        """Create cosine annealing scheduler."""
        # Ensure num_training_steps is valid (not None)
        if num_training_steps is None or num_training_steps <= 0:
            num_training_steps = self.config.get('num_epochs', 10) * 100  # Estimate
        
        # Type assertion: after this check, num_training_steps is guaranteed to be int
        assert isinstance(num_training_steps, int) and num_training_steps > 0
        
        # FIX: Get warmup steps and adjust T_max to prevent LR from hitting zero
        warmup_steps = self.config.get('warmup_steps', 0)
        
        # Critical: The cosine scheduler should use effective steps (after warmup)
        # Otherwise, warmup consumes most of the LR budget
        effective_steps = max(num_training_steps - warmup_steps, 1)
        
        # FIX: Set eta_min to a small non-zero value to prevent complete LR collapse
        eta_min = self.config.get('eta_min', 1e-7)  # Changed from 0.0
        
        LOG.info(f"Cosine scheduler: total_steps={num_training_steps}, "
                 f"warmup_steps={warmup_steps}, effective_steps={effective_steps}, "
                 f"eta_min={eta_min}")
        
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=effective_steps,  # Use effective steps, not total
            eta_min=eta_min
        )
        return scheduler
    
    def _create_linear_scheduler(self, num_training_steps: Optional[int]):
        """Create linear decay scheduler."""
        # Ensure num_training_steps is valid (not None)
        if num_training_steps is None or num_training_steps <= 0:
            num_training_steps = self.config.get('num_epochs', 10) * 100
        
        # Type assertion: after this check, num_training_steps is guaranteed to be int
        assert isinstance(num_training_steps, int) and num_training_steps > 0
        
        start_factor = self.config.get('start_factor', 1.0)
        end_factor = self.config.get('end_factor', 0.0)
        
        scheduler = LinearLR(
            self.optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=num_training_steps
        )
        return scheduler
    
    def _create_polynomial_scheduler(self, num_training_steps: Optional[int]):
        """Create polynomial decay scheduler."""
        # Ensure num_training_steps is valid (not None)
        if num_training_steps is None or num_training_steps <= 0:
            num_training_steps = self.config.get('num_epochs', 10) * 100
        
        # Type assertion: after this check, num_training_steps is guaranteed to be int
        assert isinstance(num_training_steps, int) and num_training_steps > 0
        
        power = self.config.get('power', 1.0)
        
        scheduler = PolynomialLR(
            self.optimizer,
            total_iters=num_training_steps,
            power=power
        )
        return scheduler
    
    def _create_step_scheduler(self):
        """Create step decay scheduler."""
        step_size = self.config.get('step_size', 10)
        gamma = self.config.get('gamma', 0.1)
        
        scheduler = StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )
        return scheduler
    
    def _create_multistep_scheduler(self):
        """Create multi-step decay scheduler."""
        milestones = self.config.get('milestones', [30, 60, 90])
        gamma = self.config.get('gamma', 0.1)
        
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=gamma
        )
        return scheduler
    
    def _create_exponential_scheduler(self):
        """Create exponential decay scheduler."""
        gamma = self.config.get('gamma', 0.95)
        
        scheduler = ExponentialLR(
            self.optimizer,
            gamma=gamma
        )
        return scheduler
    
    def _create_plateau_scheduler(self):
        """Create reduce-on-plateau scheduler."""
        mode = self.config.get('mode', 'max')  # 'max' for accuracy, 'min' for loss
        factor = self.config.get('factor', 0.5)
        patience = self.config.get('patience', 10)
        threshold = self.config.get('threshold', 1e-4)
        min_lr = self.config.get('min_lr', 1e-7)
        
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )
        return scheduler
    
    def _create_constant_scheduler(self):
        """Create constant LR scheduler (no change)."""
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        return scheduler
    
    def _create_onecycle_scheduler(self, num_training_steps: Optional[int]):
        """Create OneCycle scheduler for super-convergence."""
        if num_training_steps is None:
            num_training_steps = self.config.get('num_epochs', 10) * 100
        
        max_lr = self.config.get('max_lr', self.config.get('lr', 1e-3))
        pct_start = self.config.get('pct_start', 0.3)
        anneal_strategy = self.config.get('anneal_strategy', 'cos')
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=num_training_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy
        )
        return scheduler
    
    def _create_cyclic_scheduler(self):
        """Create cyclic LR scheduler."""
        base_lr = self.config.get('base_lr', 1e-5)
        max_lr = self.config.get('max_lr', 1e-3)
        step_size_up = self.config.get('step_size_up', 2000)
        mode = self.config.get('cyclic_mode', 'triangular')
        
        scheduler = CyclicLR(
            self.optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode
        )
        return scheduler
    
    def _create_multistage_scheduler(self, num_training_steps: Optional[int]):
        """Create multi-stage scheduler for complex pipelines."""
        stages = self.config.get('stages', [
            {'scheduler': 'linear', 'steps': 1000},
            {'scheduler': 'cosine', 'steps': 4000}
        ])
        
        scheduler = MultiStageScheduler(
            self.optimizer,
            stages=stages,
            num_training_steps=num_training_steps
        )
        return scheduler


# =============================================================================
# Warmup Scheduler Wrapper
# =============================================================================

class WarmupScheduler:
    """
    Learning rate warmup scheduler wrapper.
    Gradually increases LR from 0 to base LR during warmup phase,
    then follows the main scheduler.
    """
    
    def __init__(
        self,
        scheduler,
        warmup_steps: int,
        warmup_type: str = 'linear'
    ):
        """
        Args:
            scheduler: Base scheduler to wrap
            warmup_steps: Number of warmup steps
            warmup_type: Type of warmup ('linear', 'cosine', 'constant')
        """
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_type = warmup_type
        self.current_step = 0
        
        # Store initial LRs
        self.base_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]
        
    def step(self, *args, **kwargs):
        """Perform scheduler step with warmup."""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            warmup_factor = self._get_warmup_factor()
            for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # Main scheduler phase
            self.scheduler.step(*args, **kwargs)
        
        self.current_step += 1
    
    def _get_warmup_factor(self) -> float:
        """Calculate warmup factor based on current step."""
        progress = self.current_step / self.warmup_steps
        
        if self.warmup_type == 'linear':
            return progress
        elif self.warmup_type == 'cosine':
            return 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
        elif self.warmup_type == 'constant':
            return 0.5  # Constant 50% during warmup
        else:
            return progress
    
    def get_last_lr(self):
        """Get last learning rate."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.scheduler.optimizer.param_groups]
    
    def state_dict(self):
        """Return state dict."""
        return {
            'scheduler': self.scheduler.state_dict(),
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'warmup_type': self.warmup_type,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.current_step = state_dict.get('current_step', 0)
        self.warmup_steps = state_dict.get('warmup_steps', 0)
        self.warmup_type = state_dict.get('warmup_type', 'linear')
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)


# =============================================================================
# Multi-Stage Scheduler
# =============================================================================

class MultiStageScheduler:
    """
    Multi-stage scheduler for complex training pipelines.
    Allows different scheduling policies for different training stages.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        stages: List[Dict[str, Any]],
        num_training_steps: Optional[int] = None
    ):
        """
        Args:
            optimizer: Optimizer instance
            stages: List of stage configurations
            num_training_steps: Total training steps
        """
        self.optimizer = optimizer
        self.stages = stages
        self.current_stage_idx = 0
        self.current_step = 0
        self.stage_start_step = 0
        
        # Create schedulers for each stage
        self.stage_schedulers = []
        for stage in stages:
            stage_config = {**stage, 'num_epochs': 1}
            factory = SchedulerFactory(optimizer, stage_config)
            scheduler = factory.get_scheduler(stage.get('steps', 1000))
            self.stage_schedulers.append(scheduler)
        
        LOG.info(f"Created multi-stage scheduler with {len(stages)} stages")
    
    def step(self, *args, **kwargs):
        """Perform scheduler step."""
        # Check if we need to transition to next stage
        if self.current_stage_idx < len(self.stages):
            stage = self.stages[self.current_stage_idx]
            stage_steps = stage.get('steps', 1000)
            
            if self.current_step - self.stage_start_step >= stage_steps:
                # Transition to next stage
                self.current_stage_idx += 1
                self.stage_start_step = self.current_step
                if self.current_stage_idx < len(self.stages):
                    LOG.info(f"Transitioning to stage {self.current_stage_idx + 1}")
        
        # Execute current stage scheduler
        if self.current_stage_idx < len(self.stage_schedulers):
            self.stage_schedulers[self.current_stage_idx].step(*args, **kwargs)
        
        self.current_step += 1
    
    def get_last_lr(self):
        """Get last learning rate."""
        if self.current_stage_idx < len(self.stage_schedulers):
            scheduler = self.stage_schedulers[self.current_stage_idx]
            if hasattr(scheduler, 'get_last_lr'):
                return scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]


# =============================================================================
# Adaptive Scheduler (Metric-Based)
# =============================================================================

class AdaptiveScheduler:
    """
    Adaptive scheduler that adjusts based on evaluation metrics.
    Integrates with DEI (Distillation Efficacy Index) and CAS (Compression-Aware Score).
    """
    
    def __init__(
        self,
        scheduler,
        enable_adaptive: bool = True,
        mode: str = 'max',  # 'max' for accuracy/DEI, 'min' for loss
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-7
    ):
        """
        Args:
            scheduler: Base scheduler to wrap
            enable_adaptive: Enable adaptive adjustments
            mode: Optimization mode ('max' or 'min')
            patience: Epochs to wait before adjusting
            factor: Factor to multiply LR by
            min_lr: Minimum learning rate
        """
        self.scheduler = scheduler
        self.enable_adaptive = enable_adaptive
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.wait_count = 0
        
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """
        Perform scheduler step with optional metric-based adjustment.
        
        Args:
            metrics: Evaluation metrics (optional)
        """
        # Regular scheduler step
        if isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau requires metric
            if metrics:
                metric_value = metrics.get('accuracy', metrics.get('loss', 0.0))
                self.scheduler.step(metric_value)
        else:
            self.scheduler.step()
        
        # Adaptive adjustment based on metrics
        if self.enable_adaptive and metrics:
            self._adaptive_adjust(metrics)
    
    def _adaptive_adjust(self, metrics: Dict[str, float]):
        """Adaptively adjust LR based on metrics."""
        # Prefer DEI, fallback to accuracy/loss
        key_metric = metrics.get('dei', metrics.get('accuracy', metrics.get('loss', 0.0)))
        
        # Check if metric improved
        if self.mode == 'max':
            improved = key_metric > self.best_metric
        else:
            improved = key_metric < self.best_metric
        
        if improved:
            self.best_metric = key_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Adjust LR if patience exceeded
        if self.wait_count >= self.patience:
            self._reduce_lr()
            self.wait_count = 0
    
    def _reduce_lr(self):
        """Reduce learning rate."""
        for param_group in self.scheduler.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            LOG.info(f"Adaptive LR reduction: {old_lr:.2e} → {new_lr:.2e}")
    
    def get_last_lr(self):
        """Get last learning rate."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.scheduler.optimizer.param_groups]


# =============================================================================
# Convenience Function
# =============================================================================

def get_scheduler(
    optimizer: Optimizer,
    config: Optional[Dict] = None,
    num_training_steps: Optional[int] = None
):
    """
    Convenience function for creating scheduler.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dict (if None, returns constant scheduler)
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    if config is None:
        config = {'scheduler': 'constant'}
    
    factory = SchedulerFactory(optimizer, config)
    return factory.get_scheduler(num_training_steps)


# =============================================================================
# Module Metadata
# =============================================================================

__all__ = [
    'SchedulerFactory',
    'WarmupScheduler',
    'MultiStageScheduler',
    'AdaptiveScheduler',
    'get_scheduler',
]

__version__ = '2.0.0'
__author__ = 'Zynthe Team'
