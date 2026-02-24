"""
Base Distiller - Foundation for Knowledge Distillation
========================================================

This module provides the core BaseDistiller class that serves as the template
for all specialized knowledge distillation methods in Zynthe.

Core Components:
1. Teacher/Student model management
2. Loss composition and criterion management
3. Optimizer and scheduler initialization
4. Forward hook registration for intermediate features
5. Training and evaluation step methods
6. Logging and metrics tracking
7. Configuration handling

All specialized distillers (Logit, Feature, Attention, Hybrid, etc.) extend
this base class and implement their specific loss computation logic.

Reference: Zynthe Architecture Blueprint v2.0
"""

from typing import Any, Dict, Tuple, List, Optional, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW, SGD
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR
from collections import OrderedDict
import warnings


class BaseDistiller(nn.Module):
    """
    Base class for all knowledge distillation methods.
    
    Provides a comprehensive template for distillation with:
    - Automatic hook management for feature extraction
    - Configurable loss composition
    - Optimizer and scheduler setup
    - Training/evaluation step methods
    - Metrics logging and visualization hooks
    - Loss-agnostic design for maximum flexibility
    
    All specialized distillers should extend this class and implement:
    - compute_loss(): Specific distillation loss computation
    - (Optional) _register_hooks(): Custom hook registration
    - (Optional) _init_losses(): Custom loss initialization
    
    Architecture:
    ```
    BaseDistiller
    ├── Teacher Model (frozen)
    ├── Student Model (trainable)
    ├── Feature Extractors (hooks)
    ├── Loss Composer (criterion manager)
    ├── Optimizer & Scheduler
    └── Metrics Logger
    ```
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        freeze_teacher: bool = True,
        **kwargs
    ):
        """
        Initialize the base distiller.
        
        Args:
            teacher: Pre-trained teacher model
            student: Student model to be trained
            config: Configuration dictionary with distillation parameters
            device: Target device (auto-detected if None)
            freeze_teacher: Whether to freeze teacher parameters (default: True)
            **kwargs: Additional arguments for subclass customization
        """
        super().__init__()
        
        # Core components
        self.teacher = teacher
        self.student = student
        self.config = config or {}
        self.device = device or self._auto_detect_device()
        
        # Move models to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        
        # Freeze teacher and set to eval mode
        if freeze_teacher:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        
        # Feature extraction hooks
        self.teacher_hooks: Dict[str, torch.Tensor] = {}
        self.student_hooks: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[Any] = []
        
        # Register hooks for intermediate features
        self._register_hooks()
        
        # Loss composer (criterion manager)
        self.losses: Dict[str, nn.Module] = {}
        self.loss_weights: Dict[str, float] = {}
        self._init_losses()
        
        # Optimizer and scheduler (initialized externally or internally)
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        
        # Metrics tracking
        self.metrics: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.current_epoch: int = 0
        self.global_step: int = 0
        
        # Visualization hooks (optional)
        self.visualize_features: bool = self.config.get('visualize_features', False)
        self.visualization_callback: Optional[Callable] = None
    
    # ============================================================================
    # CORE SETUP METHODS
    # ============================================================================
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _register_hooks(self) -> None:
        """
        Register forward hooks to capture intermediate features.
        
        Subclasses can override this to specify which layers to hook.
        Default: No hooks (logit-only distillation).
        
        Example:
        ```python
        def _register_hooks(self):
            # Hook teacher layer
            teacher_layer = dict(self.teacher.named_modules())['encoder.layer.6']
            handle = teacher_layer.register_forward_hook(
                self._get_teacher_hook('encoder.layer.6')
            )
            self._hook_handles.append(handle)
        ```
        """
        pass  # Subclasses implement as needed

    def _move_to_device(self, data: Any) -> Any:
        """Recursively move tensors contained in `data` onto the distiller's device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._move_to_device(v) for v in data)
        return data
    
    def _get_teacher_hook(self, name: str) -> Callable:
        """Create a forward hook for teacher features."""
        def hook(module, input, output):
            # Detach to prevent gradients flowing to teacher
            self.teacher_hooks[name] = output.detach() if isinstance(output, torch.Tensor) else output
        return hook
    
    def _get_student_hook(self, name: str) -> Callable:
        """Create a forward hook for student features."""
        def hook(module, input, output):
            self.student_hooks[name] = output  # Keep gradients for student
        return hook
    
    def _collect_features(self, hook_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Collect features from hooks and return a copy.
        
        Args:
            hook_dict: Dictionary with hooked features
            
        Returns:
            Copy of hooked features (prevents mutation)
        """
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in hook_dict.items()}
    
    def _init_losses(self) -> None:
        """
        Initialize loss functions based on configuration.
        
        Supports flexible loss composition via config:
        ```yaml
        losses:
          - type: logit
            weight: 1.0
            temperature: 2.0
          - type: feature
            weight: 0.5
            layer: encoder.layer.6
          - type: attention
            weight: 0.3
        ```
        
        Subclasses can override to add custom losses.
        """
        loss_configs = self.config.get('losses', [])
        
        if not loss_configs:
            # Default: supervised cross-entropy only
            self.losses['supervised'] = nn.CrossEntropyLoss()
            self.loss_weights['supervised'] = 1.0
        else:
            for loss_cfg in loss_configs:
                loss_type = loss_cfg.get('type', 'supervised')
                weight = loss_cfg.get('weight', 1.0)
                
                if loss_type == 'supervised' or loss_type == 'ce':
                    self.losses['supervised'] = nn.CrossEntropyLoss()
                    self.loss_weights['supervised'] = weight
                
                # Subclasses add their specific losses here

    def _resolve_task_type(self, logits: Optional[torch.Tensor] = None) -> str:
        """Infer task type from config with logits-shape fallback."""
        distill_cfg = self.config.get('distillation', {}) if isinstance(self.config, dict) else {}
        task_type = None
        if isinstance(distill_cfg, dict):
            task_type = distill_cfg.get('task_type')
        if not task_type and isinstance(self.config, dict):
            task_type = self.config.get('task_type')

        if isinstance(task_type, str) and task_type.strip():
            return task_type.strip().lower()

        if isinstance(logits, torch.Tensor) and logits.dim() == 3:
            return 'causal_lm'
        return 'classification'

    @staticmethod
    def _extract_logits_tensor(outputs: Any) -> torch.Tensor:
        """Extract logits tensor from common output formats."""
        if isinstance(outputs, dict):
            return outputs['logits']
        if hasattr(outputs, 'logits'):
            return outputs.logits
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    @staticmethod
    def _flatten_lm_logits_and_targets(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_index: int = -100,
        shift_labels: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten LM logits/labels for token-level CE/KD computations."""
        lm_logits = logits
        lm_targets = targets

        if shift_labels and lm_logits.dim() == 3 and lm_targets.dim() >= 2:
            lm_logits = lm_logits[:, :-1, :].contiguous()
            lm_targets = lm_targets[:, 1:].contiguous()

        if lm_logits.dim() != 3:
            raise ValueError(f"Expected LM logits [B,T,V], got {tuple(lm_logits.shape)}")

        flat_logits = lm_logits.view(-1, lm_logits.size(-1))
        flat_targets = lm_targets.reshape(-1)
        valid_mask = flat_targets != ignore_index
        if valid_mask.any():
            flat_logits = flat_logits[valid_mask]
            flat_targets = flat_targets[valid_mask]
        else:
            flat_logits = flat_logits[:0]
            flat_targets = flat_targets[:0]

        return flat_logits, flat_targets
    
    def _init_optimizers(
        self,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'cosine',
        **kwargs
    ) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Initialize optimizer and learning rate scheduler.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            optimizer_type: "adamw", "sgd", or "adam"
            scheduler_type: "cosine", "step", or "none"
            **kwargs: Additional optimizer/scheduler arguments
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Get student parameters
        params = [p for p in self.student.parameters() if p.requires_grad]
        
        # Extract scheduler-specific kwargs
        total_steps = kwargs.pop('total_steps', 1000)
        step_size = kwargs.pop('step_size', 100)
        gamma = kwargs.pop('gamma', 0.1)
        
        # Create optimizer (remaining kwargs go to optimizer)
        if optimizer_type.lower() == 'adamw':
            optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs)
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Create scheduler
        scheduler: Optional[_LRScheduler] = None
        if scheduler_type.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)  # type: ignore[assignment]
        elif scheduler_type.lower() == 'step':
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # type: ignore[assignment]
        
        return optimizer, scheduler
    
    # ============================================================================
    # FORWARD PASS
    # ============================================================================
    
    def forward(
        self,
        inputs: Any,
        return_features: bool = False,
        **kwargs
    ) -> Union[Tuple[Any, Any], Tuple[Any, Any, Dict, Dict]]:
        """
        Forward pass through teacher and student.
        
        Args:
            inputs: Input data (dict for transformers, tensor for CNNs)
            return_features: Whether to return extracted features
            **kwargs: Additional arguments for model forward
            
        Returns:
            If return_features=False: (student_outputs, teacher_outputs)
            If return_features=True: (student_outputs, teacher_outputs, teacher_features, student_features)
        """
        # Clear previous features
        self.teacher_hooks.clear()
        self.student_hooks.clear()
        
        # Ensure inputs/kwargs live on the desired device (important when tests bypass training_step)
        inputs = self._move_to_device(inputs)
        kwargs = {k: self._move_to_device(v) for k, v in kwargs.items()}

        # Teacher forward (no gradients)
        with torch.no_grad():
            if isinstance(inputs, dict):
                teacher_outputs = self.teacher(**inputs, **kwargs)
            else:
                teacher_outputs = self.teacher(inputs, **kwargs)
        
        # Student forward (with gradients)
        if isinstance(inputs, dict):
            student_outputs = self.student(**inputs, **kwargs)
        else:
            student_outputs = self.student(inputs, **kwargs)
        
        if return_features:
            teacher_features = self._collect_features(self.teacher_hooks)
            student_features = self._collect_features(self.student_hooks)
            return student_outputs, teacher_outputs, teacher_features, student_features
        
        return student_outputs, teacher_outputs
    
    # ============================================================================
    # LOSS COMPUTATION (TO BE IMPLEMENTED BY SUBCLASSES)
    # ============================================================================
    
    def compute_loss(
        self,
        student_outputs: Any,
        teacher_outputs: Any,
        targets: Optional[torch.Tensor] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss (MUST BE IMPLEMENTED BY SUBCLASSES).
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            targets: Ground truth labels (optional)
            student_features: Hooked student features (optional)
            teacher_features: Hooked teacher features (optional)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (total_loss, loss_dict)
            - total_loss: Scalar tensor for backprop
            - loss_dict: Dictionary with loss components for logging
            
        Example:
        ```python
        def compute_loss(self, s_out, t_out, targets, **kwargs):
            # Supervised loss
            loss_ce = F.cross_entropy(s_out.logits, targets)
            
            # Distillation loss
            loss_kd = self.kd_loss(s_out.logits, t_out.logits)
            
            # Total loss
            total = loss_ce + self.config['alpha'] * loss_kd
            
            return total, {
                'supervised': loss_ce.item(),
                'kd': loss_kd.item(),
                'total': total.item()
            }
        ```
        """
        raise NotImplementedError(
            "Subclasses must implement compute_loss(). "
            "This is the core method where distillation logic lives."
        )
    
    # ============================================================================
    # TRAINING & EVALUATION STEPS
    # ============================================================================
    
    def training_step(
        self,
        batch: Union[Tuple, Dict],
        optimizer: Optional[Optimizer] = None,
        grad_clip: Optional[float] = None,
        return_outputs: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Any]]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch (tuple of (inputs, targets) or dict)
            optimizer: Optimizer to use (uses self.optimizer if None)
            grad_clip: Gradient clipping value (None = no clipping)
            return_outputs: Whether to return model outputs
            
        Returns:
            Dictionary with loss components, optionally with outputs
        """
        self.student.train()
        
        # Parse batch
        if isinstance(batch, dict):
            inputs = {k: self._move_to_device(v) for k, v in batch.items() if k != 'labels'}
            targets = batch.get('labels', None)
            if targets is not None:
                targets = self._move_to_device(targets)
        else:
            inputs, targets = batch
            inputs = self._move_to_device(inputs)
            targets = self._move_to_device(targets) if targets is not None else None
        
        # Forward pass with feature extraction
        forward_result = self.forward(inputs, return_features=True)
        if len(forward_result) == 4:
            student_outputs, teacher_outputs, teacher_features, student_features = forward_result
        else:
            student_outputs, teacher_outputs = forward_result  # type: ignore[misc]
            teacher_features, student_features = {}, {}
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=targets,
            student_features=student_features,
            teacher_features=teacher_features
        )
        
        # Backward pass
        opt = optimizer or self.optimizer
        if opt is None:
            raise ValueError("No optimizer provided. Either pass optimizer or set self.optimizer")
        
        opt.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip)
        
        opt.step()
        
        # Update scheduler if available
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update global step
        self.global_step += 1
        
        # Visualization hook (if enabled)
        if self.visualize_features and self.visualization_callback is not None:
            self.visualization_callback(teacher_features, student_features, self.global_step)
        
        if return_outputs:
            return loss_dict, student_outputs
        return loss_dict
    
    @torch.no_grad()
    def evaluation_step(
        self,
        batch: Union[Tuple, Dict],
        return_outputs: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Any]]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Evaluation batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Dictionary with loss components, optionally with outputs
        """
        self.student.eval()
        
        # Parse batch (same as training_step)
        if isinstance(batch, dict):
            inputs = {k: self._move_to_device(v) for k, v in batch.items() if k != 'labels'}
            targets = batch.get('labels', None)
            if targets is not None:
                targets = self._move_to_device(targets)
        else:
            inputs, targets = batch
            inputs = self._move_to_device(inputs)
            targets = self._move_to_device(targets) if targets is not None else None
        
        # Forward pass
        forward_result = self.forward(inputs, return_features=True)
        if len(forward_result) == 4:
            student_outputs, teacher_outputs, teacher_features, student_features = forward_result
        else:
            student_outputs, teacher_outputs = forward_result  # type: ignore[misc]
            teacher_features, student_features = {}, {}
        
        # Compute loss (no backward)
        total_loss, loss_dict = self.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=targets,
            student_features=student_features,
            teacher_features=teacher_features
        )
        
        if return_outputs:
            return loss_dict, student_outputs
        return loss_dict
    
    # ============================================================================
    # HOOKS (OPTIONAL CUSTOMIZATION POINTS)
    # ============================================================================
    
    def pre_forward_hook(self, inputs: Any, **kwargs) -> None:
        """Hook executed before forward pass. Override for custom preprocessing."""
        pass
    
    def post_forward_hook(
        self,
        inputs: Any,
        student_outputs: Any,
        teacher_outputs: Any,
        **kwargs
    ) -> None:
        """Hook executed after forward pass. Override for custom postprocessing."""
        pass
    
    # ============================================================================
    # LOGGING & METRICS
    # ============================================================================
    
    def log_metrics(self, metrics: Dict[str, float], phase: str = 'train') -> None:
        """
        Log metrics for current step.
        
        Args:
            metrics: Dictionary with metric values
            phase: "train" or "val"
        """
        for key, value in metrics.items():
            metric_name = f"{phase}_{key}"
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
    
    def get_metrics_summary(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get summary statistics for logged metrics.
        
        Args:
            last_n: Average over last N steps (None = all steps)
            
        Returns:
            Dictionary with metric averages
        """
        summary = {}
        for name, values in self.metrics.items():
            if values:
                data = values[-last_n:] if last_n else values
                summary[f"{name}_mean"] = sum(data) / len(data)
                summary[f"{name}_last"] = data[-1]
        return summary
    
    def reset_metrics(self) -> None:
        """Clear all logged metrics."""
        self.metrics = {key: [] for key in self.metrics.keys()}
    
    # ============================================================================
    # UTILITIES
    # ============================================================================
    
    def freeze_teacher(self) -> None:
        """Freeze all teacher parameters."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def unfreeze_teacher(self) -> None:
        """Unfreeze teacher parameters (for co-training scenarios)."""
        self.teacher.train()
        for param in self.teacher.parameters():
            param.requires_grad = True
    
    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.teacher_hooks.clear()
        self.student_hooks.clear()
    
    def __del__(self):
        """Cleanup: remove hooks when distiller is destroyed."""
        try:
            self.remove_hooks()
        except:
            pass  # Ignore errors during cleanup
