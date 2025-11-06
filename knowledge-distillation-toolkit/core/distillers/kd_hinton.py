"""
KD-Hinton Distiller - Advanced Knowledge Distillation with Hint-Based Learning
===============================================================================

Implements the classical Hinton KD with modern enhancements:
1. Classical Logit Distillation (Hinton et al., 2015)
2. Hint-Based Learning (FitNets, Romero et al., 2015)
3. Progressive Hint Decay & Curriculum Learning
4. Multi-Hint Distillation (hierarchical guidance)
5. Dynamic Temperature Scaling
6. Label Smoothing & Class Balancing
7. Contrastive Hint Learning (optional)

Architecture:
```
KDHintonDistiller (extends BaseDistiller)
├── Logit KD (soft targets)
├── HintModule (intermediate feature guidance)
│   ├── HintSelector (auto layer selection)
│   ├── HintRegressor (1x1 conv adapters)
│   ├── HintLossComposer (MSE, Cosine, KL)
│   └── HintScheduler (progressive decay)
├── Temperature Scheduler (dynamic T)
└── Evaluation Metrics (HFA, HGI, alignment)
```

Reference: 
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Romero et al., "FitNets: Hints for Thin Deep Nets" (2015)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from collections import OrderedDict

from .base_distiller import BaseDistiller


# ============================================================================
# HINT REGRESSOR - Learnable Feature Transformation
# ============================================================================

class HintRegressor(nn.Module):
    """
    Learnable regressor (Ψ) that maps student features to teacher hint shape.
    
    Supports multiple architectures:
    - 1x1 Conv (default, most efficient)
    - Linear projection
    - MLP (deeper transformation)
    - Attention-based projection
    """
    
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        regressor_type: str = '1x1conv',
        use_bn: bool = True,
        activation: str = 'none'
    ):
        """
        Args:
            student_dim: Student feature dimension
            teacher_dim: Teacher feature dimension (target)
            regressor_type: 'conv1x1', 'linear', 'mlp', 'attention'
            use_bn: Whether to use batch normalization
            activation: 'relu', 'gelu', 'none'
        """
        super().__init__()
        self.regressor_type = regressor_type
        
        if regressor_type == '1x1conv' or regressor_type == 'conv1x1':
            layers: List[nn.Module] = [nn.Conv2d(student_dim, teacher_dim, 1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(teacher_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            self.regressor = nn.Sequential(*layers)
        
        elif regressor_type == 'linear':
            layers: List[nn.Module] = [nn.Linear(student_dim, teacher_dim, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm1d(teacher_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            self.regressor = nn.Sequential(*layers)
        
        elif regressor_type == 'mlp':
            hidden_dim = (student_dim + teacher_dim) // 2
            self.regressor = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, teacher_dim)
            )
        
        elif regressor_type == 'attention':
            # Attention-based transformation
            self.query = nn.Linear(student_dim, teacher_dim)
            self.key = nn.Linear(student_dim, teacher_dim)
            self.value = nn.Linear(student_dim, teacher_dim)
            self.scale = teacher_dim ** -0.5
        
        else:
            raise ValueError(f"Unknown regressor type: {regressor_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform student features to teacher hint space."""
        if self.regressor_type == 'attention':
            # x: [B, C, H, W] or [B, L, C]
            if len(x.shape) == 4:
                b, c, h, w = x.shape
                x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
                
                Q = self.query(x_flat)
                K = self.key(x_flat)
                V = self.value(x_flat)
                
                attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
                out = attn @ V
                out = out.transpose(1, 2).reshape(b, -1, h, w)
                return out
            else:
                x_flat = x
                Q = self.query(x_flat)
                K = self.key(x_flat)
                V = self.value(x_flat)
                
                attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
                out = attn @ V
                return out
        else:
            return self.regressor(x)


# ============================================================================
# HINT SCHEDULER - Progressive Hint Weight Decay
# ============================================================================

class HintScheduler:
    """
    Dynamically adjusts hint loss weight during training.
    
    Supports curriculum learning strategies:
    - constant: Fixed weight
    - linear_decay: Gradually decrease hint importance
    - exponential_decay: Rapid early guidance, gentle fade
    - cosine_decay: Smooth cosine annealing
    - adaptive: Based on validation performance
    """
    
    def __init__(
        self,
        initial_weight: float = 1.0,
        final_weight: float = 0.1,
        total_steps: int = 10000,
        scheduler_type: str = 'linear_decay'
    ):
        """
        Args:
            initial_weight: Starting hint weight
            final_weight: Ending hint weight
            total_steps: Total training steps
            scheduler_type: Scheduling strategy
        """
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.current_step = 0
    
    def step(self) -> float:
        """Get current hint weight and advance step."""
        if self.scheduler_type == 'constant':
            weight = self.initial_weight
        
        elif self.scheduler_type == 'linear_decay':
            progress = min(self.current_step / self.total_steps, 1.0)
            weight = self.initial_weight + progress * (self.final_weight - self.initial_weight)
        
        elif self.scheduler_type == 'exponential_decay':
            decay_rate = np.log(self.final_weight / self.initial_weight) / self.total_steps
            weight = self.initial_weight * np.exp(decay_rate * self.current_step)
        
        elif self.scheduler_type == 'cosine_decay':
            progress = min(self.current_step / self.total_steps, 1.0)
            weight = self.final_weight + 0.5 * (self.initial_weight - self.final_weight) * \
                     (1 + np.cos(np.pi * progress))
        
        else:
            weight = self.initial_weight
        
        self.current_step += 1
        return weight
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0


# ============================================================================
# TEMPERATURE SCHEDULER - Dynamic Temperature Scaling
# ============================================================================

class TemperatureScheduler:
    """
    Dynamically adjusts distillation temperature during training.
    
    Higher T → softer distributions (early training)
    Lower T → sharper distributions (late training)
    """
    
    def __init__(
        self,
        initial_temp: float = 4.0,
        final_temp: float = 1.0,
        total_steps: int = 10000,
        scheduler_type: str = 'cosine'
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.current_step = 0
    
    def step(self) -> float:
        """Get current temperature and advance step."""
        if self.scheduler_type == 'constant':
            temp = self.initial_temp
        
        elif self.scheduler_type == 'linear':
            progress = min(self.current_step / self.total_steps, 1.0)
            temp = self.initial_temp + progress * (self.final_temp - self.initial_temp)
        
        elif self.scheduler_type == 'cosine':
            progress = min(self.current_step / self.total_steps, 1.0)
            temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * \
                   (1 + np.cos(np.pi * progress))
        
        else:
            temp = self.initial_temp
        
        self.current_step += 1
        return temp
    
    def reset(self):
        self.current_step = 0


# ============================================================================
# HINT LOSS FUNCTIONS
# ============================================================================

class HintLossFunctions:
    """Collection of hint loss functions."""
    
    @staticmethod
    def mse_loss(hint_t: torch.Tensor, hint_s: torch.Tensor) -> torch.Tensor:
        """L2 loss between teacher hint and student projection."""
        return F.mse_loss(hint_s, hint_t)
    
    @staticmethod
    def cosine_loss(hint_t: torch.Tensor, hint_s: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss."""
        hint_t_flat = hint_t.flatten(2).mean(dim=2)  # [B, C]
        hint_s_flat = hint_s.flatten(2).mean(dim=2)
        
        cos_sim = F.cosine_similarity(hint_t_flat, hint_s_flat, dim=1)
        return (1 - cos_sim).mean()
    
    @staticmethod
    def kl_loss(hint_t: torch.Tensor, hint_s: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """KL divergence on normalized feature distributions."""
        # Normalize to distributions
        hint_t_soft = F.softmax(hint_t.flatten(1) / temperature, dim=1)
        hint_s_soft = F.log_softmax(hint_s.flatten(1) / temperature, dim=1)
        
        return F.kl_div(hint_s_soft, hint_t_soft, reduction='batchmean') * (temperature ** 2)
    
    @staticmethod
    def contrastive_hint_loss(
        hint_t: torch.Tensor,
        hint_s: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Contrastive loss for hint learning.
        Encourages same hints for similar samples.
        """
        # Flatten and normalize
        hint_t_flat = F.normalize(hint_t.flatten(1), dim=1)
        hint_s_flat = F.normalize(hint_s.flatten(1), dim=1)
        
        # Compute similarity matrix
        logits = torch.mm(hint_s_flat, hint_t_flat.T) / temperature
        
        # Diagonal elements are positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)
        
        return F.cross_entropy(logits, labels)


# ============================================================================
# MAIN KD-HINTON DISTILLER - Extended with Hint Learning
# ============================================================================

class KDHintonDistiller(BaseDistiller):
    """
    Advanced KD-Hinton distiller with hint-based intermediate layer guidance.
    
    Features:
    1. Classical Hinton KD (soft logits)
    2. Multi-hint distillation (FitNets-style)
    3. Progressive hint weight scheduling
    4. Dynamic temperature scaling
    5. Multiple hint loss functions (MSE, Cosine, KL, Contrastive)
    6. Cross-architecture support (CNN↔Transformer)
    7. Label smoothing & class balancing
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize KD-Hinton distiller with hint learning.
        
        Config structure:
        ```yaml
        kd_hinton:
          # Classical KD settings
          temperature: 4.0
          alpha: 0.7  # Weight for KD loss (1-alpha for CE)
          
          # Hint distillation
          hint_enabled: true
          hints:
            - teacher: "encoder.layer.6"
              student: "encoder.layer.3"
              weight: 0.5
              regressor: "1x1conv"
              loss: "mse"
            - teacher: "encoder.layer.10"
              student: "encoder.layer.5"
              weight: 0.3
              regressor: "1x1conv"
              loss: "cosine"
          
          # Schedulers
          hint_scheduler:
            type: "cosine_decay"
            initial_weight: 1.0
            final_weight: 0.1
            total_steps: 10000
          
          temperature_scheduler:
            type: "cosine"
            initial_temp: 4.0
            final_temp: 2.0
            total_steps: 10000
          
          # Loss settings
          label_smoothing: 0.1
          class_weights: [0.9766, 1.0246]  # For imbalanced datasets
        ```
        """
        # Parse config before super().__init__
        if config is None:
            config = {}
        
        kd_config = config.get('kd_hinton', {})
        self.hint_enabled = kd_config.get('hint_enabled', False)
        self.hint_configs = kd_config.get('hints', [])
        
        # Call parent
        super().__init__(teacher, student, config, device, **kwargs)
        
        # Classical KD parameters
        self.temperature = kd_config.get('temperature', 4.0)
        self.alpha = kd_config.get('alpha', 0.7)
        self.label_smoothing = kd_config.get('label_smoothing', 0.0)
        
        # Class weights for imbalanced datasets
        class_weights = kd_config.get('class_weights', None)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        else:
            self.class_weights = None
        
        # Initialize hint regressors
        self.hint_regressors = nn.ModuleDict()
        if self.hint_enabled:
            self._init_hint_regressors()
        
        # Initialize schedulers
        hint_sched_cfg = kd_config.get('hint_scheduler', {})
        self.hint_scheduler = HintScheduler(
            initial_weight=hint_sched_cfg.get('initial_weight', 1.0),
            final_weight=hint_sched_cfg.get('final_weight', 0.1),
            total_steps=hint_sched_cfg.get('total_steps', 10000),
            scheduler_type=hint_sched_cfg.get('type', 'cosine_decay')
        )
        
        temp_sched_cfg = kd_config.get('temperature_scheduler', {})
        self.temp_scheduler = TemperatureScheduler(
            initial_temp=temp_sched_cfg.get('initial_temp', self.temperature),
            final_temp=temp_sched_cfg.get('final_temp', 2.0),
            total_steps=temp_sched_cfg.get('total_steps', 10000),
            scheduler_type=temp_sched_cfg.get('type', 'cosine')
        )
        
        # Hint loss function mapping
        self.hint_loss_fns = {
            'mse': HintLossFunctions.mse_loss,
            'cosine': HintLossFunctions.cosine_loss,
            'kl': lambda h_t, h_s: HintLossFunctions.kl_loss(h_t, h_s, self.temperature),
            'contrastive': lambda h_t, h_s: HintLossFunctions.contrastive_hint_loss(h_t, h_s, 0.07)
        }
    
    def _init_hint_regressors(self):
        """Initialize hint regressors for each hint pair."""
        teacher_modules = dict(self.teacher.named_modules())
        student_modules = dict(self.student.named_modules())
        
        for i, hint_cfg in enumerate(self.hint_configs):
            t_layer_name = hint_cfg['teacher']
            s_layer_name = hint_cfg['student']
            
            # Get layer dimensions (this is simplified, real implementation needs shape inference)
            # In practice, you'd do a forward pass to get actual shapes
            regressor_type = hint_cfg.get('regressor', '1x1conv')
            
            # Create a unique key for this hint pair
            hint_key = f"hint_{i}_{s_layer_name}"
            
            # For now, create a placeholder regressor
            # In production, you'd infer dimensions from a forward pass
            # self.hint_regressors[hint_key] = HintRegressor(
            #     student_dim=inferred_s_dim,
            #     teacher_dim=inferred_t_dim,
            #     regressor_type=regressor_type
            # )
    
    def _register_hooks(self):
        """Register hooks for hint layers."""
        if not self.hint_enabled or not self.hint_configs:
            return
        
        teacher_modules = dict(self.teacher.named_modules())
        student_modules = dict(self.student.named_modules())
        
        for hint_cfg in self.hint_configs:
            t_layer_name = hint_cfg['teacher']
            s_layer_name = hint_cfg['student']
            
            if t_layer_name in teacher_modules:
                t_handle = teacher_modules[t_layer_name].register_forward_hook(
                    self._get_teacher_hook(t_layer_name)
                )
                self._hook_handles.append(t_handle)
            
            if s_layer_name in student_modules:
                s_handle = student_modules[s_layer_name].register_forward_hook(
                    self._get_student_hook(s_layer_name)
                )
                self._hook_handles.append(s_handle)
    
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
        Compute total KD loss with hint guidance.
        
        L_total = α * L_KD + (1-α) * L_CE + β * L_hint
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Extract logits - handle dict, object with logits attr, or tensor
        if isinstance(student_outputs, dict):
            student_logits = student_outputs['logits']
            teacher_logits = teacher_outputs['logits']
        elif hasattr(student_outputs, 'logits'):
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
        else:
            student_logits = student_outputs
            teacher_logits = teacher_outputs
        
        # Get current temperature
        current_temp = self.temp_scheduler.step() if self.training else self.temperature
        
        # 1. Classical Hinton KD Loss
        student_soft = F.log_softmax(student_logits / current_temp, dim=1)
        teacher_soft = F.softmax(teacher_logits / current_temp, dim=1)
        
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (current_temp ** 2)
        total_loss += self.alpha * kd_loss
        loss_dict['kd_loss'] = kd_loss.item()
        
        # 2. Supervised Cross-Entropy Loss
        if targets is not None:
            if self.label_smoothing > 0:
                ce_loss = self._label_smoothed_cross_entropy(student_logits, targets)
            else:
                if self.class_weights is not None:
                    ce_loss = F.cross_entropy(student_logits, targets, weight=self.class_weights)
                else:
                    ce_loss = F.cross_entropy(student_logits, targets)
            
            total_loss += (1 - self.alpha) * ce_loss
            loss_dict['ce_loss'] = ce_loss.item()
        
        # 3. Hint-Based Intermediate Layer Guidance
        if self.hint_enabled and teacher_features and student_features:
            hint_loss_total = 0.0
            current_hint_weight = self.hint_scheduler.step() if self.training else 1.0
            
            for i, hint_cfg in enumerate(self.hint_configs):
                t_layer = hint_cfg['teacher']
                s_layer = hint_cfg['student']
                hint_weight = hint_cfg.get('weight', 1.0)
                loss_type = hint_cfg.get('loss', 'mse')
                
                if t_layer in teacher_features and s_layer in student_features:
                    hint_t = teacher_features[t_layer]
                    hint_s = student_features[s_layer]
                    
                    # Apply regressor if available
                    hint_key = f"hint_{i}_{s_layer}"
                    if hint_key in self.hint_regressors:
                        hint_s = self.hint_regressors[hint_key](hint_s)
                    
                    # Align spatial dimensions if needed
                    if hint_t.shape[2:] != hint_s.shape[2:]:
                        hint_s = F.interpolate(
                            hint_s,
                            size=hint_t.shape[2:],
                            mode='bilinear' if len(hint_t.shape) == 4 else 'linear',
                            align_corners=False
                        )
                    
                    # Align channel dimensions if needed (create adapter on-the-fly)
                    if hint_t.shape[1] != hint_s.shape[1]:
                        # Create temporary adapter for channel alignment
                        adapter = nn.Conv2d(
                            hint_s.shape[1],
                            hint_t.shape[1],
                            1,
                            bias=False
                        ).to(self.device)
                        hint_s = adapter(hint_s)
                    
                    # Compute hint loss
                    if loss_type in self.hint_loss_fns:
                        hint_loss = self.hint_loss_fns[loss_type](hint_t, hint_s)
                        weighted_hint_loss = hint_weight * current_hint_weight * hint_loss
                        hint_loss_total += weighted_hint_loss
                        
                        loss_dict[f'hint_{i}_{loss_type}'] = hint_loss.item()
            
            if hint_loss_total > 0:
                total_loss += hint_loss_total
                # Ensure hint_loss_total is tensor before calling .item()
                if isinstance(hint_loss_total, torch.Tensor):
                    loss_dict['hint_total'] = hint_loss_total.item()
                else:
                    loss_dict['hint_total'] = float(hint_loss_total)
                loss_dict['hint_weight'] = current_hint_weight
        
        loss_dict['temperature'] = current_temp
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _label_smoothed_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy with label smoothing."""
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot with smoothing
        smooth_targets = torch.zeros_like(log_probs).scatter_(
            -1, targets.unsqueeze(-1), 1.0
        )
        smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                        self.label_smoothing / n_classes
        
        return -(smooth_targets * log_probs).sum(dim=-1).mean()
    
    @torch.no_grad()
    def compute_hint_alignment_metrics(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute Hint Feature Alignment (HFA) metrics.
        
        Returns:
            Dictionary with cosine similarity and L2 distance per hint
        """
        metrics = {}
        
        for i, hint_cfg in enumerate(self.hint_configs):
            t_layer = hint_cfg['teacher']
            s_layer = hint_cfg['student']
            
            if t_layer in teacher_features and s_layer in student_features:
                hint_t = teacher_features[t_layer]
                hint_s = student_features[s_layer]
                
                # Apply regressor if available
                hint_key = f"hint_{i}_{s_layer}"
                if hint_key in self.hint_regressors:
                    hint_s = self.hint_regressors[hint_key](hint_s)
                
                # Align spatial dimensions
                if hint_t.shape[2:] != hint_s.shape[2:]:
                    hint_s = F.interpolate(hint_s, size=hint_t.shape[2:], mode='bilinear', align_corners=False)
                
                # Align channel dimensions if needed
                if hint_t.shape[1] != hint_s.shape[1]:
                    adapter = nn.Conv2d(
                        hint_s.shape[1],
                        hint_t.shape[1],
                        1,
                        bias=False
                    ).to(self.device)
                    hint_s = adapter(hint_s)
                
                # Cosine similarity
                hint_t_flat = hint_t.flatten(1)
                hint_s_flat = hint_s.flatten(1)
                cos_sim = F.cosine_similarity(hint_t_flat, hint_s_flat, dim=1).mean()
                metrics[f'hint_{i}_cosine'] = cos_sim.item()
                
                # L2 distance
                l2_dist = F.mse_loss(hint_s, hint_t).item()
                metrics[f'hint_{i}_l2'] = l2_dist
        
        if metrics:
            # Average HFA score
            cos_scores = [v for k, v in metrics.items() if 'cosine' in k]
            metrics['avg_hfa'] = np.mean(cos_scores) if cos_scores else 0.0
        
        return metrics


# ============================================================================
# LEGACY COMPATIBILITY WRAPPER
# ============================================================================

class LegacyKDHintonDistiller:
    """Legacy KD-Hinton distiller for backward compatibility."""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, class_weights=None):
        self.temperature = temperature
        self.alpha = alpha
        # Default class weights for binary classification (adjust based on data distribution)
        if class_weights is None:
            self.class_weights = torch.tensor([0.9766, 1.0246], dtype=torch.float32)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, student_logits, teacher_logits, labels, loss_fn_ce=None):
        """
        Compute the knowledge distillation loss as described by Hinton et al.
        """
        T = self.temperature
        alpha = self.alpha

        # Soft targets with temperature scaling
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL divergence loss for distillation
        kl_div = torch.nn.KLDivLoss(reduction="batchmean")
        distill_loss = kl_div(student_soft, teacher_soft) * (T * T)

        # Class-balanced cross-entropy loss with ground truth
        if loss_fn_ce is None:
            # Move class weights to the same device as the logits
            device = student_logits.device
            class_weights = self.class_weights.to(device)
            loss_fn_ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        ce_loss = loss_fn_ce(student_logits, labels)

        # Weighted sum
        return alpha * distill_loss + (1 - alpha) * ce_loss