"""
Zynthé Toolkit - Advanced Optimizer System
===========================================

Smart optimizer system with:
- Multi-optimizer support (AdamW, Lion, SGD, Adam, etc.)
- Phase-aware optimization (Distillation, Quantization, Fine-tuning)
- Gradient management (clipping, centralization, noise injection)
- Adaptive LR tuning based on distillation metrics (DEI, CAS)
- Parameter grouping (encoder vs classifier, layer-wise)
    - Checkpoint compatibility

Author: Zynthé Team
License: MIT
"""

from __future__ import annotations


import torch
import torch.optim as optim
from torch.optim import Optimizer
from typing import Dict, List, Optional, Any, DefaultDict
import logging
import math
from collections import defaultdict

LOG = logging.getLogger(__name__)


# =============================================================================
# Optimizer Factory - Phase-Aware Optimizer Creation
# =============================================================================


class OptimizerFactory:
    """
    Factory for creating optimizers with phase-aware configurations.
    Supports: AdamW, Lion, SGD, Adam, AdamW8bit
    """

    SUPPORTED_OPTIMIZERS = {
        "adamw": "AdamW with decoupled weight decay",
        "adam": "Standard Adam optimizer",
        "sgd": "SGD with momentum and Nesterov",
        "lion": "Lion optimizer (memory efficient)",
        "adamw8bit": "8-bit AdamW for large models",
    }

    @staticmethod
    def get_optimizer(
        model: torch.nn.Module, config: Dict[str, Any], phase: str = "distillation"
    ) -> Optimizer:
        """
        Create optimizer based on config and training phase.

        Args:
            model: Model to optimize
            config: Configuration dict with optimizer settings
            phase: Training phase ('distillation', 'quantization', 'finetuning')

        Returns:
            Configured optimizer
        """
        optimizer_name = str(config.get("optimizer", "adamw")).lower()
        raw_lr = config.get("learning_rate", config.get("lr", 2e-5))
        try:
            lr = float(raw_lr)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid learning rate value: {raw_lr!r}") from exc

        raw_weight_decay = config.get("weight_decay", 0.01)
        try:
            weight_decay = float(raw_weight_decay)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight_decay value: {raw_weight_decay!r}") from exc

        # Phase-aware learning rate adjustment
        phase_lr_multipliers = {
            "distillation": 1.0,  # Standard LR for distillation
            "quantization": 0.5,  # Lower LR for QAT stability
            "finetuning": 1.5,  # Higher LR for final tuning
        }
        lr *= phase_lr_multipliers.get(phase, 1.0)

        LOG.info(f"Creating {optimizer_name} optimizer for {phase} phase")
        LOG.info(f"Learning rate: {lr}, Weight decay: {weight_decay}")

        # Parameter grouping for differential learning rates
        param_groups = OptimizerFactory._create_param_groups(
            model,
            base_lr=lr,
            weight_decay=weight_decay,
            use_layer_wise=config.get("layer_wise_lr", False),
        )

        # Select optimizer
        if optimizer_name == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=lr,
                betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
                eps=float(config.get("eps", 1e-8)),
                weight_decay=weight_decay,
            )

        elif optimizer_name == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=lr,
                betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
                eps=float(config.get("eps", 1e-8)),
                weight_decay=weight_decay,
            )

        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                param_groups,
                lr=lr,
                momentum=float(config.get("momentum", 0.9)),
                weight_decay=weight_decay,
                nesterov=config.get("nesterov", True),
            )

        elif optimizer_name == "lion":
            # Lion optimizer - memory efficient alternative to AdamW
            try:
                from lion_pytorch import Lion  # type: ignore[import]

                optimizer = Lion(
                    param_groups,
                    lr=lr * 0.3,  # Lion uses ~3x smaller LR
                    betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.99))),
                    weight_decay=weight_decay,
                )
                LOG.info("Using Lion optimizer (memory efficient)")
            except ImportError:
                LOG.warning(
                    "Lion optimizer not available; install 'lion-pytorch'. Falling back to AdamW"
                )
                optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        elif optimizer_name == "adamw8bit":
            # 8-bit AdamW for large models
            try:
                import bitsandbytes as bnb  # type: ignore[import]

                optimizer = bnb.optim.AdamW8bit(
                    param_groups,
                    lr=lr,
                    betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
                    eps=float(config.get("eps", 1e-8)),
                    weight_decay=weight_decay,
                )
                LOG.info("Using 8-bit AdamW optimizer")
            except ImportError:
                LOG.warning("8-bit AdamW not available, falling back to standard AdamW")
                optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        else:
            LOG.warning(f"Unknown optimizer '{optimizer_name}', using AdamW")
            optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        return optimizer

    @staticmethod
    def _create_param_groups(
        model: torch.nn.Module, base_lr: float, weight_decay: float, use_layer_wise: bool = False
    ) -> List[Dict]:
        """
        Create parameter groups with differential learning rates.

        Args:
            model: Model to group parameters for
            base_lr: Base learning rate
            weight_decay: Weight decay
            use_layer_wise: Use layer-wise learning rate decay

        Returns:
            List of parameter groups
        """
        if not use_layer_wise:
            # Simple grouping: no decay for bias and layer norm
            decay_params = []
            no_decay_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            return [
                {"params": decay_params, "weight_decay": weight_decay, "lr": base_lr},
                {"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr},
            ]

        # Layer-wise learning rate decay (LLRD)
        # Earlier layers get smaller learning rates
        param_groups = []

        # Try to detect model structure
        if hasattr(model, "encoder") or hasattr(model, "bert") or hasattr(model, "roberta"):
            # Transformer model - use layer-wise decay
            num_layers = 12  # Default for BERT-base
            # Type guard: Check if model has config and num_hidden_layers
            model_config = getattr(model, "config", None)
            if (
                model_config is not None
                and hasattr(model_config, "num_hidden_layers")
                and isinstance(model_config.num_hidden_layers, int)
            ):
                num_layers = model_config.num_hidden_layers

            decay_rate = 0.95  # Each layer gets 95% of the next layer's LR

            # Group by layer depth
            for layer_idx in range(num_layers):
                layer_lr = base_lr * (decay_rate ** (num_layers - layer_idx - 1))
                layer_params = []

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if f"layer.{layer_idx}" in name or f"layers.{layer_idx}" in name:
                        layer_params.append(param)

                if layer_params:
                    param_groups.append(
                        {"params": layer_params, "lr": layer_lr, "weight_decay": weight_decay}
                    )

            # Classifier head gets full learning rate
            classifier_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "classifier" in name or "head" in name or "pooler" in name:
                    classifier_params.append(param)

            if classifier_params:
                param_groups.append(
                    {"params": classifier_params, "lr": base_lr, "weight_decay": weight_decay}
                )
        else:
            # Fallback to simple grouping
            return OptimizerFactory._create_param_groups(
                model, base_lr, weight_decay, use_layer_wise=False
            )

        if not param_groups:
            # Fallback if structure detection failed
            param_groups = [
                {"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}
            ]

        LOG.info(f"Created {len(param_groups)} parameter groups with layer-wise LR")
        return param_groups


# =============================================================================
# Gradient Management Tools
# =============================================================================


class GradientManager:
    """
    Utilities for safe and efficient gradient management.
    """

    @staticmethod
    def clip_gradients(
        model: torch.nn.Module, max_norm: float = 1.0, norm_type: float = 2.0
    ) -> float:
        """
        Clip gradients to prevent exploding gradients.

        Args:
            model: Model with gradients
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2)

        Returns:
            Total gradient norm before clipping
        """
        if max_norm <= 0:
            return 0.0

        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=max_norm, norm_type=norm_type
        )

        return total_norm.item()

    @staticmethod
    def centralize_gradients(model: torch.nn.Module) -> None:
        """
        Centralize gradients to improve distillation training stability.

        This technique subtracts the mean of each gradient tensor,
        which can help with convergence in knowledge distillation.

        Args:
            model: Model with gradients to centralize
        """
        for param in model.parameters():
            if param.grad is not None and param.grad.dim() > 1:
                # Centralize by subtracting mean
                param.grad.data = param.grad.data - param.grad.data.mean(
                    dim=tuple(range(1, param.grad.dim())), keepdim=True
                )

    @staticmethod
    def inject_gradient_noise(
        model: torch.nn.Module, noise_scale: float = 0.01, noise_decay: float = 0.55
    ) -> None:
        """
        Add small noise to gradients during QAT for improved robustness.

        Args:
            model: Model with gradients
            noise_scale: Scale of Gaussian noise
            noise_decay: Decay factor for noise over time
        """
        if noise_scale <= 0:
            return

        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.data.add_(noise)

    @staticmethod
    def get_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
        """
        Compute gradient statistics for monitoring.

        Args:
            model: Model with gradients

        Returns:
            Dict with gradient norm, mean, std
        """
        total_norm = 0.0
        total_mean = 0.0
        total_std = 0.0
        num_params = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2
                total_mean += param.grad.data.mean().item()
                total_std += param.grad.data.std().item()
                num_params += 1

        if num_params == 0:
            return {"grad_norm": 0.0, "grad_mean": 0.0, "grad_std": 0.0}

        return {
            "grad_norm": math.sqrt(total_norm),
            "grad_mean": total_mean / num_params,
            "grad_std": total_std / num_params,
        }


# =============================================================================
# Adaptive Optimization Control
# =============================================================================


class AdaptiveOptimizer:
    """
    Dynamically adjust optimizer based on evaluation metrics.
    Uses DEI (Distillation Efficacy Index) and CAS (Compression-Aware Score).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        enable_auto_tune: bool = True,
        patience: int = 3,
        factor: float = 0.5,
        min_lr: float = 1e-7,
    ):
        """
        Args:
            optimizer: Optimizer to adapt
            enable_auto_tune: Enable automatic tuning
            patience: Epochs to wait before adjusting
            factor: Factor to multiply LR by
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.enable_auto_tune = enable_auto_tune
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

        self.best_metric = float("-inf")
        self.wait_count = 0
        self.history: list[float] = []

    def auto_tune(self, metrics: Dict[str, float], epoch: int) -> Dict[str, Any]:
        """
        Automatically tune learning rate based on metrics.

        Args:
            metrics: Evaluation metrics (should include 'dei', 'cas', or 'accuracy')
            epoch: Current epoch

        Returns:
            Dict with tuning actions taken
        """
        if not self.enable_auto_tune:
            return {"action": "disabled"}

        # Extract key metric (prefer DEI, fallback to accuracy)
        key_metric = metrics.get("dei", metrics.get("accuracy", 0.0))

        actions = {
            "epoch": epoch,
            "metric_value": key_metric,
            "action": "none",
            "lr_changed": False,
            "old_lr": self._get_current_lr(),
            "new_lr": self._get_current_lr(),
        }

        # Track history
        self.history.append(key_metric)

        # Rule 1: If DEI < 0.8, drastically reduce LR (poor distillation)
        dei = metrics.get("dei", None)
        if dei is not None and dei < 0.8:
            self._adjust_lr(factor=0.5)
            actions["action"] = "dei_emergency_reduction"
            actions["lr_changed"] = True
            actions["new_lr"] = self._get_current_lr()
            LOG.warning(f"DEI too low ({dei:.4f}), reducing LR by 50%")
            return actions

        # Rule 2: If CAS is improving rapidly, slightly increase LR
        cas = metrics.get("cas", None)
        if cas is not None and len(self.history) >= 2:
            if self.history[-1] > self.history[-2] * 1.2:  # 20% improvement
                self._adjust_lr(factor=1.1)
                actions["action"] = "cas_boost"
                actions["lr_changed"] = True
                actions["new_lr"] = self._get_current_lr()
                LOG.info("CAS improving rapidly, increasing LR by 10%")
                return actions

        # Rule 3: Standard plateau detection
        if key_metric > self.best_metric:
            self.best_metric = key_metric
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self._adjust_lr(factor=self.factor)
            actions["action"] = "plateau_reduction"
            actions["lr_changed"] = True
            actions["new_lr"] = self._get_current_lr()
            self.wait_count = 0
            LOG.info(f"Plateau detected, reducing LR to {actions['new_lr']:.2e}")

        return actions

    def _adjust_lr(self, factor: float) -> None:
        """Adjust learning rate by factor."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * factor, self.min_lr)
            param_group["lr"] = new_lr

    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


# =============================================================================
# Checkpoint Support
# =============================================================================


class OptimizerCheckpoint:
    """
    Save and load optimizer state for fault-tolerant training.
    """

    @staticmethod
    def save_checkpoint(
        optimizer: Optimizer, path: str, epoch: int, best_metric: Optional[float] = None, **kwargs
    ) -> None:
        """
        Save optimizer checkpoint.

        Args:
            optimizer: Optimizer to save
            path: Path to save checkpoint
            epoch: Current epoch
            best_metric: Best metric achieved
            **kwargs: Additional data to save
        """
        checkpoint = {
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            **kwargs,
        }
        torch.save(checkpoint, path)
        LOG.info(f"Saved optimizer checkpoint to {path}")

    @staticmethod
    def load_checkpoint(optimizer: Optimizer, path: str) -> Dict[str, Any]:
        """
        Load optimizer checkpoint.

        Args:
            optimizer: Optimizer to load state into
            path: Path to checkpoint

        Returns:
            Dict with checkpoint metadata
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "best_metric": checkpoint.get("best_metric", None),
        }

        LOG.info(f"Loaded optimizer checkpoint from {path} (epoch {metadata['epoch']})")
        return metadata


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================


def get_optimizer(
    model: torch.nn.Module,
    config: Optional[Dict] = None,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    phase: str = "distillation",
) -> Optimizer:
    """
    Convenience function for creating optimizer.

    Args:
        model: Model to optimize
        config: Configuration dict (if None, uses lr and weight_decay)
        lr: Learning rate
        weight_decay: Weight decay
        phase: Training phase

    Returns:
        Configured optimizer
    """
    if config is None:
        config = {"optimizer": "adamw", "learning_rate": lr, "weight_decay": weight_decay}

    return OptimizerFactory.get_optimizer(model, config, phase)


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """Convenience function for gradient clipping."""
    return GradientManager.clip_gradients(model, max_norm)


def centralize_gradients(model: torch.nn.Module) -> None:
    """Convenience function for gradient centralization."""
    GradientManager.centralize_gradients(model)


def inject_gradient_noise(model: torch.nn.Module, noise_scale: float = 0.01) -> None:
    """Convenience function for gradient noise injection."""
    GradientManager.inject_gradient_noise(model, noise_scale)


# =============================================================================
# Lookahead Wrapper (Future Enhancement)
# =============================================================================


class LookaheadOptimizer(Optimizer):
    """
    Lookahead optimizer wrapper for improved convergence.
    Reference: https://arxiv.org/abs/1907.08610

    Usage:
        base_opt = OptimizerFactory.get_optimizer(model, config)
        optimizer = LookaheadOptimizer(base_opt, k=5, alpha=0.5)
    """

    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        """
        Args:
            optimizer: Base optimizer
            k: Number of steps before slow weights update
            alpha: Slow weights step size
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        # Use defaultdict to match base class type
        self.state: DefaultDict[Any, Any] = defaultdict(dict)

        # Cache slow weights
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["slow_buffer"] = torch.empty_like(p.data)
                param_state["slow_buffer"].copy_(p.data)

        self.step_count = 0

    def step(self, closure=None) -> None:  # type: ignore[override]
        """Perform optimization step."""
        loss = self.optimizer.step(closure)
        self.step_count += 1

        if self.step_count % self.k == 0:
            # Update slow weights
            for group in self.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    slow = param_state["slow_buffer"]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)

        # Note: Base optimizer returns None, but we need the loss value for monitoring
        return loss  # type: ignore[return-value]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return state dict."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "lookahead": self.state,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        # Load lookahead state separately to avoid type conflicts
        lookahead_state = state_dict.get("lookahead", {})
        for param, param_state in lookahead_state.items():
            self.state[param] = param_state
        self.step_count = state_dict.get("step_count", 0)


# =============================================================================
# Module Metadata
# =============================================================================

__all__ = [
    "OptimizerFactory",
    "GradientManager",
    "AdaptiveOptimizer",
    "OptimizerCheckpoint",
    "LookaheadOptimizer",
    "get_optimizer",
    "clip_gradients",
    "centralize_gradients",
    "inject_gradient_noise",
]

__version__ = "2.0.0"
__author__ = "Zynthé Team"
