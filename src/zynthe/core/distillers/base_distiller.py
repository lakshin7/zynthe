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

from __future__ import annotations

import copy
import logging
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

import torch.optim as optim
from torch.optim import Optimizer, AdamW, SGD
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, ReduceLROnPlateau

from zynthe.core.utils.device_utils import (
    auto_detect_device as _shared_auto_detect_device,
    move_to_device as _shared_move_to_device,
    normalize_model_output,
)


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
     Teacher Model (frozen)
     Student Model (trainable)
     Feature Extractors (hooks)
     Loss Composer (criterion manager)
     Optimizer & Scheduler
     Metrics Logger
    ```
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        freeze_teacher: bool = True,
        **kwargs,
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
        self.scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None
        self.scheduler_step_per_batch: bool = True
        self.scheduler_requires_metric: bool = False

        # Metrics tracking
        self.metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self.current_epoch: int = 0
        self.global_step: int = 0

        # Visualization hooks (optional)
        self.visualize_features: bool = self.config.get("visualize_features", False)
        self.visualization_callback: Optional[Callable] = None

    # ============================================================================
    # CORE SETUP METHODS
    # ============================================================================

    @property
    def modality_type(self) -> str:
        """Return the modality this distiller operates on.

        Subclasses override to return ``"vision"``, ``"multimodal"``,
        ``"vlm"``, etc.  Used by adapters and pipelines to select the
        correct I/O normalisation strategy.
        """
        return "text"

    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        return _shared_auto_detect_device()

    @staticmethod
    def _get_model_dtype(model: nn.Module) -> torch.dtype:
        """Detect the parameter dtype of a model (float16, bfloat16, float32, etc.)."""
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _cast_inputs_to_model_dtype(self, data: Any, model: nn.Module) -> Any:
        """Cast floating-point tensors in *data* to match *model*'s parameter dtype.

        Integer tensors (e.g. ``input_ids``, ``attention_mask``, ``labels``)
        are left unchanged. This prevents the ``mat1 and mat2 must have the
        same dtype`` crash when a model is loaded in float16/bfloat16 but
        receives float32 inputs.
        """
        target_dtype = self._get_model_dtype(model)
        if target_dtype == torch.float32:
            return data  # No casting needed, everything defaults to float32
        return self._recursive_cast(data, target_dtype)

    @staticmethod
    def _recursive_cast(data: Any, dtype: torch.dtype) -> Any:
        """Recursively cast floating-point tensors to *dtype*."""
        if isinstance(data, torch.Tensor):
            if data.is_floating_point():
                return data.to(dtype)
            return data  # Keep integer tensors (input_ids, labels, etc.) as-is
        if isinstance(data, dict):
            return {k: BaseDistiller._recursive_cast(v, dtype) for k, v in data.items()}
        if isinstance(data, list):
            return [BaseDistiller._recursive_cast(v, dtype) for v in data]
        if isinstance(data, tuple):
            return tuple(BaseDistiller._recursive_cast(v, dtype) for v in data)
        return data

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
        return _shared_move_to_device(data, self.device)

    @staticmethod
    def normalize_outputs(raw_output: Any) -> dict:
        """Normalize HuggingFace model outputs into a standard dict.

        Returns dict with keys: ``logits``, ``hidden_states``,
        ``attentions``, ``loss``.
        """
        return normalize_model_output(raw_output)

    def _get_teacher_hook(self, name: str) -> Callable:
        """Create a forward hook for teacher features."""

        def hook(module, input, output):
            # Detach to prevent gradients flowing to teacher
            self.teacher_hooks[name] = (
                output.detach() if isinstance(output, torch.Tensor) else output
            )

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
        loss_configs = self.config.get("losses", [])

        if not loss_configs:
            # Default: supervised cross-entropy only
            self.losses["supervised"] = nn.CrossEntropyLoss()
            self.loss_weights["supervised"] = 1.0
        else:
            for loss_cfg in loss_configs:
                loss_type = loss_cfg.get("type", "supervised")
                weight = loss_cfg.get("weight", 1.0)

                if loss_type == "supervised" or loss_type == "ce":
                    self.losses["supervised"] = nn.CrossEntropyLoss()
                    self.loss_weights["supervised"] = weight

                # Subclasses add their specific losses here

    def _resolve_task_type(self, logits: Optional[torch.Tensor] = None) -> str:
        """Infer task type from config with logits-shape fallback."""
        distill_cfg = self.config.get("distillation", {}) if isinstance(self.config, dict) else {}
        task_type = None
        if isinstance(distill_cfg, dict):
            task_type = distill_cfg.get("task_type")
        if not task_type and isinstance(self.config, dict):
            task_type = self.config.get("task_type")

        if isinstance(task_type, str) and task_type.strip():
            return task_type.strip().lower()

        if isinstance(logits, torch.Tensor) and logits.dim() == 3:
            return "causal_lm"
        return "classification"

    @staticmethod
    def _extract_logits_tensor(outputs: Any) -> torch.Tensor:
        """Extract logits tensor from common output formats.

        Always returns float32 logits for numerically stable loss
        computation (softmax, KL-div, cross-entropy).  Models loaded in
        float16/bfloat16 produce half-precision logits that can overflow
        (float16 max ≈ 65504) producing ``inf`` values.  ``softmax(inf)``
        yields ``NaN``, so we clamp after upcasting.
        """
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        if isinstance(logits, torch.Tensor):
            # Upcast to float32 for stable loss computation
            if logits.dtype != torch.float32:
                logits = logits.float()
            # Clamp inf/nan from float16 overflow — keeps softmax/KL-div stable
            if logits.is_floating_point() and (
                torch.isinf(logits).any() or torch.isnan(logits).any()
            ):
                logits = torch.clamp(logits, min=-1e4, max=1e4)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits

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
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        **kwargs,
    ) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Initialize optimizer and learning rate scheduler.

        Args:
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            optimizer_type: "adamw", "sgd", or "adam"
            scheduler_type: "cosine", "step", "plateau", or "none"
            **kwargs: Additional optimizer/scheduler arguments

        Returns:
            Tuple of (optimizer, scheduler)
        """
        try:
            lr = float(lr)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid optimizer lr value: {lr!r}") from exc

        try:
            weight_decay = float(weight_decay)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid optimizer weight_decay value: {weight_decay!r}") from exc

        # Get all trainable parameters (student + distiller adapters/regressors)
        params = [p for p in self.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in params)
        student_params = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )
        extra_params = max(0, total_params - student_params)
        if extra_params > 0:
            logger.info(
                "Optimizer params: total=%.2fM (student=%.2fM, extra=%.2fM)",
                total_params / 1e6,
                student_params / 1e6,
                extra_params / 1e6,
            )
        else:
            logger.info("Optimizer params: total=%.2fM (student only)", total_params / 1e6)

        # Extract scheduler-specific kwargs
        total_steps = kwargs.pop("total_steps", 1000)
        step_size = kwargs.pop("step_size", 100)
        gamma = kwargs.pop("gamma", 0.1)
        plateau_mode = kwargs.pop("plateau_mode", "min")
        plateau_factor = kwargs.pop("plateau_factor", 0.5)
        plateau_patience = kwargs.pop("plateau_patience", 1)
        plateau_threshold = kwargs.pop("plateau_threshold", 1e-4)
        plateau_cooldown = kwargs.pop("plateau_cooldown", 0)
        plateau_min_lr = kwargs.pop("plateau_min_lr", 0.0)

        # Create optimizer (remaining kwargs go to optimizer)
        if optimizer_type.lower() == "adamw":
            optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer_type.lower() == "sgd":
            optimizer = SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs)
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Create scheduler
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None
        scheduler_type_norm = str(scheduler_type or "none").lower()
        self.scheduler_step_per_batch = True
        self.scheduler_requires_metric = False
        if scheduler_type_norm == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)  # type: ignore[assignment]
        elif scheduler_type_norm == "step":
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # type: ignore[assignment]
        elif scheduler_type_norm in {"plateau", "reduceonplateau", "reduce_on_plateau"}:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=plateau_mode,
                factor=plateau_factor,
                patience=plateau_patience,
                threshold=plateau_threshold,
                cooldown=plateau_cooldown,
                min_lr=plateau_min_lr,
            )
            self.scheduler_step_per_batch = False
            self.scheduler_requires_metric = True

        return optimizer, scheduler

    # ============================================================================
    # FORWARD PASS
    # ============================================================================

    def forward(
        self, inputs: Any, return_features: bool = False, **kwargs
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
            teacher_outputs = self._safe_forward(
                self.teacher, inputs, kwargs
            )

        # Student forward (with gradients)
        student_outputs = self._safe_forward(
            self.student, inputs, kwargs
        )

        if return_features:
            teacher_features = self._collect_features(self.teacher_hooks)
            student_features = self._collect_features(self.student_hooks)
            return student_outputs, teacher_outputs, teacher_features, student_features

        return student_outputs, teacher_outputs

    def _safe_forward(
        self, model: nn.Module, inputs: Any, kwargs: dict
    ) -> Any:
        """Forward pass with automatic dtype retry on mismatch.

        Most models (especially HuggingFace transformers) handle float32
        inputs even when loaded in float16 — their embedding layers
        convert internally.  We only cast inputs when the forward pass
        actually fails with a ``RuntimeError`` dtype mismatch.
        """
        try:
            if isinstance(inputs, dict):
                return model(**inputs, **kwargs)
            return model(inputs, **kwargs)
        except RuntimeError as e:
            if "dtype" not in str(e).lower():
                raise
            # Retry with inputs cast to the model's parameter dtype
            cast_inputs = self._cast_inputs_to_model_dtype(inputs, model)
            cast_kwargs = self._cast_inputs_to_model_dtype(kwargs, model)
            if isinstance(cast_inputs, dict):
                return model(**cast_inputs, **cast_kwargs)
            return model(cast_inputs, **cast_kwargs)

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
        **kwargs,
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
        grad_clip: Optional[Union[float, Dict[str, Any], str]] = None,
        return_outputs: bool = False,
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
            inputs = {k: self._move_to_device(v) for k, v in batch.items() if k != "labels"}
            targets = batch.get("labels", None)
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
            teacher_features=teacher_features,
        )

        # Backward pass
        opt = optimizer or self.optimizer
        if opt is None:
            raise ValueError("No optimizer provided. Either pass optimizer or set self.optimizer")

        opt.zero_grad()

        # Diagnostic: verify loss has gradient graph
        if not total_loss.requires_grad:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "total_loss does NOT require grad (grad_fn=%s). "
                "Backward will be a no-op -- student weights won't update!",
                total_loss.grad_fn,
            )

        total_loss.backward()

        # Gradient clipping
        pre_clip_norm = None
        agc_stats = None
        if grad_clip is not None:
            if isinstance(grad_clip, dict):
                clip_mode = str(grad_clip.get("type") or grad_clip.get("mode") or "norm").lower()
                if clip_mode == "agc":
                    agc_stats = self._apply_agc(
                        self.student.parameters(),
                        clip_factor=float(grad_clip.get("clip_factor", grad_clip.get("factor", 0.01))),
                        eps=float(grad_clip.get("eps", 1e-3)),
                        norm_type=int(grad_clip.get("norm_type", 2)),
                        exclude_bias_and_norm=bool(
                            grad_clip.get("exclude_bias_and_norm", True)
                        ),
                    )
                else:
                    max_norm = grad_clip.get(
                        "max_norm", grad_clip.get("value", grad_clip.get("norm", None))
                    )
                    if max_norm is not None:
                        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                            self.student.parameters(), float(max_norm)
                        )
            elif isinstance(grad_clip, str) and grad_clip.lower() == "agc":
                agc_stats = self._apply_agc(self.student.parameters())
            else:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), float(grad_clip)
                )

        # Diagnostic: log gradient norms periodically
        if self.global_step % 200 == 0:
            import logging as _lg
            _diag_logger = _lg.getLogger(__name__)
            grad_norms = [
                p.grad.data.norm(2).item()
                for p in self.student.parameters()
                if p.grad is not None
            ]
            total_norm = sum(g ** 2 for g in grad_norms) ** 0.5 if grad_norms else 0.0
            params_with_grad = len(grad_norms)
            total_params = sum(1 for p in self.student.parameters() if p.requires_grad)
            if agc_stats is not None:
                _diag_logger.info(
                    "  [GRAD] step=%d | grad_norm=%.6f | agc_clipped=%d/%d | agc_max_ratio=%.3f | params_with_grad=%d/%d | loss=%.6f (requires_grad=%s)",
                    self.global_step, total_norm, agc_stats["clipped"], agc_stats["total"],
                    agc_stats["max_ratio"], params_with_grad, total_params,
                    total_loss.item(), total_loss.requires_grad,
                )
            elif pre_clip_norm is not None:
                _diag_logger.info(
                    "  [GRAD] step=%d | grad_norm=%.6f | pre_clip=%.6f | clip=%.3f | params_with_grad=%d/%d | loss=%.6f (requires_grad=%s)",
                    self.global_step, total_norm, float(pre_clip_norm), float(grad_clip),
                    params_with_grad, total_params, total_loss.item(), total_loss.requires_grad,
                )
            else:
                _diag_logger.info(
                    "  [GRAD] step=%d | grad_norm=%.6f | params_with_grad=%d/%d | loss=%.6f (requires_grad=%s)",
                    self.global_step, total_norm, params_with_grad, total_params,
                    total_loss.item(), total_loss.requires_grad,
                )

        opt.step()

        # Update scheduler if available
        self.step_scheduler()

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
        self, batch: Union[Tuple, Dict], return_outputs: bool = False
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
            inputs = {k: self._move_to_device(v) for k, v in batch.items() if k != "labels"}
            targets = batch.get("labels", None)
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
            teacher_features=teacher_features,
        )

        if return_outputs:
            return loss_dict, student_outputs
        return loss_dict

    def _apply_agc(
        self,
        params: Iterable[nn.Parameter],
        *,
        clip_factor: float = 0.01,
        eps: float = 1e-3,
        norm_type: int = 2,
        exclude_bias_and_norm: bool = True,
    ) -> Dict[str, float]:
        """Apply adaptive gradient clipping (AGC) to parameters in-place."""
        clipped = 0
        total = 0
        max_ratio = 0.0

        for param in params:
            if param.grad is None:
                continue
            if exclude_bias_and_norm and param.ndim <= 1:
                continue

            grad = param.grad
            param_norm = torch.norm(param.detach(), norm_type)
            grad_norm = torch.norm(grad.detach(), norm_type)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                continue

            max_norm = (param_norm + eps) * clip_factor
            if grad_norm > max_norm:
                scale = max_norm / (grad_norm + 1e-6)
                grad.mul_(scale)
                clipped += 1
                ratio = float(grad_norm / (max_norm + 1e-12))
                if ratio > max_ratio:
                    max_ratio = ratio
            total += 1

        return {"clipped": clipped, "total": total, "max_ratio": max_ratio}

    def step_scheduler(self, metric: Optional[float] = None) -> None:
        """Step the scheduler, honoring per-batch vs per-epoch modes."""
        if self.scheduler is None:
            return
        if getattr(self, "scheduler_step_per_batch", True):
            self.scheduler.step()
            return
        if metric is None:
            return
        try:
            self.scheduler.step(metric)
        except TypeError:
            self.scheduler.step()

    # ============================================================================
    # HOOKS (OPTIONAL CUSTOMIZATION POINTS)
    # ============================================================================

    def pre_forward_hook(self, inputs: Any, **kwargs) -> None:
        """Hook executed before forward pass. Override for custom preprocessing."""

    def post_forward_hook(
        self, inputs: Any, student_outputs: Any, teacher_outputs: Any, **kwargs
    ) -> None:
        """Hook executed after forward pass. Override for custom postprocessing."""

    # ============================================================================
    # LOGGING & METRICS
    # ============================================================================

    def log_metrics(self, metrics: Dict[str, float], phase: str = "train") -> None:
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
        except Exception:
            logger.debug("Hook cleanup failed during distiller destruction")
