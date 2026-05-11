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

from __future__ import annotations


from typing import Dict, List, Any, Optional, Tuple, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import yaml
import json
from datetime import datetime
import warnings
from copy import deepcopy
import time

from .base_distiller import BaseDistiller
from .kd_hinton import KDHintonDistiller
from .feature_distiller import FeatureDistiller
from .similarity_transfer import SimilarityTransfer
from .presets import get_preset, list_presets
from zynthe.evaluation.metrics_extended import DistillationEfficacyIndex, CompressionAwareScore

# Optional imports
try:
    from .attention_transfer import AttentionTransferDistiller

    HAS_ATTENTION = True
except ImportError:
    HAS_ATTENTION = False
    AttentionTransferDistiller = None  # type: ignore

# There is no distiller-shaped QAT class in core.quant. QATRunner is an
# orchestration runner, not a BaseDistiller implementation, so it should not be
# registered here until a real QAT distiller exists.
import logging

HAS_QAT = False
QATDistiller = None  # type: ignore

logger = logging.getLogger(__name__)


class StageController:
    """Controls stage execution, checkpoints, and dependencies."""

    def __init__(self, stages: List[Dict], output_dir: str):
        """Initialize stage controller."""
        self.stages = stages
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_stage = 0
        self.stage_history: List[Dict[str, Any]] = []
        self.checkpoints: Dict[int, Path] = {}

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
        metrics: Dict[str, float],
    ) -> Path:
        """Save stage checkpoint."""
        checkpoint_path = self.output_dir / f"stage_{stage_idx}_checkpoint.pt"

        checkpoint = {
            "stage_idx": stage_idx,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.checkpoints[stage_idx] = checkpoint_path
        logger.info(f" Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, stage_idx: int, model: nn.Module) -> Dict:
        """Load checkpoint from previous stage."""
        if stage_idx not in self.checkpoints:
            raise ValueError(f"No checkpoint found for stage {stage_idx}")

        checkpoint_path = self.checkpoints[stage_idx]
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f" Loaded checkpoint from stage {stage_idx}")
        return checkpoint

    def check_dependencies(self, stage: Dict) -> bool:
        """Check if stage dependencies are satisfied."""
        dependencies = stage.get("depends_on", [])

        for dep_idx in dependencies:
            if dep_idx not in self.checkpoints:
                logger.info(f" Dependency not satisfied: stage {dep_idx} not completed")
                return False

        return True

    def log_stage(self, stage_idx: int, stage_name: str, metrics: Dict):
        """Log stage completion."""
        self.stage_history.append(
            {
                "stage": stage_idx,
                "name": stage_name,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def generate_report(self) -> Dict:
        """Generate training report."""
        return {
            "total_stages": len(self.stages),
            "completed_stages": len(self.stage_history),
            "stage_history": self.stage_history,
            "checkpoints": {k: str(v) for k, v in self.checkpoints.items()},
        }


class DistillerRegistry:
    """Registry for distiller modules with plug-and-play support."""

    def __init__(self):
        """Initialize registry with built-in distillers."""
        self._registry = {
            "kd": KDHintonDistiller,
            "kd_hinton": KDHintonDistiller,
            "feature": FeatureDistiller,
            "similarity": SimilarityTransfer,
            "similarity_transfer": SimilarityTransfer,
            "multi_stage": MultiStageDistiller,  # Full multi-stage orchestrator
        }

        # Add optional distillers
        if HAS_ATTENTION and AttentionTransferDistiller is not None:
            self._registry["attention"] = AttentionTransferDistiller

        if HAS_QAT and QATDistiller is not None:
            self._registry["qat"] = QATDistiller

    def register(self, name: str, distiller_cls: type):
        """Register custom distiller."""
        self._registry[name] = distiller_cls
        logger.info(f" Registered distiller: {name}")

    def get(self, name: str) -> Optional[type]:
        """Get distiller by name."""
        return self._registry.get(name)

    def list_available(self) -> List[str]:
        """List all available distillers."""
        return list(self._registry.keys())


class AdaptiveLossScheduler:
    """Dynamically adjusts loss weights across stages."""

    def __init__(
        self, initial_weights: Optional[Dict[str, float]] = None, schedule_type: str = "linear"
    ):
        """
        Args:
            initial_weights: Initial loss weights (alpha, beta, gamma). If None, uses defaults.
            schedule_type: 'linear', 'cosine', or 'step'
        """
        # Use default weights if none provided
        if initial_weights is None:
            initial_weights = {"alpha": 0.7, "beta": 0.5, "gamma": 0.3}

        self.weights = initial_weights.copy()
        self.initial_weights = initial_weights.copy()
        self.schedule_type = schedule_type
        self.history: List[Dict[str, float]] = []

    def update(self, stage_idx: int, total_stages: int, metrics: Dict[str, float]):
        """Update weights based on progress and metrics."""
        progress = stage_idx / total_stages

        if self.schedule_type == "linear":
            # Linear decay for early distillation
            self.weights["alpha"] = self.initial_weights["alpha"] * (1 - 0.5 * progress)
            self.weights["beta"] = self.initial_weights["beta"] * (1 + 0.5 * progress)

        elif self.schedule_type == "cosine":
            # Cosine annealing
            import math

            self.weights["alpha"] = (
                self.initial_weights["alpha"] * (1 + math.cos(math.pi * progress)) / 2
            )

        elif self.schedule_type == "adaptive":
            # Performance-based adjustment
            if "student_acc" in metrics and "teacher_acc" in metrics:
                gap = metrics["teacher_acc"] - metrics["student_acc"]
                if gap > 0.1:
                    self.weights["alpha"] *= 1.1  # Increase KD weight
                else:
                    self.weights["beta"] *= 1.1  # Focus on features

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

    @staticmethod
    def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries without mutating inputs."""
        result: Dict[str, Any] = deepcopy(dict(base))
        for key, value in overrides.items():
            if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
                result[key] = MultiStageDistiller._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]],
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
        output_dir: str = "experiments/multi_stage",
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

        .. deprecated::
            Use :class:`core.pipelines.multi_stage_pipeline.MultiStagePipeline`
            instead for new distillation workflows.
        """
        warnings.warn(
            "MultiStageDistiller is deprecated. Use MultiStagePipeline from "
            "core.pipelines instead for new distillation workflows.",
            DeprecationWarning,
            stacklevel=2,
        )

        if config is None:
            config = {}
        else:
            config = deepcopy(config)

        preset_name = config.get("preset") or config.get("distillation", {}).get("preset")
        if preset_name:
            try:
                preset_cfg = get_preset(preset_name)
                config = self._deep_merge(preset_cfg, config)
            except KeyError:
                warnings.warn(
                    f"Preset '{preset_name}' not found. Available presets: {list_presets()}"
                )

        if device is None:
            device = config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.teacher_params = int(sum(p.numel() for p in self.teacher.parameters()))
        self.student_params = int(sum(p.numel() for p in self.student.parameters()))
        self._teacher_baseline: Optional[Dict[str, float]] = None

        callbacks_cfg = self.config.get("callbacks", {})
        self.progress_callback = callbacks_cfg.get("on_stage_end")
        self.metrics_callback = callbacks_cfg.get("on_stage_metrics")
        self.plan_metadata = deepcopy(self.config.get("metadata", {}))
        quality_gate_cfg = self.config.get("quality_gate", {})
        self.stop_on_regression = bool(quality_gate_cfg.get("stop_on_regression", False))
        self.max_accuracy_drop = float(quality_gate_cfg.get("max_accuracy_drop", 0.0))
        self.min_stage_accuracy = quality_gate_cfg.get("min_stage_accuracy", None)
        self.run_state: Dict[str, Any] = {
            "stopped_early": False,
            "stop_reason": None,
            "stopped_at_stage": None,
        }

        # Initialize components
        self.registry = DistillerRegistry()
        self.stages = self._parse_stages(config)
        self.controller = StageController(self.stages, output_dir)
        self.loss_scheduler = AdaptiveLossScheduler(
            config.get("distillation", {}).get("loss_schedule")
        )

        # Stage tracking
        self.stage_metrics: List[Dict[str, float]] = []
        self.knowledge_bank: List[List[torch.Tensor]] = []  # type: ignore[type-arg]  # For knowledge replay

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
        distill_cfg = config.get("distillation", {})
        method = distill_cfg.get("method") or distill_cfg.get("type") or "kd_hinton"
        method_aliases = {
            "hinton": "kd_hinton",
            "kdhinton": "kd_hinton",
            "feature_distillation": "feature",
            "similarity_transfer": "similarity",
            "attention_transfer": "attention",
        }
        method = method_aliases.get(str(method).lower(), str(method).lower())
        if method == "multi_stage":
            method = "kd_hinton"

        # Check if multi-stage is enabled
        if not distill_cfg.get("multi_stage", False):
            # Single stage fallback
            return [
                {
                    "name": "Single Stage Distillation",
                    "type": method,
                    "epochs": config.get("train", {}).get(
                        "epochs", config.get("training", {}).get("epochs", 10)
                    ),
                    "config": distill_cfg,
                }
            ]

        # Parse stages from config
        stages = distill_cfg.get("stages", [])

        if not stages:
            # Auto-generate stages from preflight
            stages = self._auto_generate_stages(config)

        normalized_stages: List[Dict[str, Any]] = []
        for idx, stage in enumerate(stages, 1):
            normalized = self._normalize_stage(stage, idx)
            self._validate_stage(normalized, idx)
            normalized_stages.append(normalized)

        return normalized_stages

    def _normalize_stage(self, stage: Dict[str, Any], stage_idx: int) -> Dict[str, Any]:
        """Normalize stage schema and aliases."""
        normalized = deepcopy(stage)

        stage_type = normalized.get("type") or normalized.get("method") or "kd"
        stage_type_aliases = {
            "hinton": "kd_hinton",
            "kdhinton": "kd_hinton",
            "feature_distillation": "feature",
            "similarity_transfer": "similarity",
            "attention_transfer": "attention",
        }
        stage_type = stage_type_aliases.get(str(stage_type).lower(), str(stage_type).lower())
        normalized["type"] = stage_type

        normalized.setdefault("name", f"Stage {stage_idx} - {stage_type}")
        normalized["epochs"] = int(normalized.get("epochs", 1) or 1)
        normalized.setdefault("config", {})

        depends_on = normalized.get("depends_on", [])
        if isinstance(depends_on, int):
            depends_on = [depends_on]
        normalized["depends_on"] = [int(dep) for dep in depends_on if int(dep) > 0]
        return normalized

    def _validate_stage(self, stage: Dict[str, Any], stage_idx: int) -> None:
        """Validate stage config and emit actionable warnings."""
        stage_type = stage.get("type") or ""
        if self.registry.get(stage_type) is None:
            warnings.warn(
                f"Stage {stage_idx} uses unknown distiller type '{stage_type}'. "
                f"Available: {self.registry.list_available()}"
            )

        epochs = int(stage.get("epochs", 1) or 1)
        if epochs <= 0:
            warnings.warn(f"Stage {stage_idx} has non-positive epochs={epochs}. Forcing epochs=1.")
            stage["epochs"] = 1

        if stage_type == "attention":
            config = stage.get("config", {})
            attn_cfg = config.get("attention_transfer", config)
            if not attn_cfg.get("teacher_layers") and not attn_cfg.get("student_layers"):
                attn_cfg.setdefault("auto_detect_layers", True)
                warnings.warn(
                    f"Stage {stage_idx} attention transfer has no explicit layers; enabling auto_detect_layers=true."
                )
                if "attention_transfer" in config:
                    config["attention_transfer"] = attn_cfg

    def _auto_generate_stages(self, config: Dict) -> List[Dict]:
        """
        Auto-generate stages based on preflight analysis.

        Args:
            config: Configuration with preflight results

        Returns:
            Generated stage configurations
        """
        logger.info("[AUTO] Auto-generating stages from preflight analysis...")
        preflight = config.get("preflight", {})
        compression_ratio = preflight.get("compression_ratio", 2.0)
        model_type = preflight.get("model_type", "unknown")

        stages = []

        # Stage 1: Always start with logit alignment
        stages.append(
            {
                "name": "Stage 1 - Logit Alignment",
                "type": "kd",
                "epochs": 3,
                "config": {"temperature": 4.0, "alpha": 0.9},
            }
        )

        # Stage 2: Feature distillation for high compression
        if compression_ratio > 3.0:
            stages.append(
                {
                    "name": "Stage 2 - Feature Refinement",
                    "type": "feature",
                    "epochs": 3,
                    "config": {"beta": 0.6, "feature_stages": [1, 2, 3]},
                }
            )

        # Stage 3: Attention transfer for transformers
        if ("transformer" in model_type.lower() or "bert" in model_type.lower()) and HAS_ATTENTION:
            stages.append(
                {
                    "name": "Stage 3 - Attention Imitation",
                    "type": "attention",
                    "epochs": 2,
                    "config": {"gamma": 0.3},
                }
            )

        # Stage 4: QAT for very high compression
        if compression_ratio > 8.0 and HAS_QAT:
            stages.append(
                {
                    "name": "Stage 4 - QAT Fine-Tuning",
                    "type": "qat",
                    "epochs": 2,
                    "config": {"precision": "int8"},
                }
            )

        logger.info(f"  → Generated {len(stages)} stages")
        for i, stage in enumerate(stages, 1):
            logger.info(f"     {i}. {stage['name']} ({stage['type']})")
        return stages

    def run(self) -> Dict[str, Any]:
        """
        Execute multi-stage distillation.

        Returns:
            Training report with all stages
        """
        logger.info("\n" + "=" * 70)
        logger.info(">>> MULTI-STAGE DISTILLATION")
        logger.info("=" * 70)
        logger.info(f"Total Stages: {len(self.stages)}")
        logger.info(f"Output Dir: {self.output_dir}")
        logger.info("")
        # Run each stage
        for stage_idx, stage_cfg in enumerate(self.stages, 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f">>> STAGE {stage_idx}/{len(self.stages)}: {stage_cfg['name']}")
            logger.info(f"{'=' * 70}")
            # Check dependencies
            if not self.controller.check_dependencies(stage_cfg):
                logger.error("[FAIL] Dependencies not satisfied. Skipping stage.")
                continue

            # Run stage
            stage_metrics = self._run_stage(stage_idx, stage_cfg)

            # Save checkpoint
            self.controller.save_checkpoint(
                stage_idx, self.student, None, stage_metrics  # Can add optimizer if needed
            )

            # Log stage
            self.controller.log_stage(stage_idx, stage_cfg["name"], stage_metrics)
            self.stage_metrics.append(stage_metrics)

            quality_ok, reason = self._check_quality_gate(stage_idx, stage_cfg, stage_metrics)
            if not quality_ok:
                warn_msg = (
                    f"Quality gate failed at stage {stage_idx} ({stage_cfg.get('name')}): {reason}"
                )
                warnings.warn(warn_msg)
                if self.stop_on_regression:
                    self.run_state["stopped_early"] = True
                    self.run_state["stop_reason"] = reason
                    self.run_state["stopped_at_stage"] = stage_idx

            if callable(self.metrics_callback):
                try:
                    self.metrics_callback(stage_idx, stage_cfg, stage_metrics)
                except Exception as callback_error:
                    warnings.warn(f"Metrics callback failed at stage {stage_idx}: {callback_error}")

            if callable(self.progress_callback):
                try:
                    self.progress_callback(stage_idx, stage_cfg, stage_metrics)
                except Exception as callback_error:
                    warnings.warn(
                        f"Progress callback failed at stage {stage_idx}: {callback_error}"
                    )

            # Adaptive weight adjustment
            # Update loss scheduler weights based on stage metrics
            if hasattr(self.loss_scheduler, "update"):
                self.loss_scheduler.update(stage_idx, len(self.stages), stage_metrics)

            logger.info(f"\n[OK] Stage {stage_idx} completed!")
            self._print_stage_summary(stage_idx, stage_metrics)

            if self.run_state.get("stopped_early"):
                logger.info("\n Early stop triggered by quality gate.")
                break

        # Generate final report
        report = self._generate_final_report()

        # Save report
        self._save_report(report)

        logger.info("\n" + "=" * 70)
        logger.info("[DONE] MULTI-STAGE DISTILLATION COMPLETED")
        logger.info("=" * 70)
        self._print_final_summary(report)

        return report

    def _check_quality_gate(
        self, stage_idx: int, stage_cfg: Dict[str, Any], stage_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Return (is_ok, reason) for configured stage quality gates."""
        current_acc = float(stage_metrics.get("val_accuracy", 0.0) or 0.0)

        if self.min_stage_accuracy is not None:
            min_required = float(self.min_stage_accuracy)
            if current_acc < min_required:
                return (
                    False,
                    f"val_accuracy {current_acc:.2f}% < min_stage_accuracy {min_required:.2f}%",
                )

        if stage_idx > 1 and self.max_accuracy_drop > 0 and len(self.stage_metrics) >= 2:
            prev_acc = float(self.stage_metrics[-2].get("val_accuracy", 0.0) or 0.0)
            drop = prev_acc - current_acc
            if drop > self.max_accuracy_drop:
                return (
                    False,
                    f"accuracy drop {drop:.2f}% exceeds threshold {self.max_accuracy_drop:.2f}%",
                )

        return True, "ok"

    def _instantiate_distiller(
        self, distiller_cls: type, distiller_config: Dict[str, Any]
    ) -> BaseDistiller:
        """Instantiate a distiller with graceful fallback for legacy signatures."""
        try:
            return distiller_cls(
                self.teacher,
                self.student,
                config=distiller_config,
                device=torch.device(self.device) if isinstance(self.device, str) else self.device,
            )
        except TypeError:
            return distiller_cls(self.teacher, self.student, distiller_config)

    def _adaptive_lr(self, stage_idx: int) -> float:
        """Compute an adaptive learning rate based on model size and stage.

        Base LR is chosen from the student's trainable parameter count
        using empirically proven scaling (smaller models tolerate higher
        LR).  Each subsequent stage decays by ``stage_decay`` since later
        stages fine-tune an already partially-trained student.

        Returns:
            Computed learning rate (float).
        """
        trainable_params = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )

        # Base LR from model size (matches DistilBERT/TinyBERT/MiniLM ranges)
        if trainable_params < 5_000_000:       # < 5M  (tiny)
            base_lr = 2e-3
        elif trainable_params < 30_000_000:    # < 30M (small, e.g. DistilBERT)
            base_lr = 1e-3
        elif trainable_params < 100_000_000:   # < 100M (medium, e.g. BERT-base)
            base_lr = 5e-4
        elif trainable_params < 500_000_000:   # < 500M (large, e.g. BERT-large)
            base_lr = 2e-4
        else:                                  # 500M+ (XL / LLM)
            base_lr = 5e-5

        # Stage decay: later stages refine with smaller steps
        stage_decay = 0.7
        lr = base_lr * (stage_decay ** (stage_idx - 1))

        logger.info(
            "  Adaptive LR: %.1e (base=%.1e, params=%.1fM, stage_decay=%.1f^%d)",
            lr, base_lr, trainable_params / 1e6, stage_decay, stage_idx - 1,
        )
        return lr

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
        distiller_cls = self.registry.get(stage_cfg["type"])

        if distiller_cls is None:
            warnings.warn(
                f"Distiller type '{stage_cfg['type']}' not found in registry. Skipping stage."
            )
            return {"train_loss": 0.0, "val_loss": 0.0, "val_accuracy": 0.0}

        # Get loss weights (update scheduler first if needed)
        if hasattr(self.loss_scheduler, "update"):
            self.loss_scheduler.update(stage_idx, len(self.stages), {})
        loss_weights = self.loss_scheduler.get_weights()
        logger.info(
            "Loss weights: \u03b1=%.2f, \u03b2=%.2f, \u03b3=%.2f",
            loss_weights.get("alpha", 0),
            loss_weights.get("beta", 0),
            loss_weights.get("gamma", 0),
        )

        # Merge stage config with loss weights without mutating presets
        distiller_config = deepcopy(stage_cfg.get("config", {}))
        for key, value in loss_weights.items():
            distiller_config.setdefault(key, value)
        metadata = distiller_config.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            distiller_config["metadata"] = metadata
        metadata["stage_name"] = stage_cfg.get("name")
        metadata["stage_index"] = stage_idx

        # Initialize distiller
        logger.info(f"Initializing {stage_cfg['type']} distiller...")
        distiller = self._instantiate_distiller(distiller_cls, distiller_config)

        # Initialize optimizer and scheduler
        # Adaptive LR: scale based on student size and stage progression.
        # Users can always override via config "lr" or "learning_rate".
        config_lr = distiller_config.get("lr", distiller_config.get("learning_rate", None))
        if config_lr is not None:
            learning_rate = float(config_lr)
        else:
            learning_rate = self._adaptive_lr(stage_idx)
        weight_decay = distiller_config.get("weight_decay", 0.01)
        optimizer_type = distiller_config.get(
            "optimizer_type", distiller_config.get("optimizer", "adamw")
        )
        scheduler_type = distiller_config.get(
            "scheduler_type", distiller_config.get("scheduler", "cosine")
        )
        distiller.optimizer, distiller.scheduler = distiller._init_optimizers(
            lr=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            total_steps=(
                max(1, len(self.train_loader) * stage_cfg.get("epochs", 3))
                if self.train_loader
                else 1000
            ),
        )

        # Apply layer freezing if specified
        if stage_cfg.get("freeze_layers"):
            self._freeze_layers(stage_cfg["freeze_layers"])

        # Train stage
        epochs = stage_cfg.get("epochs", 3)
        logger.info(f"Training for {epochs} epochs (lr={learning_rate:.1e}, optimizer={optimizer_type})...")
        stage_metrics = {"train_loss": 0.0, "val_loss": 0.0, "val_accuracy": 0.0}

        # Training loop
        if self.train_loader is not None:
            for epoch in range(epochs):
                epoch_loss = self._train_epoch(distiller, epoch + 1, epochs)
                stage_metrics["train_loss"] = epoch_loss

                # Evaluate
                if self.val_loader is not None and (epoch + 1) % 1 == 0:
                    val_metrics = self._evaluate(distiller)
                    stage_metrics.update(val_metrics)

                    logger.info(
                        "  Epoch %d/%d: Loss=%.4f, Val Acc=%.2f%%",
                        epoch + 1,
                        epochs,
                        epoch_loss,
                        val_metrics.get("val_accuracy", 0),
                    )
        else:
            warnings.warn("No train_loader provided, skipping training")

        # Unfreeze layers
        if stage_cfg.get("freeze_layers"):
            self._unfreeze_all()

        # Knowledge replay (store teacher outputs)
        if stage_cfg.get("knowledge_replay", False):
            self._store_knowledge(distiller)

        return stage_metrics

    def _train_epoch(self, distiller: BaseDistiller, epoch: int, total_epochs: int) -> float:
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

        import time

        total_loss = 0.0
        num_batches = 0
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # Log every ~10%
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Perform training step (forward + backward + optimizer)
                loss_dict = distiller.training_step(
                    batch=batch,
                    optimizer=None,  # Use distiller.optimizer
                    grad_clip=1.0,  # Default gradient clipping
                )

                # Accumulate loss
                if isinstance(loss_dict, dict):
                    loss = loss_dict.get("total", 0.0)
                    if isinstance(loss, (float, int)):
                        total_loss += loss
                    else:
                        total_loss += loss.item()  # Should be float, but being safe
                else:
                    # Fallback if return format implies direct loss
                    total_loss += float(loss_dict)  # type: ignore[arg-type]

                num_batches += 1

                # Progress logging
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                    elapsed = time.time() - epoch_start
                    avg_loss = total_loss / max(num_batches, 1)
                    batches_per_sec = num_batches / max(elapsed, 0.001)
                    remaining = (total_batches - batch_idx - 1) / max(batches_per_sec, 0.001)
                    pct = 100 * (batch_idx + 1) / total_batches
                    logger.info(
                        "    [%5.1f%%] batch %d/%d | loss: %.4f | %.1f batch/s | ETA: %.0fs",
                        pct,
                        batch_idx + 1,
                        total_batches,
                        avg_loss,
                        batches_per_sec,
                        remaining,
                    )

            except Exception as e:
                warnings.warn(f"Error in batch {batch_idx}: {e}")
                continue

        return total_loss / max(num_batches, 1)

    def _evaluate(self, distiller: BaseDistiller) -> Dict[str, float]:
        """
        Evaluate current student model (and teacher for reference).

        Args:
            distiller: Distiller instance

        Returns:
            Evaluation metrics
        """
        if self.val_loader is None:
            warnings.warn("val_loader is None, returning empty metrics")
            return {"val_loss": 0.0, "val_accuracy": 0.0}

        self.student.eval()
        self.teacher.eval()

        total_loss = 0.0
        student_correct = 0
        teacher_correct = 0
        total = 0
        eval_errors = 0
        student_latencies_ms: List[float] = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    labels = batch.get("labels", None)
                    if labels is not None:
                        labels = labels.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    labels = None

                # Forward pass
                try:
                    t0 = time.perf_counter()
                    # Use _safe_forward to handle dtype mismatches (float16 models)
                    student_out = distiller._safe_forward(
                        self.student, inputs, {}
                    )
                    student_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                    # Extract logits
                    student_logits = distiller._extract_logits_tensor(student_out)

                    # Teacher forward for reference accuracy
                    teacher_out = distiller._safe_forward(
                        self.teacher, inputs, {}
                    )
                    teacher_logits = distiller._extract_logits_tensor(teacher_out)

                    # Compute accuracy
                    if labels is not None and hasattr(student_logits, "dim") and student_logits.dim() >= 2:
                        try:
                            total_loss += float(F.cross_entropy(student_logits, labels).item())
                        except Exception:
                            logger.debug("Cross-entropy loss computation failed during eval")
                        _, s_pred = student_logits.max(1)
                        _, t_pred = teacher_logits.max(1)
                        total += labels.size(0)
                        student_correct += s_pred.eq(labels).sum().item()
                        teacher_correct += t_pred.eq(labels).sum().item()

                except Exception as e:
                    eval_errors += 1
                    if eval_errors <= 3:
                        warnings.warn(f"Error in evaluation batch: {e}")
                    continue

        student_acc = 100.0 * student_correct / max(total, 1)
        teacher_acc = 100.0 * teacher_correct / max(total, 1)
        latency_ms = float(sum(student_latencies_ms) / max(len(student_latencies_ms), 1))

        if eval_errors > 0:
            logger.warning("  [EVAL] %d/%d batches failed during evaluation!", eval_errors, eval_errors + len(student_latencies_ms))

        logger.info(
            "  [EVAL] Student=%.2f%% | Teacher=%.2f%% | total_samples=%d | eval_errors=%d",
            student_acc, teacher_acc, total, eval_errors,
        )

        return {
            "val_loss": total_loss / max(len(self.val_loader), 1),
            "val_accuracy": student_acc,
            "teacher_accuracy": teacher_acc,
            "student_latency_ms": latency_ms,
        }

    def _freeze_layers(self, layer_spec: List[int]):
        """
        Freeze specified layers.

        Args:
            layer_spec: List of layer indices to freeze
        """
        logger.info(f"  [LOCK] Freezing layers: {layer_spec}")
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

        logger.info("  [SAVE] Storing knowledge for replay...")
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
        logger.info(f"   Stored {len(knowledge_samples)} knowledge samples")

    def _print_stage_summary(self, stage_idx: int, metrics: Dict):
        """Print stage summary."""
        logger.info("\n[INFO] Stage Summary:")
        logger.info(f"  Train Loss: {metrics.get('train_loss', 0):.4f}")
        logger.info(f"  Val Loss: {metrics.get('val_loss', 0):.4f}")
        logger.info(f"  Val Accuracy: {metrics.get('val_accuracy', 0):.2f}%")

    def _compute_teacher_baseline(self) -> Dict[str, float]:
        """Compute and cache teacher baseline metrics on validation split."""
        if self._teacher_baseline is not None:
            return self._teacher_baseline

        baseline = {
            "teacher_accuracy_pct": 0.0,
            "teacher_latency_ms": 1.0,
        }

        if self.val_loader is None:
            self._teacher_baseline = baseline
            return baseline

        self.teacher.eval()
        correct = 0
        total = 0
        latencies_ms: List[float] = []

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    labels = batch.get("labels", None)
                    if labels is not None:
                        labels = labels.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    labels = None

                try:
                    t0 = time.perf_counter()
                    if isinstance(inputs, dict):
                        teacher_out = self.teacher(**inputs)
                    else:
                        teacher_out = self.teacher(inputs)
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                    if isinstance(teacher_out, dict):
                        logits = teacher_out["logits"]
                    elif hasattr(teacher_out, "logits"):
                        logits = teacher_out.logits
                    elif isinstance(teacher_out, tuple):
                        logits = teacher_out[0]
                    else:
                        logits = teacher_out

                    if labels is not None and hasattr(logits, "dim") and logits.dim() >= 2:
                        _, predicted = logits.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                except Exception:
                    continue

        baseline["teacher_accuracy_pct"] = float(100.0 * correct / max(total, 1))
        baseline["teacher_latency_ms"] = float(sum(latencies_ms) / max(len(latencies_ms), 1))
        self._teacher_baseline = baseline
        return baseline

    def _build_stage_comparison(self) -> List[Dict[str, Any]]:
        """Build stage-wise trend artifact including DEI/CAS and accuracy deltas."""
        baseline = self._compute_teacher_baseline()
        teacher_acc_pct = float(baseline.get("teacher_accuracy_pct", 0.0))
        teacher_latency_ms = max(float(baseline.get("teacher_latency_ms", 1.0)), 1e-6)

        comparison: List[Dict[str, Any]] = []
        prev_acc = None
        prev_dei = None
        prev_cas = None

        for idx, (stage_cfg, metrics) in enumerate(zip(self.stages, self.stage_metrics), 1):
            student_acc_pct = float(metrics.get("val_accuracy", 0.0) or 0.0)
            student_latency_ms = max(
                float(metrics.get("student_latency_ms", teacher_latency_ms) or teacher_latency_ms),
                1e-6,
            )

            dei_payload = DistillationEfficacyIndex.compute_dei(
                teacher_acc=max(teacher_acc_pct / 100.0, 1e-8),
                student_acc=max(student_acc_pct / 100.0, 0.0),
                teacher_params=max(self.teacher_params, 1),
                student_params=max(self.student_params, 1),
            )
            cas_payload = CompressionAwareScore.compute_cas(
                accuracy=max(student_acc_pct / 100.0, 0.0),
                teacher_params=max(self.teacher_params, 1),
                student_params=max(self.student_params, 1),
                teacher_latency=teacher_latency_ms,
                student_latency=student_latency_ms,
            )

            rec: Dict[str, Any] = {
                "stage": idx,
                "name": stage_cfg.get("name", f"Stage {idx}"),
                "type": stage_cfg.get("type", "unknown"),
                "val_accuracy_pct": student_acc_pct,
                "student_latency_ms": student_latency_ms,
                "teacher_accuracy_pct": teacher_acc_pct,
                "teacher_latency_ms": teacher_latency_ms,
                "dei": float(dei_payload.get("dei", 0.0)),
                "cas": float(cas_payload.get("cas", 0.0)),
                "compression_ratio": float(dei_payload.get("compression_ratio", 0.0)),
            }

            rec["accuracy_delta_pct"] = None if prev_acc is None else (student_acc_pct - prev_acc)
            rec["dei_delta"] = None if prev_dei is None else (rec["dei"] - prev_dei)
            rec["cas_delta"] = None if prev_cas is None else (rec["cas"] - prev_cas)

            comparison.append(rec)
            prev_acc = student_acc_pct
            prev_dei = rec["dei"]
            prev_cas = rec["cas"]

        return comparison

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        # Calculate aggregate metrics
        total_gain = 0.0
        for i, metrics in enumerate(self.stage_metrics):
            if i > 0:
                prev_acc = self.stage_metrics[i - 1].get("val_accuracy", 0)
                curr_acc = metrics.get("val_accuracy", 0)
                gain = curr_acc - prev_acc
                total_gain += gain

        # Get preflight info
        preflight = self.config.get("preflight", {})

        report = {
            "summary": {
                "total_stages": len(self.stages),
                "model_type": preflight.get("model_type", "unknown"),
                "compression_ratio": preflight.get("compression_ratio", 0),
                "total_accuracy_gain": 0.0,  # Will be updated below
            },
            "preflight": {
                "model_type": preflight.get("model_type", "unknown"),
                "compression_ratio": preflight.get("compression_ratio", 0),
                "stages_completed": [],
            },
            "stages": [],
            "final_metrics": {},
            "stage_controller_report": self.controller.generate_report(),
            "metadata": self.plan_metadata,
            "run_state": deepcopy(self.run_state),
        }

        if self.plan_metadata and "preset" in self.plan_metadata:
            report["summary"]["preset"] = self.plan_metadata["preset"]

        # Add stage details
        for i, (stage_cfg, metrics) in enumerate(zip(self.stages, self.stage_metrics)):
            stage_info = {
                "stage": i + 1,
                "name": stage_cfg["name"],
                "type": stage_cfg["type"],
                "epochs": stage_cfg.get("epochs", 0),
                "metrics": metrics,
            }

            report["stages"].append(stage_info)

            # Add to preflight summary
            acc_gain = 0.0
            if i > 0:
                prev_acc = self.stage_metrics[i - 1].get("val_accuracy", 0)
                curr_acc = metrics.get("val_accuracy", 0)
                acc_gain = curr_acc - prev_acc

            report["preflight"]["stages_completed"].append(
                {"name": stage_cfg["name"], "accuracy_gain": acc_gain}
            )

        # Final metrics
        if self.stage_metrics:
            final = self.stage_metrics[-1]
            report["final_metrics"] = {
                "final_accuracy": final.get("val_accuracy", 0),
                "total_accuracy_gain": total_gain,
                "final_loss": final.get("val_loss", 0),
            }
            # Update summary as well
            report["summary"]["total_accuracy_gain"] = total_gain

        stage_comparison = self._build_stage_comparison()
        report["stage_comparison"] = stage_comparison
        if stage_comparison:
            report["summary"]["best_dei"] = max(row.get("dei", 0.0) for row in stage_comparison)
            report["summary"]["best_cas"] = max(row.get("cas", 0.0) for row in stage_comparison)
            report["summary"]["best_stage_by_dei"] = max(
                stage_comparison, key=lambda row: row.get("dei", 0.0)
            ).get("name")
            report["summary"]["best_stage_by_cas"] = max(
                stage_comparison, key=lambda row: row.get("cas", 0.0)
            ).get("name")

        return report

    def _print_final_summary(self, report: Dict):
        """Print final summary."""
        logger.info("\n[INFO] FINAL SUMMARY")
        logger.info("-" * 70)
        preflight = report["preflight"]
        logger.info(f"Model Type: {preflight['model_type']}")
        logger.info(f"Compression Ratio: {preflight['compression_ratio']:.1f}x")
        logger.info(f"Stages Completed: {len(preflight['stages_completed'])}")
        logger.info("\n[INFO] Stage-wise Progress:")
        for stage_info in preflight["stages_completed"]:
            gain = stage_info["accuracy_gain"]
            sign = "+" if gain >= 0 else ""
            logger.info(f"  {stage_info['name']}: {sign}{gain:.2f}% accuracy gain")
        if report.get("final_metrics"):
            final = report["final_metrics"]
            logger.info("\n\ud83c\udfaf Final Results:")
            logger.info(f"  Final Accuracy: {final['final_accuracy']:.2f}%")
            logger.info(f"  Total Gain: {final['total_accuracy_gain']:.2f}%")

    def _save_report(self, report: Dict):
        """Save report to file."""
        # Save JSON
        json_path = self.output_dir / "multi_stage_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n[SAVE] Report saved: {json_path}")
        # Save YAML
        yaml_path = self.output_dir / "multi_stage_report.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)
        logger.info(f"[SAVE] Report saved: {yaml_path}")
        comparison_rows = report.get("stage_comparison", [])
        if comparison_rows:
            comparison_json = self.output_dir / "stage_comparison.json"
            with open(comparison_json, "w") as f:
                json.dump(comparison_rows, f, indent=2, default=str)
            logger.info(f"[SAVE] Stage comparison saved: {comparison_json}")
            comparison_csv = self.output_dir / "stage_comparison.csv"
            fieldnames = [
                "stage",
                "name",
                "type",
                "val_accuracy_pct",
                "accuracy_delta_pct",
                "student_latency_ms",
                "teacher_accuracy_pct",
                "teacher_latency_ms",
                "compression_ratio",
                "dei",
                "dei_delta",
                "cas",
                "cas_delta",
            ]
            with open(comparison_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in comparison_rows:
                    writer.writerow({key: row.get(key) for key in fieldnames})
            logger.info(f"[SAVE] Stage comparison saved: {comparison_csv}")


# Convenience function for backward compatibility
def run_multi_stage_distillation(
    teacher: nn.Module,
    student: nn.Module,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    output_dir: str = "experiments/multi_stage",
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
        output_dir=output_dir,
    )

    return distiller.run()
