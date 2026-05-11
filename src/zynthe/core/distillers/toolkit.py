"""High-level orchestration helpers for knowledge distillation.

The :class:`DistillationToolkit` provides an opinionated, user-friendly facade
around the lower-level distiller modules. It lets product teams assemble
multi-stage distillation plans by selecting a preset or specifying a high-level
business goal (e.g. ``"compression"`` or ``"vision"``) without touching the
underlying research code.

The :class:`Distiller` is a beginner-friendly wrapper that reduces the API
surface to a single ``fit()`` call.
"""

from __future__ import annotations

import enum
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn

from .multi_stage_distiller import MultiStageDistiller
from .presets import get_preset, list_presets, describe_preset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Goal enum (backward-compatible with bare strings)
# ---------------------------------------------------------------------------


class Goal(str, enum.Enum):
    """Distillation goal specifier.

    Inherits from ``str`` so that ``Goal.VISION == "vision"`` is ``True``
    and existing code using bare strings continues to work.

    Example::

        from zynthe import Goal
        toolkit.build_plan(goal=Goal.VISION)
        toolkit.build_plan(goal="vision")  # equivalent
    """

    # Text / NLP
    TEXT = "text"
    NLP = "nlp"
    CODE = "code"
    BERT = "bert"

    # Quick
    SPEED = "speed"
    QUICK = "quick"
    BASELINE = "baseline"

    # Balanced
    BALANCED = "balanced"
    DEFAULT = "default"

    # Full suites
    ALL = "all"
    ALL_DISTILLERS = "all_distillers"
    FULL = "full"
    COMPLETE = "complete"
    SMOKE = "smoke"
    CLASSIFICATION_SMOKE = "classification_smoke"

    # Causal LM / GPT
    GPT = "gpt"
    CAUSAL_LM = "causal_lm"
    GPT_SMOKE = "gpt_smoke"

    # Vision
    TRANSFORMER = "transformer"
    VISION = "vision"
    INTERPRETABILITY = "interpretability"

    # Compression
    COMPRESSION = "compression"
    AGGRESSIVE = "aggressive"

    # Multimodal
    MULTIMODAL = "multimodal"
    CLIP = "clip"
    VLM = "vlm"
    VISION_LANGUAGE = "vision_language"


# ---------------------------------------------------------------------------
# DistillationConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class DistillationConfig:
    """Central configuration for distillation hyperparameters.

    Users can pass an instance to :class:`DistillationToolkit` or
    :class:`Distiller` to override preset defaults without manually
    constructing override dicts.

    Example::

        from zynthe import Distiller, DistillationConfig

        cfg = DistillationConfig(temperature=6.0, alpha=0.9, epochs=10)
        distiller = Distiller(teacher, student, goal="vision", config=cfg)
        distiller.fit(train_loader, val_loader)

    Attributes:
        temperature: Softmax temperature for KD loss.
        alpha: Weight of the distillation loss vs. task loss.
        grad_clip: Maximum gradient norm for clipping.
        learning_rate: Base learning rate for the student optimizer.
        weight_decay: L2 regularization strength.
        epochs: Number of training epochs (overrides preset).
        batch_size: Training batch size (informational, DataLoader is external).
        mixed_precision: Enable AMP mixed-precision training.
        optimizer: Optimizer type (``'adamw'`` or ``'sgd'``).
        scheduler: LR scheduler type (``'cosine'`` or ``'step'``).
    """

    temperature: float = 4.0
    alpha: float = 0.85
    grad_clip: float = 1.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    mixed_precision: bool = False
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    def to_overrides(self) -> Dict[str, Any]:
        """Convert non-default values into a plan overrides dict."""
        overrides: Dict[str, Any] = {}
        if self.epochs is not None:
            overrides.setdefault("training", {})["epochs"] = self.epochs
            overrides.setdefault("train", {})["epochs"] = self.epochs
        if self.batch_size is not None:
            overrides.setdefault("train", {})["batch_size"] = self.batch_size
        if self.mixed_precision:
            overrides.setdefault("train", {})["mixed_precision"] = True
        return overrides


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_GOALS = frozenset(g.value for g in Goal)


def _validate_model(model: Any, name: str) -> None:
    """Raise TypeError if *model* is not a torch.nn.Module."""
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{name} must be a torch.nn.Module, got {type(model).__name__}. "
            f"Pass a PyTorch model (e.g. from transformers or torchvision)."
        )


def _validate_goal(goal: str) -> None:
    """Raise ValueError if *goal* is not a recognized goal string."""
    if goal.lower() not in _VALID_GOALS and goal.lower() not in DistillationToolkit.GOAL_TO_PRESET:
        valid = sorted(_VALID_GOALS)
        raise ValueError(
            f"Unknown distillation goal: {goal!r}. " f"Valid goals are: {', '.join(valid)}"
        )


# ---------------------------------------------------------------------------
# DistillationToolkit
# ---------------------------------------------------------------------------


class DistillationToolkit:
    """Full-featured orchestration layer for knowledge distillation.

    Provides plan building, execution, evaluation, and comparison in a
    single object.  For a simpler API, see :class:`Distiller`.

    Args:
        teacher: Pre-trained teacher model (``torch.nn.Module``).
        student: Student model to be distilled (``torch.nn.Module``).
        device: Device string (``'cuda'``, ``'cpu'``, ``'mps'``).
            Auto-detected if ``None``.
        default_preset: Fallback preset when no goal/preset is given.
        config: Optional :class:`DistillationConfig` with default overrides.

    Raises:
        TypeError: If teacher or student is not a ``torch.nn.Module``.

    Example::

        toolkit = DistillationToolkit(teacher, student)
        plan = toolkit.build_plan(goal="compression")
        report = toolkit.run(plan, train_loader, val_loader)
    """

    GOAL_TO_PRESET = {
        "speed": "quick_start",
        "quick": "quick_start",
        "baseline": "quick_start",
        "balanced": "balanced",
        "default": "balanced",
        "text": "balanced",
        "nlp": "balanced",
        "code": "balanced",
        "bert": "balanced",
        "all": "all_distillers_t4",
        "all_distillers": "all_distillers_t4",
        "full": "all_distillers_t4",
        "complete": "all_distillers_t4",
        "smoke": "all_distillers_classification_smoke",
        "classification_smoke": "all_distillers_classification_smoke",
        "gpt": "all_distillers_causal_lm_smoke",
        "causal_lm": "all_distillers_causal_lm_smoke",
        "gpt_smoke": "all_distillers_causal_lm_smoke",
        "transformer": "vision_transformer",
        "vision": "vision_transformer",
        "interpretability": "vision_transformer",
        "compression": "compression_max",
        "aggressive": "compression_max",
        "multimodal": "multimodal",
        "clip": "multimodal",
        "vlm": "multimodal",
        "vision_language": "multimodal",
    }

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        *,
        device: Optional[str] = None,
        default_preset: str = "balanced",
        config: Optional[DistillationConfig] = None,
    ) -> None:
        _validate_model(teacher, "teacher")
        _validate_model(student, "student")

        self.teacher = teacher
        self.student = student
        self.device = device
        self.default_preset = default_preset
        self.config = config

        # Auto-move models to same device
        resolved = self._resolve_device()
        self._ensure_same_device(resolved)

        logger.debug(
            "DistillationToolkit initialized (device=%s, preset=%s)",
            resolved,
            default_preset,
        )

    def _ensure_same_device(self, target_device: str) -> None:
        """Move teacher and student to *target_device* if not already there."""
        device = torch.device(target_device)
        teacher_device = next(self.teacher.parameters(), torch.tensor(0)).device
        student_device = next(self.student.parameters(), torch.tensor(0)).device

        if teacher_device != device:
            logger.info("Moving teacher from %s to %s", teacher_device, device)
            self.teacher = self.teacher.to(device)
        if student_device != device:
            logger.info("Moving student from %s to %s", student_device, device)
            self.student = self.student.to(device)

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def available_presets(self) -> Iterable[str]:
        """Return iterable of preset identifiers."""
        return list_presets()

    def describe_preset(self, name: str) -> str:
        """Return human-friendly description string."""
        return describe_preset(name)

    def resolve_preset(self, goal: Optional[Union[str, Goal]], preset: Optional[str]) -> str:
        """Resolve a goal or preset name into a concrete preset identifier.

        Args:
            goal: High-level goal string or :class:`Goal` enum member.
            preset: Explicit preset name (takes priority over *goal*).

        Returns:
            Preset identifier string.

        Raises:
            ValueError: If *goal* is not a recognized goal string.
        """
        if preset:
            return preset
        if goal:
            goal_str = goal.value if isinstance(goal, Goal) else str(goal)
            _validate_goal(goal_str)
            return self.GOAL_TO_PRESET.get(goal_str.lower(), self.default_preset)
        return self.default_preset

    def build_plan(
        self,
        *,
        goal: Optional[Union[str, Goal]] = None,
        preset: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a distillation configuration ready for :class:`MultiStageDistiller`.

        Args:
            goal: High-level business goal (mapped to preset).
            preset: Explicit preset name.
            overrides: Optional dict merged into preset configuration.
            context: Optional metadata stored under ``metadata`` in resulting plan.

        Returns:
            Plan dict consumable by :meth:`run`.

        Raises:
            ValueError: If *goal* is not recognized.
        """
        preset_name = self.resolve_preset(goal, preset)

        # Merge DistillationConfig overrides
        effective_overrides: Dict[str, Any] = {}
        if self.config:
            effective_overrides = self.config.to_overrides()
        if overrides:
            effective_overrides = self._deep_merge(effective_overrides, overrides)

        plan = get_preset(preset_name)
        if effective_overrides:
            plan = self._deep_merge(plan, effective_overrides)
        if context:
            plan.setdefault("metadata", {})["context"] = dict(context)
            plan["metadata"]["preset"] = preset_name
        else:
            plan.setdefault("metadata", {})["preset"] = preset_name

        logger.debug(
            "Built plan with preset=%s, %d stages",
            preset_name,
            len(plan.get("distillation", {}).get("stages", [])),
        )
        return plan

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def run(
        self,
        plan: Optional[Dict[str, Any]] = None,
        train_loader: Any = None,
        val_loader: Any = None,
        *,
        output_dir: str = "experiments/auto_suite",
        dry_run: bool = False,
    ) -> Any:
        """Execute a plan via :class:`MultiStageDistiller`.

        Args:
            plan: Distillation configuration (build via :meth:`build_plan`).
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            output_dir: Directory where reports/checkpoints are stored.
            dry_run: When ``True`` returns the plan without executing training.

        Returns:
            Training report dict with metrics and ``best_model_path``.
        """
        plan = plan or self.build_plan()
        if dry_run:
            return deepcopy(plan)

        logger.info("Starting distillation run (output_dir=%s)", output_dir)

        orchestrator = MultiStageDistiller(
            teacher=self.teacher,
            student=self.student,
            config=plan,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self._resolve_device(),
            output_dir=output_dir,
        )
        report = orchestrator.run()

        best_model_path = str(Path(output_dir) / "student_model")
        report["best_model_path"] = best_model_path
        return report

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        goal: Optional[Union[str, Goal]] = None,
        preset: Optional[str] = None,
        output_dir: str = "experiments/auto_suite",
    ) -> str:
        """Alias for run() that returns only the path to the best model.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            goal: High-level goal string or :class:`Goal` enum.
            preset: Explicit preset name.
            output_dir: Directory for outputs.

        Returns:
            Path to the best saved student model.
        """
        plan = self.build_plan(goal=goal, preset=preset)
        report = self.run(
            plan=plan, train_loader=train_loader, val_loader=val_loader, output_dir=output_dir
        )
        return report.get("best_model_path", output_dir)

    def evaluate(
        self,
        val_loader: Any,
        *,
        tokenizer: Any = None,
        loss_fn: Any = None,
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """Evaluate the student model on a validation set.

        Args:
            val_loader: Validation DataLoader.
            tokenizer: Tokenizer (used for text decoding in reports).
            loss_fn: Optional loss function for computing validation loss.
            task_type: ``'classification'``, ``'multi_label'``, or ``'regression'``.

        Returns:
            Dictionary with metrics, loss, diagnostics, and runtime stats.
        """
        from zynthe.evaluation.evaluator import Evaluator

        evaluator = Evaluator(
            model=self.student,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=self._resolve_device(),
            loss_fn=loss_fn,
            task_type=task_type,
        )
        return evaluator.evaluate()

    def compare(
        self,
        val_loader: Any,
        *,
        tokenizer: Any = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run teacher vs student comparison with metrics and optional visualizations.

        Args:
            val_loader: Validation DataLoader.
            tokenizer: Tokenizer instance.
            save_dir: If provided, generate comparison visualizations and report.

        Returns:
            Dictionary with teacher_results, student_results, and comparison summary.
        """
        from zynthe.evaluation.model_comparison import ModelComparator

        comparator = ModelComparator(
            self.teacher,
            self.student,
            tokenizer=tokenizer,
            device=self._resolve_device(),
        )
        teacher_results, student_results = comparator.compare_models(val_loader)

        if save_dir:
            comparator.visualize_comparison(teacher_results, student_results, save_dir)
            comparator.save_results(teacher_results, student_results, save_dir)
            comparator.generate_report(teacher_results, student_results, save_dir)

        return {
            "teacher": teacher_results,
            "student": student_results,
            "compression_ratio": comparator.compression_ratio,
            "accuracy_gap": teacher_results.get("accuracy", 0) - student_results.get("accuracy", 0),
            "f1_gap": teacher_results.get("f1", 0) - student_results.get("f1", 0),
        }

    def preview(self, plan: Optional[Dict[str, Any]] = None) -> None:
        """Print a human-readable summary of *plan* without executing anything.

        Args:
            plan: Distillation plan dict (from :meth:`build_plan`).
                  Defaults to ``build_plan()`` with the default preset.
        """
        plan = plan or self.build_plan()
        preset_name = plan.get("metadata", {}).get("preset", "unknown")
        distill_cfg = plan.get("distillation", {})
        stages = distill_cfg.get("stages", [])
        is_multi = distill_cfg.get("multi_stage", False)
        train_cfg = plan.get("train", plan.get("training", {}))

        lines = [
            f"\n{'='*56}",
            "  Distillation Plan Preview",
            f"{'='*56}",
            f"  Preset    : {preset_name}",
            f"  Multi-stage: {is_multi}",
            f"  Stages    : {len(stages)}",
        ]
        if train_cfg:
            epochs = train_cfg.get("epochs", "?")
            bs = train_cfg.get("batch_size", "?")
            lines.append(f"  Epochs    : {epochs}  |  Batch size: {bs}")
        desc = plan.get(
            "description", describe_preset(preset_name) if preset_name in list_presets() else ""
        )
        lines.append(f"  Description: {desc}")
        lines.append("")
        for i, stage in enumerate(stages, 1):
            name = stage.get("name", f"Stage {i}")
            stype = stage.get("type", "?")
            sepochs = stage.get("epochs", "?")
            deps = stage.get("depends_on", [])
            dep_str = f"  (after stage(s) {deps})" if deps else ""
            lines.append(f"  [{i}] {name}")
            lines.append(f"       type={stype}, epochs={sepochs}{dep_str}")
        lines.append(f"{'='*56}\n")
        logger.info("\n".join(lines))

    def save_plan(self, plan: Dict[str, Any], path: str | Path) -> Path:
        """Persist a plan to JSON for auditing or later reuse.

        Args:
            plan: Plan dict to save.
            path: Destination file path.

        Returns:
            The resolved :class:`Path` where the plan was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(plan, fh, indent=2)
        logger.info("Plan saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        """Auto-detect the best available device."""
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _deep_merge(self, base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
        """Recursively merge *overrides* into *base* without mutating either."""
        merged = deepcopy(base)
        for key, value in overrides.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged


# ---------------------------------------------------------------------------
# Distiller — beginner-friendly wrapper
# ---------------------------------------------------------------------------


class Distiller:
    """Beginner-friendly one-shot wrapper around :class:`DistillationToolkit`.

    This is a simpler interface that mirrors the pattern common in popular
    ML libraries (scikit-learn, fastai), letting users get started without
    knowing about presets or plans::

        from zynthe import Distiller

        distiller = Distiller(teacher, student, goal="balanced")
        distiller.fit(train_loader, val_loader, epochs=3)

    Args:
        teacher: Pre-trained teacher model (``torch.nn.Module``).
        student: Student model to be distilled (``torch.nn.Module``).
        goal: High-level training goal. One of ``'quick'``, ``'balanced'``,
            ``'compression'``, ``'vision'``, ``'causal_lm'``, ``'multimodal'``,
            ``'clip'``, ``'code'``, ``'text'``. Defaults to ``'balanced'``.
            Also accepts :class:`Goal` enum members.
        device: Device string (``'cuda'``, ``'cpu'``, ``'mps'``).
            Auto-detected if ``None``.
        config: Optional :class:`DistillationConfig` with hyperparameter overrides.

    Raises:
        TypeError: If teacher or student is not a ``torch.nn.Module``.
        ValueError: If *goal* is not recognized.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        goal: Union[str, Goal] = "balanced",
        device: Optional[str] = None,
        config: Optional[DistillationConfig] = None,
    ) -> None:
        goal_str = goal.value if isinstance(goal, Goal) else str(goal)
        _validate_goal(goal_str)
        self._toolkit = DistillationToolkit(
            teacher,
            student,
            device=device,
            config=config,
        )
        self._goal = goal_str

    def fit(
        self,
        train_loader: Any,
        val_loader: Any = None,
        *,
        epochs: Optional[int] = None,
        output_dir: str = "./distill_output",
    ) -> Dict[str, Any]:
        """Train the student model using knowledge distillation.

        Args:
            train_loader: PyTorch DataLoader for training data.
            val_loader: Optional PyTorch DataLoader for validation data.
            epochs: Override the number of training epochs from the preset.
            output_dir: Directory to save checkpoints and reports.

        Returns:
            Training report dict with metrics and ``best_model_path``.
        """
        overrides: Dict[str, Any] = {}
        if epochs is not None:
            overrides["training"] = {"epochs": epochs}
            overrides["train"] = {"epochs": epochs}

        plan = self._toolkit.build_plan(
            goal=self._goal,
            overrides=overrides if overrides else None,
        )
        return self._toolkit.run(
            plan=plan,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
        )

    def evaluate(self, val_loader: Any, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate the distilled student model.

        Args:
            val_loader: Validation DataLoader.
            **kwargs: Forwarded to :meth:`DistillationToolkit.evaluate`.

        Returns:
            Evaluation metrics dict.
        """
        return self._toolkit.evaluate(val_loader, **kwargs)

    def compare(self, val_loader: Any, **kwargs: Any) -> Dict[str, Any]:
        """Compare teacher vs student side-by-side.

        Args:
            val_loader: Validation DataLoader.
            **kwargs: Forwarded to :meth:`DistillationToolkit.compare`.

        Returns:
            Comparison results dict.
        """
        return self._toolkit.compare(val_loader, **kwargs)


__all__ = ["DistillationToolkit", "Distiller", "Goal", "DistillationConfig"]
