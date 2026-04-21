"""High level orchestration helpers for knowledge distillation.

The :class:`DistillationToolkit` provides an opinionated, user-friendly façade
around the lower-level distiller modules. It lets product teams assemble
multi-stage distillation plans by selecting a preset or specifying a high-level
business goal (e.g. "compression" or "vision_transformer") without touching the
underlying research code.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import json
import torch

from .multi_stage_distiller import MultiStageDistiller
from .presets import get_preset, list_presets, describe_preset


class DistillationToolkit:
    """Simple orchestration layer for non-expert usage.

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
    }

    def __init__(
        self,
        teacher,
        student,
        *,
        device: Optional[str] = None,
        default_preset: str = "balanced",
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.device = device
        self.default_preset = default_preset

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def available_presets(self) -> Iterable[str]:
        """Return iterable of preset identifiers."""
        return list_presets()

    def describe_preset(self, name: str) -> str:
        """Return human-friendly description string."""
        return describe_preset(name)

    def resolve_preset(self, goal: Optional[str], preset: Optional[str]) -> str:
        if preset:
            return preset
        if goal:
            return self.GOAL_TO_PRESET.get(goal.lower(), self.default_preset)
        return self.default_preset

    def build_plan(
        self,
        *,
        goal: Optional[str] = None,
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
        """
        preset_name = self.resolve_preset(goal, preset)
        plan = get_preset(preset_name)
        if overrides:
            plan = self._deep_merge(plan, overrides)
        if context:
            plan.setdefault("metadata", {})["context"] = dict(context)
            plan["metadata"]["preset"] = preset_name
        else:
            plan.setdefault("metadata", {})["preset"] = preset_name
        return plan

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def run(
        self,
        plan: Optional[Dict[str, Any]] = None,
        train_loader=None,
        val_loader=None,
        *,
        output_dir: str = "experiments/auto_suite",
        dry_run: bool = False,
    ) -> Any:
        """Execute a plan via :class:`MultiStageDistiller`.

        Args:
            plan: Distillation configuration (build via :meth:`build_plan`).
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            output_dir: Directory where reports/checkpoints are stored.
            dry_run: When ``True`` returns the plan without executing training.
        """
        plan = plan or self.build_plan()
        if dry_run:
            return deepcopy(plan)

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
        
        # Determine the best model path (the last stage's student model)
        # Note: MultiStageDistiller's orchestrator saves to output_dir
        best_model_path = str(Path(output_dir) / "student_model")
        # In case the orchestrator saves in subdirectories, we could enhance this
        
        # Add best model path to report
        report['best_model_path'] = best_model_path
        return report

    def train(
        self,
        train_loader,
        val_loader,
        goal: Optional[str] = None,
        preset: Optional[str] = None,
        output_dir: str = "experiments/auto_suite",
    ) -> str:
        """Alias for run() that returns only the path to the best model."""
        plan = self.build_plan(goal=goal, preset=preset)
        report = self.run(plan=plan, train_loader=train_loader, val_loader=val_loader, output_dir=output_dir)
        return report.get('best_model_path', output_dir)

    def save_plan(self, plan: Dict[str, Any], path: str | Path) -> Path:
        """Persist a plan to JSON for auditing or later reuse."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(plan, fh, indent=2)
        return path

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _deep_merge(self, base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in overrides.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, Mapping)
            ):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged


__all__ = ["DistillationToolkit"]
