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

    def evaluate(
        self,
        val_loader,
        *,
        tokenizer=None,
        loss_fn=None,
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """Evaluate the student model on a validation set.

        Args:
            val_loader: Validation DataLoader.
            tokenizer: Tokenizer (used for text decoding in reports).
            loss_fn: Optional loss function for computing validation loss.
            task_type: 'classification', 'multi_label', or 'regression'.

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
        val_loader,
        *,
        tokenizer=None,
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

        print(f"\n{'='*56}")
        print(f"  Distillation Plan Preview")
        print(f"{'='*56}")
        print(f"  Preset    : {preset_name}")
        print(f"  Multi-stage: {is_multi}")
        print(f"  Stages    : {len(stages)}")
        if train_cfg:
            epochs = train_cfg.get("epochs", "?")
            bs = train_cfg.get("batch_size", "?")
            print(f"  Epochs    : {epochs}  |  Batch size: {bs}")
        print(f"  Description: {plan.get('description', describe_preset(preset_name) if preset_name in list_presets() else '')}")
        print()
        for i, stage in enumerate(stages, 1):
            name = stage.get("name", f"Stage {i}")
            stype = stage.get("type", "?")
            sepochs = stage.get("epochs", "?")
            deps = stage.get("depends_on", [])
            dep_str = f"  (after stage(s) {deps})" if deps else ""
            print(f"  [{i}] {name}")
            print(f"       type={stype}, epochs={sepochs}{dep_str}")
        print(f"{'='*56}\n")

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
