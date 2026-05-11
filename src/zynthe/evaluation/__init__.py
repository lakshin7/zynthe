"""Evaluation utilities with lazy optional imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "Evaluator": ("zynthe.evaluation.evaluator", "Evaluator"),
    "DualEvaluator": ("zynthe.evaluation.evaluator_extended", "DualEvaluator"),
    "CurriculumEvaluator": ("zynthe.evaluation.evaluator_extended", "CurriculumEvaluator"),
    "EvaluationReport": ("zynthe.evaluation.evaluation_report", "EvaluationReport"),
    "ModelComparator": ("zynthe.evaluation.model_comparison", "ModelComparator"),
    "build_eval_diagnostics": ("zynthe.evaluation.diagnostics", "build_eval_diagnostics"),
    "generate_report": ("zynthe.evaluation.report", "generate_report"),
    "plot_training_curves": ("zynthe.evaluation.visualizer", "plot_training_curves"),
    "plot_teacher_student_comparison": (
        "zynthe.evaluation.visualizer",
        "plot_teacher_student_comparison",
    ),
    "plot_epoch_micro_series": ("zynthe.evaluation.visualizer", "plot_epoch_micro_series"),
    "plot_metric_grid": ("zynthe.evaluation.visualizer", "plot_metric_grid"),
    "plot_calibration_curve": ("zynthe.evaluation.visualizer", "plot_calibration_curve"),
    "plot_runtime_profile": ("zynthe.evaluation.visualizer", "plot_runtime_profile"),
    "plot_evaluation_dashboard": ("zynthe.evaluation.visualizer", "plot_evaluation_dashboard"),
    "plot_distillation_gap": ("zynthe.evaluation.visualizer", "plot_distillation_gap"),
    "plot_extended_metrics": ("zynthe.evaluation.visualizer", "plot_extended_metrics"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'zynthe.evaluation' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "Evaluation features require the optional eval dependencies. "
            "Install with `pip install zynthe[eval]`."
        ) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
