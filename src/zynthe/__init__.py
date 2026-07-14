"""
Zynthe - Universal Knowledge Distillation Toolkit.

The top-level package intentionally uses lazy exports so importing ``zynthe``
does not require optional extras such as evaluation plotting, vision, or ONNX
support. Public symbols keep the same import style::

    from zynthe import DistillationToolkit, Evaluator, ModelComparator
"""

from __future__ import annotations

import logging as _logging
from importlib import import_module
from typing import Any

__version__ = "0.3.0"

# ---------------------------------------------------------------------------
# Default logging — ensure zynthe loggers are visible in notebooks (Kaggle,
# Colab, Jupyter) even when the user has not configured the root logger.
# ---------------------------------------------------------------------------
_pkg_logger = _logging.getLogger("zynthe")
if not _pkg_logger.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    _pkg_logger.addHandler(_handler)
    _pkg_logger.setLevel(_logging.INFO)
    _pkg_logger.propagate = False

_EXPORTS: dict[str, tuple[str, str]] = {
    # Primary API
    "DistillationToolkit": ("zynthe.core.distillers.toolkit", "DistillationToolkit"),
    "Distiller": ("zynthe.core.distillers.toolkit", "Distiller"),
    "Goal": ("zynthe.core.distillers.toolkit", "Goal"),
    "DistillationConfig": ("zynthe.core.distillers.toolkit", "DistillationConfig"),
    # Distillers
    "KDHintonDistiller": ("zynthe.core.distillers.kd_hinton", "KDHintonDistiller"),
    "AttentionTransferDistiller": (
        "zynthe.core.distillers.attention_transfer",
        "AttentionTransferDistiller",
    ),
    "FeatureDistiller": ("zynthe.core.distillers.feature_distiller", "FeatureDistiller"),
    "SimilarityTransfer": ("zynthe.core.distillers.similarity_transfer", "SimilarityTransfer"),
    "SafeCausalLMTrainer": ("zynthe.core.distillers.causal_lm", "SafeCausalLMTrainer"),
    "list_presets": ("zynthe.core.distillers.presets", "list_presets"),
    "describe_preset": ("zynthe.core.distillers.presets", "describe_preset"),
    "get_preset": ("zynthe.core.distillers.presets", "get_preset"),
    # Adapters
    "AdapterRegistry": ("zynthe.core.adapters", "AdapterRegistry"),
    # Config
    "ConfigManager": ("zynthe.core.config.config_manager", "ConfigManager"),
    # Models
    "ModelBundle": ("zynthe.core.models", "ModelBundle"),
    "ModelLoader": ("zynthe.core.models", "ModelLoader"),
    "ModelWrapper": ("zynthe.core.models", "ModelWrapper"),
    "get_device": ("zynthe.core.models", "get_device"),
    "load_models": ("zynthe.core.models", "load_models"),
    "model_summary": ("zynthe.core.models", "model_summary"),
    "save_model": ("zynthe.core.models", "save_model"),
    "load_model": ("zynthe.core.models", "load_model"),
    "save_checkpoint": ("zynthe.core.models", "save_checkpoint"),
    "load_checkpoint": ("zynthe.core.models", "load_checkpoint"),
    "export_onnx": ("zynthe.core.models", "export_onnx"),
    "export_torchscript": ("zynthe.core.models", "export_torchscript"),
    "CheckpointMetadata": ("zynthe.core.models", "CheckpointMetadata"),
    "ProjectionHead": ("zynthe.core.models", "ProjectionHead"),
    "ProjectionHeadFactory": ("zynthe.core.models", "ProjectionHeadFactory"),
    # Evaluation
    "Evaluator": ("zynthe.evaluation", "Evaluator"),
    "DualEvaluator": ("zynthe.evaluation", "DualEvaluator"),
    "CurriculumEvaluator": ("zynthe.evaluation", "CurriculumEvaluator"),
    "EvaluationReport": ("zynthe.evaluation", "EvaluationReport"),
    "ModelComparator": ("zynthe.evaluation", "ModelComparator"),
    # Inference
    "StudentInference": ("zynthe.core.inference.predict", "StudentInference"),
    # Pipelines
    "PipelineBuilder": ("zynthe.core.pipelines", "PipelineBuilder"),
    "PipelineRegistry": ("zynthe.core.pipelines", "PipelineRegistry"),
    "get_registry": ("zynthe.core.pipelines", "get_registry"),
    # Preflight
    "PreflightAnalyzer": ("zynthe.core.preflight", "PreflightAnalyzer"),
    "run_preflight_check": ("zynthe.core.preflight", "run_preflight_check"),
    # Quantization
    "PTQRunner": ("zynthe.core.quant", "PTQRunner"),
    "QATRunner": ("zynthe.core.quant", "QATRunner"),
    "apply_ptq": ("zynthe.core.quant", "apply_ptq"),
    # Runtime
    "UnifiedTrainingRuntime": ("zynthe.app.runtime", "UnifiedTrainingRuntime"),
    "RuntimeOptions": ("zynthe.app.runtime", "RuntimeOptions"),
    "RuntimeResult": ("zynthe.app.runtime", "RuntimeResult"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'zynthe' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import zynthe.{name}. Install the matching optional extra "
            f"if this symbol depends on optional features. Original error: {exc}"
        ) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
