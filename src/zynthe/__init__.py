"""
Zynthé — Universal Knowledge Distillation Toolkit
====================================================

A modular Python library for knowledge distillation across text, code,
vision, VL (vision-language), and multimodal model architectures.

Quick Start::

    from zynthe import DistillationToolkit

    toolkit = DistillationToolkit(teacher_model, student_model)
    plan = toolkit.build_plan(goal="balanced")
    toolkit.preview(plan)
    report = toolkit.run(plan, train_loader, val_loader)

Individual distillers::

    from zynthe import KDHintonDistiller, FeatureDistiller

Adapter system for multi-modality::

    from zynthe import AdapterRegistry
    registry = AdapterRegistry()
    adapter = registry.detect(model)

Evaluation & Visualization::

    from zynthe import Evaluator, EvaluationReport, ModelComparator
    from zynthe import StudentInference

Pipeline builder::

    from zynthe import PipelineBuilder

Preflight analysis::

    from zynthe import PreflightAnalyzer, run_preflight_check
"""

from __future__ import annotations


__version__ = "0.2.4"

# ---------------------------------------------------------------------------
# Primary user-facing API
# ---------------------------------------------------------------------------
from zynthe.core.distillers.toolkit import (
    DistillationToolkit,
    Distiller,
    Goal,
    DistillationConfig,
)

# ---------------------------------------------------------------------------
# Distillation methods
# ---------------------------------------------------------------------------
from zynthe.core.distillers import (
    KDHintonDistiller,
    AttentionTransferDistiller,
    FeatureDistiller,
    SimilarityTransfer,
    SafeCausalLMTrainer,
    list_presets,
    describe_preset,
    get_preset,
)

# ---------------------------------------------------------------------------
# Multi-modality adapters
# ---------------------------------------------------------------------------
from zynthe.core.adapters import AdapterRegistry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from zynthe.core.config.config_manager import ConfigManager

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
from zynthe.core.models import (
    ModelBundle,
    ModelLoader,
    ModelWrapper,
    get_device,
    load_models,
    model_summary,
    save_model,
    load_model,
    save_checkpoint,
    load_checkpoint,
    export_onnx,
    export_torchscript,
    CheckpointMetadata,
    ProjectionHead,
    ProjectionHeadFactory,
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
from zynthe.evaluation import Evaluator, DualEvaluator, CurriculumEvaluator
from zynthe.evaluation.evaluation_report import EvaluationReport
from zynthe.evaluation.model_comparison import ModelComparator

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
from zynthe.core.inference.predict import StudentInference

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
from zynthe.core.pipelines import PipelineBuilder, PipelineRegistry, get_registry

# ---------------------------------------------------------------------------
# Preflight analysis
# ---------------------------------------------------------------------------
from zynthe.core.preflight import PreflightAnalyzer, run_preflight_check

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
from zynthe.core.quant import PTQRunner, QATRunner, apply_ptq

# ---------------------------------------------------------------------------
# Runtime (advanced / orchestration)
# ---------------------------------------------------------------------------
from zynthe.app.runtime import UnifiedTrainingRuntime, RuntimeOptions, RuntimeResult

__all__ = [
    # Version
    "__version__",
    # Primary API
    "DistillationToolkit",
    # Distillers
    "KDHintonDistiller",
    "AttentionTransferDistiller",
    "FeatureDistiller",
    "SimilarityTransfer",
    "SafeCausalLMTrainer",
    # Presets
    "list_presets",
    "describe_preset",
    "get_preset",
    # Adapters
    "AdapterRegistry",
    # Config
    "ConfigManager",
    # Models
    "ModelBundle",
    "ModelLoader",
    "ModelWrapper",
    "get_device",
    "load_models",
    "model_summary",
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    "export_onnx",
    "export_torchscript",
    "CheckpointMetadata",
    "ProjectionHead",
    "ProjectionHeadFactory",
    # Evaluation
    "Evaluator",
    "DualEvaluator",
    "CurriculumEvaluator",
    "EvaluationReport",
    "ModelComparator",
    # Inference
    "StudentInference",
    # Pipelines
    "PipelineBuilder",
    "PipelineRegistry",
    "get_registry",
    # Preflight
    "PreflightAnalyzer",
    "run_preflight_check",
    # Quantization
    "PTQRunner",
    "QATRunner",
    "apply_ptq",
    # Runtime
    "UnifiedTrainingRuntime",
    "RuntimeOptions",
    "RuntimeResult",
    # Config & Enums
    "Goal",
    "DistillationConfig",
]
