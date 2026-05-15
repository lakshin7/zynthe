"""
Zynthé Pipeline System
======================

End-to-end distillation pipelines for modular, composable knowledge distillation.

Core Components:
- BasePipeline: Abstract pipeline interface
- SingleDistillerPipeline: Wrap individual distillers
- MultiStagePipeline: Compose multiple distillers (sequential, parallel, hybrid)
- PipelineBuilder: Fluent API for building pipelines
- PipelineRegistry: Discovery and instantiation

Usage:
    from zynthe.core.pipelines import PipelineBuilder

    # Single distiller
    pipeline = PipelineBuilder() \
        .add_distiller('kd_hinton', temperature=4.0) \
        .build(teacher, student, device)

    # Multi-stage
    pipeline = PipelineBuilder() \
        .add_stage('logit', weight=0.7) \
            .add_distiller('kd_hinton') \
        .add_stage('features', weight=0.3) \
            .add_distiller('feature') \
        .build(teacher, student, device)
"""

from __future__ import annotations

from .base_pipeline import BasePipeline, PipelineMetrics
from .multi_stage_pipeline import ExecutionMode, MultiStagePipeline, PipelineStage
from .pipeline_builder import PipelineBuilder
from .pipeline_registry import PipelineRegistry, get_registry
from .single_distiller_pipeline import SingleDistillerPipeline

__all__ = [
    "BasePipeline",
    "PipelineMetrics",
    "SingleDistillerPipeline",
    "PipelineRegistry",
    "get_registry",
    "MultiStagePipeline",
    "ExecutionMode",
    "PipelineStage",
    "PipelineBuilder",
]

__version__ = "1.0.0"
