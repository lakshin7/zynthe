"""Causal-LM distillation core (stable trainer, loss, checkpoints, metrics)."""

from __future__ import annotations

from .checkpoint import (
    CheckpointLoadReport,
    CheckpointMeta,
    TrainingState,
    save_training_checkpoint,
    smart_load_checkpoint,
)
from .determinism import (
    DeterminismReport,
    DeterminismTrace,
    runtime_determinism_env,
    trace_from_trainer,
    verify_reproducibility,
)
from .distillation import CausalLMDistillationEngine, DistillationConfig, DistillationLossOutput
from .fault_injection import FaultInjectionConfig, FaultInjector
from .metrics import (
    DistillationHealthMetrics,
    MetricStabilityMonitor,
    TokenMetricsAccumulator,
    compute_distill_alignment,
)
from .regression_gate import RegressionGate, RegressionGateConfig, RegressionReport
from .trainer import SafeCausalLMTrainer
from .validation import (
    CheckpointStressReport,
    GradientSanityReport,
    NumericalValidationReport,
    TrainingHealthReport,
    gradient_sanity_check,
    run_checkpoint_stress_tests,
    validate_distillation_numerics,
)

__all__ = [
    "CausalLMDistillationEngine",
    "DistillationConfig",
    "DistillationLossOutput",
    "CheckpointMeta",
    "CheckpointLoadReport",
    "TrainingState",
    "save_training_checkpoint",
    "smart_load_checkpoint",
    "DistillationHealthMetrics",
    "MetricStabilityMonitor",
    "TokenMetricsAccumulator",
    "compute_distill_alignment",
    "NumericalValidationReport",
    "GradientSanityReport",
    "TrainingHealthReport",
    "CheckpointStressReport",
    "validate_distillation_numerics",
    "gradient_sanity_check",
    "run_checkpoint_stress_tests",
    "FaultInjectionConfig",
    "FaultInjector",
    "DeterminismReport",
    "DeterminismTrace",
    "runtime_determinism_env",
    "trace_from_trainer",
    "verify_reproducibility",
    "RegressionGate",
    "RegressionGateConfig",
    "RegressionReport",
    "SafeCausalLMTrainer",
]
