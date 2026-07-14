"""Training utilities with lightweight lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "DistributedConfig": ("zynthe.core.training.distributed", "DistributedConfig"),
    "prepare_distillation": ("zynthe.core.training.distributed", "prepare_distillation"),
    "MultiTaskT5Trainer": ("zynthe.core.training.rationale_trainer", "MultiTaskT5Trainer"),
    "OptimizerFactory": ("zynthe.training.optimizer", "OptimizerFactory"),
    "GradientManager": ("zynthe.training.optimizer", "GradientManager"),
    "AdaptiveOptimizer": ("zynthe.training.optimizer", "AdaptiveOptimizer"),
    "OptimizerCheckpoint": ("zynthe.training.optimizer", "OptimizerCheckpoint"),
    "LookaheadOptimizer": ("zynthe.training.optimizer", "LookaheadOptimizer"),
    "get_optimizer": ("zynthe.training.optimizer", "get_optimizer"),
    "clip_gradients": ("zynthe.training.optimizer", "clip_gradients"),
    "centralize_gradients": ("zynthe.training.optimizer", "centralize_gradients"),
    "inject_gradient_noise": ("zynthe.training.optimizer", "inject_gradient_noise"),
    "SchedulerFactory": ("zynthe.training.scheduler", "SchedulerFactory"),
    "WarmupScheduler": ("zynthe.training.scheduler", "WarmupScheduler"),
    "MultiStageScheduler": ("zynthe.training.scheduler", "MultiStageScheduler"),
    "AdaptiveScheduler": ("zynthe.training.scheduler", "AdaptiveScheduler"),
    "get_scheduler": ("zynthe.training.scheduler", "get_scheduler"),
    "Trainer": ("zynthe.training.trainer", "Trainer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'zynthe.training' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
