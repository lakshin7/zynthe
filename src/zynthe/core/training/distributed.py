"""Distributed-training integration (Phase 5 Iteration 1).

Wraps Zynthé's training loop in HuggingFace ``accelerate`` so a
single config flag moves the run from single-GPU to DDP without any
caller-side changes.

Usage::

    from zynthe.core.training.distributed import (
        DistributedConfig,
        prepare_distillation,
    )

    cfg = DistributedConfig(
        enabled=True,
        mixed_precision="bf16",
        num_processes=2,
    )
    bundle = prepare_distillation(teacher, student, optim, loader, cfg)
    # bundle.teacher, bundle.student, bundle.optim, bundle.loader
    # are now the ``accelerator.prepare(...)``-wrapped versions.

The actual DDP launch is the caller's responsibility — see
``scripts/smoke/run_distributed.py`` for the multi-GPU Modal entry
point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for accelerate-based distributed training.

    Attributes:
        enabled: when True, run the pipeline through ``accelerate``.
        mixed_precision: ``"no"`` / ``"fp16"`` / ``"bf16"``.  Defaults
            to ``"no"`` (no autocast) for safety.
        num_processes: number of GPUs.  ``"auto"`` defers to
            accelerate's environment-based discovery.
        mixed_precision_dtype: explicit dtype override (advanced).
        gradient_accumulation_steps: micro-batches per optimiser step.
        cpu: force CPU even when GPUs are present.  Useful for
            unit tests and Modal CPU-only fallback.
    """

    enabled: bool = False
    mixed_precision: str = "no"
    num_processes: Any = "auto"
    mixed_precision_dtype: Optional[str] = None
    gradient_accumulation_steps: int = 1
    cpu: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"mixed_precision must be 'no', 'fp16', or 'bf16'; got "
                f"{self.mixed_precision!r}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1; got "
                f"{self.gradient_accumulation_steps}"
            )


def prepare_distillation(
    teacher: nn.Module,
    student: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    dataloader: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    config: Optional[DistributedConfig] = None,
):
    """Wrap Zynthé models (+ optional optim / dataloader / scheduler)
    in ``accelerator.prepare(...)`` for distributed training.

    If ``config`` is ``None`` or ``config.enabled`` is ``False``, the
    function is a no-op and returns the inputs unchanged (just bundled
    in a ``PreparedBundle`` for the caller's convenience).
    """
    cfg = config or DistributedConfig()
    if not cfg.enabled:
        return _PreparedBundle(teacher, student, optimizer, dataloader, scheduler, accelerator=None)

    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError(
            "accelerate is not installed.  pip install accelerate to use "
            "DistributedConfig(enabled=True)."
        ) from exc

    kwargs: Dict[str, Any] = {"mixed_precision": cfg.mixed_precision}
    if cfg.mixed_precision_dtype is not None:
        kwargs["mixed_precision_dtype"] = cfg.mixed_precision_dtype
    if cfg.cpu:
        kwargs["cpu"] = True
    elif cfg.num_processes != "auto":
        kwargs["num_processes"] = cfg.num_processes
    if cfg.gradient_accumulation_steps > 1:
        kwargs["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps
    kwargs.update(cfg.extra)

    accelerator = Accelerator(**kwargs)
    prepared = accelerator.prepare(teacher, student, optimizer, dataloader, scheduler)
    # ``accelerator.prepare(...)`` returns the same number of args as
    # passed in.  When some are None, the corresponding position is
    # None.
    if len(prepared) == 5:
        t, s, o, d, sc = prepared
    elif len(prepared) == 4:
        t, s, o, d = prepared
        sc = None
    else:
        raise RuntimeError(
            f"Unexpected number of prepared args from accelerator.prepare: "
            f"{len(prepared)}"
        )
    return _PreparedBundle(t, s, o, d, sc, accelerator=accelerator)


@dataclass
class _PreparedBundle:
    teacher: nn.Module
    student: nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    dataloader: Optional[Any]
    scheduler: Optional[Any]
    accelerator: Optional[Any] = None

    def step(self) -> None:
        """Convenience: call ``.step()`` on the accelerator (if any).

        accelerate's ``.step()`` does a no-op when no scheduler is
        configured, so calling it unconditionally is safe.
        """
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
