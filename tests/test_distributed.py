"""Tests for Phase-5 distributed-training integration (Iteration 1).

Pins:
* ``DistributedConfig`` validation.
* ``prepare_distillation`` is a no-op when ``enabled=False``.
* When ``enabled=True`` and accelerate is installed, the bundled
  objects match the caller's expected signatures.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from zynthe.core.training.distributed import (
    DistributedConfig,
    _PreparedBundle,
    prepare_distillation,
)


class _TinyMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


# ----------------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------------


def test_config_defaults_are_off() -> None:
    c = DistributedConfig()
    assert c.enabled is False
    assert c.mixed_precision == "no"
    assert c.num_processes == "auto"
    assert c.gradient_accumulation_steps == 1
    assert c.cpu is False


def test_config_rejects_invalid_precision() -> None:
    with pytest.raises(ValueError, match="mixed_precision"):
        DistributedConfig(enabled=True, mixed_precision="bogus")


def test_config_rejects_zero_grad_accumulation() -> None:
    with pytest.raises(ValueError, match="gradient_accumulation"):
        DistributedConfig(gradient_accumulation_steps=0)


# ----------------------------------------------------------------------------
# No-op when disabled
# ----------------------------------------------------------------------------


def test_prepare_distillation_is_noop_when_disabled() -> None:
    teacher = _TinyMod()
    student = _TinyMod()
    opt = torch.optim.SGD(student.parameters(), lr=0.01)
    bundle = prepare_distillation(teacher, student, opt, dataloader=None, config=DistributedConfig())
    assert bundle.teacher is teacher
    assert bundle.student is student
    assert bundle.optimizer is opt
    assert bundle.accelerator is None
    # step() is a no-op when no accelerator.
    bundle.step()


def test_prepare_distillation_is_noop_when_config_is_none() -> None:
    teacher = _TinyMod()
    student = _TinyMod()
    bundle = prepare_distillation(teacher, student, config=None)
    assert bundle.teacher is teacher
    assert bundle.accelerator is None


# ----------------------------------------------------------------------------
# Real accelerate path (only when accelerate is importable)
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("accelerate", reason="accelerate not installed"),
    reason="accelerate not installed",
)
def test_prepare_distillation_with_accelerate_on_cpu() -> None:
    """When accelerate is available, run a CPU-only prepare.

    Verifies the bundle is created and the accelerator object is
    non-None.  accelerate's ``prepare`` returns the same number of
    inputs; we just check shapes / types line up.
    """
    from accelerate import Accelerator  # noqa: F401 — ensure import works

    teacher = _TinyMod()
    student = _TinyMod()
    opt = torch.optim.SGD(student.parameters(), lr=0.01)
    cfg = DistributedConfig(
        enabled=True,
        mixed_precision="no",
        cpu=True,
    )
    bundle = prepare_distillation(teacher, student, opt, dataloader=None, config=cfg)
    assert bundle.accelerator is not None
    # The wrapped student should still accept a forward call.
    x = torch.randn(2, 4)
    out = bundle.student(x)
    assert out.shape == (2, 4)
    # step() should be a no-op for our use-case (we have no scheduler).
    bundle.step()
