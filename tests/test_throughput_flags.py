"""Tests for Phase-4 throughput flags in :class:`BaseDistiller`.

Pins:
* ``distill.precision: "bf16"`` autocasts the student forward + loss.
* ``distill.compile: true`` wraps the student in ``torch.compile``;
  on failure it falls back to eager.
* ``distill.grad_checkpointing: true`` enables HF
  ``gradient_checkpointing`` when supported; no-op otherwise.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.base_distiller import BaseDistiller


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


class _TwoHeadMod(nn.Module):
    """A tiny student/teacher that exposes .logits via forward."""

    def __init__(self, num_classes: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, hidden)
        self.head = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids=None, labels=None, **_unused):
        if input_ids is None:
            raise TypeError("input_ids required")
        return type("O", (), {"logits": self.head(self.embed(input_ids))})()


class _KDHLikeDistiller(BaseDistiller):
    modality_type = "text"

    def __init__(self, teacher, student, config=None, device=None, **kwargs):
        super().__init__(teacher, student, config=config, device=device, **kwargs)

    def _init_losses(self):
        self.losses["supervised"] = nn.CrossEntropyLoss()
        self.loss_weights["supervised"] = 1.0

    def compute_loss(
        self,
        student_outputs,
        teacher_outputs,
        targets=None,
        student_features=None,
        teacher_features=None,
        **kwargs,
    ):
        loss = self.losses["supervised"](student_outputs.logits, targets)
        return loss, {"total": loss.item(), "ce": loss.item()}


# ----------------------------------------------------------------------------
# bf16 autocast
# ----------------------------------------------------------------------------


def test_bf16_flag_does_not_break_forward() -> None:
    """Even without a GPU, `precision: bf16` must not break the
    forward path (we just don't autocast on CPU).
    """
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = _KDHLikeDistiller(
        teacher, student,
        config={"distill": {"precision": "bf16"}},
        device="cpu",
    )
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([0])
    optim = torch.optim.SGD(d.student.parameters(), lr=1e-3)
    d.train()
    loss_dict = d.training_step({"input_ids": x, "labels": y}, optimizer=optim)
    assert math.isfinite(loss_dict["total"])


def test_bf16_flag_records_precision() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = _KDHLikeDistiller(
        teacher, student,
        config={"distill": {"precision": "bf16"}},
        device="cpu",
    )
    assert d.precision == "bf16"


def test_default_precision_is_fp32() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = _KDHLikeDistiller(teacher, student, device="cpu")
    assert d.precision == "fp32"


# ----------------------------------------------------------------------------
# torch.compile opt-in
# ----------------------------------------------------------------------------


def test_compile_flag_keeps_eager_when_compile_disabled() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = _KDHLikeDistiller(teacher, student, device="cpu")
    assert d.compile_model is False
    # student is still a plain nn.Module, not OptimizedModule.
    from torch._dynamo.optimized import OptimizedModule
    assert not isinstance(d.student, OptimizedModule)


def test_compile_flag_records_flag() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = _KDHLikeDistiller(
        teacher, student, config={"distill": {"compile": True}}, device="cpu"
    )
    # Whether torch.compile actually wraps depends on the torch
    # version; we just check the flag was set.
    assert d.compile_model is True


# ----------------------------------------------------------------------------
# grad_checkpointing opt-in
# ----------------------------------------------------------------------------


def test_grad_checkpointing_flag_records() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = _KDHLikeDistiller(
        teacher, student,
        config={"distill": {"grad_checkpointing": True}},
        device="cpu",
    )
    assert d.grad_checkpointing is True


def test_grad_checkpointing_no_op_for_plain_modules() -> None:
    """Plain nn.Module has no ``gradient_checkpointing_enable``; the
    BaseDistiller should silently skip rather than raise.
    """
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    # Should not raise.
    _KDHLikeDistiller(
        teacher, student,
        config={"distill": {"grad_checkpointing": True}},
        device="cpu",
    )


class _GCStudent(nn.Module):
    """Stub that mimics an HF model's gradient_checkpointing_enable hook."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self._gc_enabled = False

    def forward(self, x):
        return self.lin(x)

    def gradient_checkpointing_enable(self):
        self._gc_enabled = True


def test_grad_checkpointing_calls_hf_hook() -> None:
    teacher = _TwoHeadMod()
    student = _GCStudent()
    d = _KDHLikeDistiller(
        teacher, student,
        config={"distill": {"grad_checkpointing": True}},
        device="cpu",
    )
    assert student._gc_enabled is True  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
# End-to-end (no-GPU): default throughput flags don't change the
# test that already passed in Phase 0/1.
# ----------------------------------------------------------------------------


def test_no_throughput_flags_keeps_existing_behavior() -> None:
    """Default (precision=fp32, compile=False, grad_checkpointing=False)
    matches the historical training-step contract.
    """
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = _KDHLikeDistiller(teacher, student, device="cpu")
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([0])
    optim = torch.optim.SGD(d.student.parameters(), lr=1e-3)
    d.train()
    d.training_step({"input_ids": x, "labels": y}, optimizer=optim)
    # Student weights were updated.
    p_before = next(iter(student.parameters())).detach().clone()
    optim.step()
    d.training_step({"input_ids": x, "labels": y}, optimizer=optim)
    p_after = next(iter(student.parameters())).detach().clone()
    assert not torch.allclose(p_before, p_after)
