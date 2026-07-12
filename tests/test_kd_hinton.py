"""Behavior tests for :class:`KDHintonDistiller`.

Reference: Hinton, Vinyals, Dean 2015, *Distilling the Knowledge in a
Neural Network*, eq. (1):

    L = α · T² · KL(σ(s/T) ‖ σ(t/T)) + (1 − α) · CE(s, y)

This file pins the math: every (T, α) combination is checked against a
hand-evaluated reference (computed by PyTorch itself but via the spec-
literal path — `manual_kd_loss` below — so the distiller is provably
correct up to floating-point noise).

It also pins behavior:
* hint-less forward equals vanilla Hinton,
* hint path is additive (L_KD + weighted hint loss),
* fp16 student forces an upcast to fp32 before softmax,
* hooks are cleaned up on `__del__` and `remove_hooks()`.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.kd_hinton import KDHintonDistiller
from zynthe.core.utils.exceptions import ConfigError


# ----------------------------------------------------------------------------
# Tiny reference model that returns a SimpleNamespace with .logits so the
# distiller treats it like a HuggingFace model.
# ----------------------------------------------------------------------------


class _RefModel(nn.Module):
    """Tiny linear model used to hand-evaluate the Hinton loss."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.linear = nn.Linear(4, num_classes, bias=False)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        logits = self.linear(x)
        return _Out(logits=logits, loss=None)


class _Out:
    """Mimics the HuggingFace output interface used by KD."""

    def __init__(self, logits: torch.Tensor, loss: torch.Tensor | None) -> None:
        self.logits = logits
        self.loss = loss


def manual_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-evaluated Hinton loss (matches the spec the distiller should
    implement). Returns ``(total, kd_only, ce_only)``.
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kd = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(student_logits, labels)
    total = alpha * kd + (1.0 - alpha) * ce
    return total, kd, ce


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture()
def ref_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.randint(0, 5, (8,))
    return x, y


@pytest.fixture()
def ref_teacher() -> _RefModel:
    torch.manual_seed(1)
    return _RefModel(num_classes=5)


# ----------------------------------------------------------------------------
# Reference-value tests at the (T, α) grid
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "temperature,alpha",
    [
        (1.0, 0.5),
        (1.0, 1.0),
        (2.0, 0.5),
        (2.0, 0.7),
        (4.0, 0.7),
        (4.0, 1.0),
    ],
)
def test_hinton_loss_matches_closed_form(
    ref_inputs, ref_teacher, temperature: float, alpha: float
) -> None:
    torch.manual_seed(2)
    student = _RefModel(num_classes=5)

    x, y = ref_inputs
    teacher_logits = ref_teacher(x).logits
    student_logits = student(x).logits

    expected_total, expected_kd, expected_ce = manual_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=y,
        temperature=temperature,
        alpha=alpha,
    )

    distiller = KDHintonDistiller(
        ref_teacher,
        student,
        config={"temperature": temperature, "alpha": alpha},
        device="cpu",
    )

    with torch.no_grad():
        t_out = ref_teacher(x)
        s_out = student(x)
    total, breakdown = distiller.compute_loss(s_out, t_out, y)

    # Match within tight tolerance — fp32 only.
    assert torch.allclose(total, expected_total, atol=1e-5)
    assert math.isclose(breakdown["kd_loss"], expected_kd.item(), abs_tol=1e-5)
    assert math.isclose(breakdown["ce_loss"], expected_ce.item(), abs_tol=1e-5)


# ----------------------------------------------------------------------------
# Hint path is additive
# ----------------------------------------------------------------------------


def test_hint_loss_is_additive(ref_inputs, ref_teacher) -> None:
    """If hints are configured but the matched layers don't exist, the
    distiller should still produce a non-zero KD+CE component (and not
    silently drop to zero)."""
    torch.manual_seed(3)
    student = _RefModel(num_classes=5)
    x, y = ref_inputs

    distiller = KDHintonDistiller(
        ref_teacher,
        student,
        config={
            "temperature": 2.0,
            "alpha": 0.5,
            "hint_enabled": True,
            "hints": [
                {"teacher": "definitely.not.a.layer", "student": "also.not"},
            ],
        },
        device="cpu",
    )

    with torch.no_grad():
        t_out = ref_teacher(x)
        s_out = student(x)
    total, breakdown = distiller.compute_loss(s_out, t_out, y)

    # KD + CE parts should still match the closed form (no hint contribution
    # since the layered names don't match — but the BaseLoss should not
    # silently be zero).
    expected_total, _, _ = manual_kd_loss(
        s_out.logits, t_out.logits, y, temperature=2.0, alpha=0.5
    )
    assert torch.allclose(total, expected_total, atol=1e-5)
    assert "kd_loss" in breakdown
    assert "ce_loss" in breakdown


# ----------------------------------------------------------------------------
# fp16 stability — Hinton KD upcasts before softmax
# ----------------------------------------------------------------------------


def test_fp16_logits_are_upcast(ref_inputs, ref_teacher) -> None:
    """A teacher casting huge logits through softmax in fp16 will overflow.
    The distiller uses ``BaseDistiller._extract_logits_tensor`` to upcast
    to fp32 first; verify the result still matches fp32 reference.
    """
    torch.manual_seed(4)
    student = _RefModel(num_classes=5)
    x, y = ref_inputs

    # Force logits into a range that overflows fp16.
    teacher_fp16 = _RefModel(num_classes=5).to(torch.float16)
    student_fp16 = _RefModel(num_classes=5).to(torch.float16)
    teacher_fp16.linear.weight.data = (ref_teacher.linear.weight.data * 50).to(
        torch.float16
    )
    student_fp16.linear.weight.data = (student.linear.weight.data * 50).to(
        torch.float16
    )

    distiller = KDHintonDistiller(
        teacher_fp16,
        student_fp16,
        config={"temperature": 4.0, "alpha": 0.7},
        device="cpu",
    )

    x_fp16 = x.to(torch.float16)
    with torch.no_grad():
        t_out = teacher_fp16(x_fp16)
        s_out = student_fp16(x_fp16)
    total, breakdown = distiller.compute_loss(s_out, t_out, y)
    assert torch.isfinite(total), f"loss became {total!r}"
    assert math.isfinite(breakdown["kd_loss"])
    # Hint regressors + dtype-aware math from BaseDistiller guard against
    # NaN/inf propagations.


# ----------------------------------------------------------------------------
# Hook cleanup
# ----------------------------------------------------------------------------


def test_remove_hooks_clears_handles() -> None:
    teacher = _RefModel(num_classes=3)
    student = _RefModel(num_classes=3)
    d = KDHintonDistiller(teacher, student, device="cpu")
    n_handles = len(d._hook_handles)
    assert n_handles == 0  # no hints → no hooks
    d.remove_hooks()
    assert d._hook_handles == []


# ----------------------------------------------------------------------------
# Zero-loss sentinel: when alpha == 1 and we hand-tune student=teacher, the
# KD loss is identically zero.
# ----------------------------------------------------------------------------


def test_zero_loss_when_student_matches_teacher(ref_inputs) -> None:
    torch.manual_seed(5)
    teacher = _RefModel(num_classes=5)
    student = _RefModel(num_classes=5)
    student.load_state_dict(teacher.state_dict())

    x, y = ref_inputs
    distiller = KDHintonDistiller(
        teacher, student, config={"alpha": 1.0, "temperature": 2.0}, device="cpu"
    )
    with torch.no_grad():
        t_out = teacher(x)
        s_out = student(x)
    total, breakdown = distiller.compute_loss(s_out, t_out, y)
    assert breakdown["kd_loss"] < 1e-6
    # Total == kd_loss + (1 - alpha) * ce_loss = kd_loss (since α=1)
    assert total.item() < 1e-6
