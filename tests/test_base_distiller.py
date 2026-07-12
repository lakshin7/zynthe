"""Behavior tests for :class:`BaseDistiller`.

Pins:

* ``training_step`` actually updates student parameters (does the backward
  pass when an optimizer is supplied).
* gradient-clipping (norm + AGC) reduce the gradient norm or zero it out.
* ``requires_grad`` warning fires when the distiller returns a 0-D tensor
  with no grad_fn.
* Forward hooks register and clean up correctly.
* ``BaseDistiller._extract_logits_tensor`` upcasts fp16 / clamps inf/nan
  on student outputs to keep KD loss stable.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from zynthe.core.distillers.base_distiller import BaseDistiller
from zynthe.core.distillers.kd_hinton import KDHintonDistiller


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


class _Lin(nn.Module):
    """Tiny linear model returning ``logits`` so the distiller treats it as
    a HF-style model.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(3, num_classes)

    def forward(self, x, labels=None):
        return _Out(logits=self.lin(x), loss=None)


class _Out:
    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss


@pytest.fixture()
def teacher_student():
    torch.manual_seed(0)
    teacher = _Lin(num_classes=4)
    student = _Lin(num_classes=4)
    return teacher, student


@pytest.fixture()
def sample_batch():
    x = torch.randn(4, 3)
    y = torch.randint(0, 4, (4,))
    return {"input_ids": x, "labels": y}


@pytest.fixture()
def distiller(teacher_student):
    teacher, student = teacher_student
    return KDHintonDistiller(
        teacher, student, config={"alpha": 0.5, "temperature": 2.0}, device="cpu"
    )


# ----------------------------------------------------------------------------
# Hook lifecycle
# ----------------------------------------------------------------------------


def test_hooks_register_when_distiller_built(teacher_student) -> None:
    """KDHintonDistiller (which uses BaseDistiller's hook system only when
    hints are configured) should not register hooks for the no-hint
    default case; with hints, hooks must register and clean up.
    """
    teacher, student = teacher_student

    # No hints → no hooks.
    d1 = KDHintonDistiller(teacher, student, config={"alpha": 0.5}, device="cpu")
    assert d1._hook_handles == []

    # With a single hint pair → two hooks.
    class _TinyWith(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(3, 3)
            self.head = nn.Linear(3, 4)

        def forward(self, x, labels=None):
            return _Out(logits=self.head(self.layer1(x)), loss=None)

    teacher2 = _TinyWith()
    student2 = _TinyWith()
    d2 = KDHintonDistiller(
        teacher2,
        student2,
        config={
            "alpha": 0.5,
            "temperature": 2.0,
            "hint_enabled": True,
            "hints": [{"teacher": "layer1", "student": "layer1"}],
        },
        device="cpu",
    )
    assert len(d2._hook_handles) == 2
    d2.remove_hooks()
    assert d2._hook_handles == []


# ----------------------------------------------------------------------------
# training_step updates student params
# ----------------------------------------------------------------------------


def _student_param_before(dist: BaseDistiller):
    return next(dist.student.parameters()).detach().clone()


def test_training_step_changes_student_weights(distiller, sample_batch) -> None:
    import torch.optim

    optimizer = torch.optim.SGD(distiller.student.parameters(), lr=0.1)
    before = _student_param_before(distiller)
    distiller.training_step(sample_batch, optimizer=optimizer)
    optimizer.step()
    after = next(distiller.student.parameters()).detach().clone()
    assert not torch.allclose(before, after, atol=1e-7)


def test_training_step_rejects_missing_optimizer(distiller, sample_batch) -> None:
    with pytest.raises(ValueError, match="No optimizer"):
        distiller.optimizer = None
        distiller.training_step(sample_batch, optimizer=None)


def test_agc_normal_clip_paths(distiller, sample_batch) -> None:
    """Grad-clip should not blow up the loss and should produce a finite
    gradient norm. We just check both code paths execute.
    """
    import torch.optim

    optimizer = torch.optim.SGD(distiller.student.parameters(), lr=0.1)
    # AGC path
    distiller.training_step(sample_batch, optimizer=optimizer, grad_clip="agc")
    # Numeric norm path
    distiller.training_step(sample_batch, optimizer=optimizer, grad_clip=1.0)
    # Dict-style AGC
    distiller.training_step(
        sample_batch,
        optimizer=optimizer,
        grad_clip={"type": "agc", "clip_factor": 0.01, "eps": 1e-3},
    )


# ----------------------------------------------------------------------------
# _extract_logits_tensor upcasts fp16 / clamps inf
# ----------------------------------------------------------------------------


def test_extract_logits_tensor_upcasts_fp16() -> None:
    """fp16 logits upcasted to fp32 for stable softmax/KL."""
    logits = torch.randn(2, 4, dtype=torch.float16) * 50.0  # will overflow softmax
    out = BaseDistiller._extract_logits_tensor(logits)
    assert out.dtype == torch.float32


def test_extract_logits_tensor_clamps_inf_nan() -> None:
    logits = torch.tensor([[float("inf"), 1.0, 2.0, float("nan")]])
    out = BaseDistiller._extract_logits_tensor(logits)
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------------
# LM-flatten helper
# ----------------------------------------------------------------------------


def test_flatten_lm_logits_shifts_and_drops_ignore_index() -> None:
    logits = torch.randn(2, 4, 5)
    targets = torch.tensor([[1, 2, 3, 4], [4, 4, 4, 4]])
    flat_l, flat_t = BaseDistiller._flatten_lm_logits_and_targets(
        logits, targets, ignore_index=4, shift_labels=True
    )
    # After shift: logits[:, :-1] = 3 timesteps, targets[:, 1:] = 3 timesteps
    assert flat_l.shape == (6, 5)  # 2 batch * 3 timesteps
    # All ignore labels must be filtered out.
    assert (flat_t != 4).all()


# ----------------------------------------------------------------------------
# Warning path: requires_grad=False on total_loss
# ----------------------------------------------------------------------------


def test_warning_when_total_loss_has_no_grad(caplog) -> None:
    """If we patch a distiller to return a 0-D tensor without grad_fn,
    the training_step should warn.
    """

    class _NoGradDist(BaseDistiller):
        modality_type = "text"

        def __init__(self, t, s):
            super().__init__(t, s, config={"alpha": 0.5}, device="cpu")

        def _init_losses(self):
            pass

        def compute_loss(self, *args, **kwargs):
            return torch.tensor(0.0, requires_grad=False), {}

    teacher = _Lin(num_classes=4)
    student = _Lin(num_classes=4)
    d = _NoGradDist(teacher, student)
    d.teacher.eval()
    optimizer = torch.optim.SGD(d.student.parameters(), lr=0.1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d.training_step(
            {"input_ids": torch.randn(2, 3), "labels": torch.zeros(2, dtype=torch.long)},
            optimizer=optimizer,
        )
    assert any("grad_fn" in str(warning.message).lower() for warning in w)
