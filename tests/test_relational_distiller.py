"""Behavior tests for :class:`RelationalDistiller` (PKT).

Reference: Park et al. 2019, "Relational Knowledge Distillation", CVPR.

The PKT loss is a Frobenius distance between the pairwise cosine-similarity
matrices of student and teacher features. We pin the closed-form,
shape handling, strict layer match, and FP16 stability.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.relational_distiller import (
    RelationalDistiller,
    _pairwise_cosine,
)
from zynthe.core.utils.exceptions import ConfigError


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _TinyHFMod(nn.Module):
    def __init__(self, num_classes: int = 4, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(4)]
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids=None, labels=None, **_unused):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = F.relu(layer(x))
        return _Out(logits=self.head(x))


# ----------------------------------------------------------------------------
# Pairwise cosine helper
# ----------------------------------------------------------------------------


def test_pairwise_cosine_diagonal_is_one() -> None:
    torch.manual_seed(0)
    f = torch.randn(4, 8)
    m = _pairwise_cosine(f)
    assert m.shape == (4, 4)
    assert torch.allclose(m.diagonal(), torch.ones(4), atol=1e-5)


def test_pairwise_cosine_is_symmetric() -> None:
    torch.manual_seed(1)
    f = torch.randn(6, 16)
    m = _pairwise_cosine(f)
    assert torch.allclose(m, m.T, atol=1e-6)


# ----------------------------------------------------------------------------
# Closed-form loss
# ----------------------------------------------------------------------------


def test_pkt_matches_closed_form_when_features_align() -> None:
    """When student and teacher features are identical, PKT loss == 0."""
    torch.manual_seed(2)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = RelationalDistiller(
        teacher,
        student,
        config={
            "relational": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    feats = torch.randn(5, 8)
    student_features = {"layers.3": feats.clone()}
    teacher_features = {"layers.3": feats.clone()}
    loss = d._compute_pair(student_features, teacher_features)
    assert loss.item() < 1e-9


def test_pkt_matches_mse_against_hand_eval() -> None:
    """For different student / teacher features, the loss equals
    ``F.mse_loss(cos(s), cos(t))``.
    """
    torch.manual_seed(3)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = RelationalDistiller(
        teacher,
        student,
        config={
            "relational": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    s = torch.randn(6, 8)
    t = torch.randn(6, 8) + 1.0
    student_features = {"layers.3": s}
    teacher_features = {"layers.3": t}

    actual = d._compute_pair(student_features, teacher_features)
    expected = F.mse_loss(_pairwise_cosine(s), _pairwise_cosine(t.detach()))
    assert torch.allclose(actual, expected, atol=1e-6)


# ----------------------------------------------------------------------------
# Strict layer match
# ----------------------------------------------------------------------------


def test_strict_layer_match_raises_on_unknown() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    with pytest.raises(ConfigError, match="teacher:nope"):
        RelationalDistiller(
            teacher,
            student,
            config={
                "relational": {
                    "student_layers": ["layers.3"],
                    "teacher_layers": ["nope"],
                    "strict_layer_match": True,
                }
            },
            device="cpu",
        )


# ----------------------------------------------------------------------------
# FP16 stability
# ----------------------------------------------------------------------------


def test_pkt_finite_under_extreme_inputs() -> None:
    torch.manual_seed(4)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = RelationalDistiller(
        teacher,
        student,
        config={
            "relational": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    student_features = {"layers.3": torch.randn(4, 8) * 1e3}
    teacher_features = {"layers.3": torch.randn(4, 8) * 1e3}
    loss = d._compute_pair(student_features, teacher_features)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# Different feature widths
# ----------------------------------------------------------------------------


def test_pkt_handles_mismatched_feature_widths() -> None:
    """If teacher and student have different last-dim, the distiller
    truncates to the shorter — PKT only needs an inner-product.
    """
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = RelationalDistiller(
        teacher,
        student,
        config={
            "relational": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    student_features = {"layers.3": torch.randn(4, 16)}  # double width
    teacher_features = {"layers.3": torch.randn(4, 8)}
    loss = d._compute_pair(student_features, teacher_features)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# Pooling across ranks
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(4, 16), (4, 8, 16), (4, 16, 4, 4)],
)
def test_pool_reduces_to_batch_features(shape) -> None:
    out = RelationalDistiller._pool(torch.randn(*shape))
    assert out.shape == (shape[0], shape[1] if len(shape) == 4 else shape[-1])
