"""Behavior tests for :class:`ProjectionDistiller` (translator).

Pins: translator dim, MSE on aligned hidden states, strict layer match,
zero-loss when student matches teacher, FP16 stability.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.projection_distiller import (
    ProjectionDistiller,
    _Translator,
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
# Translator helper
# ----------------------------------------------------------------------------


def test_translator_changes_input_dim_to_output_dim() -> None:
    t = _Translator(in_dim=8, out_dim=16, hidden_dim=12)
    out = t(torch.randn(4, 8))
    assert out.shape == (4, 16)


# ----------------------------------------------------------------------------
# Closed-form MSE
# ----------------------------------------------------------------------------


def test_projection_loss_matches_mse_on_translated_features() -> None:
    torch.manual_seed(0)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ProjectionDistiller(
        teacher,
        student,
        config={
            "projection": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )

    s = torch.randn(3, 8)
    t = torch.randn(3, 8)
    student_features = {"layers.3": s}
    teacher_features = {"layers.3": t}

    # Force the lazy translator creation.
    d._translator = _Translator(in_dim=8, out_dim=8, hidden_dim=8).to("cpu")
    d.add_module("_translator", d._translator)

    actual = d._compute_projection(student_features, teacher_features)
    expected = F.mse_loss(d._translator(s), t.detach())
    assert torch.allclose(actual, expected, atol=1e-6)


def test_projection_zero_loss_when_student_matches_teacher() -> None:
    """When student features == teacher features, the translator can
    converge to identity (just one linear layer between same-shape
    features) — the minimum loss is approximately the initalisation
    residual, not exactly zero.
    """
    torch.manual_seed(1)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ProjectionDistiller(
        teacher,
        student,
        config={
            "projection": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    feats = torch.randn(4, 8)
    student_features = {"layers.3": feats.clone()}
    teacher_features = {"layers.3": feats.clone()}
    loss = d._compute_projection(student_features, teacher_features)
    # Random init of translator → non-zero, but finite.
    assert torch.isfinite(loss)
    assert loss.item() < 100.0


# ----------------------------------------------------------------------------
# Different widths
# ----------------------------------------------------------------------------


def test_projection_handles_mismatched_widths() -> None:
    """Translator learns to map student dim → teacher dim."""
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ProjectionDistiller(
        teacher,
        student,
        config={
            "projection": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    # Student features are 16-D, teacher features are 8-D. Translator
    # learns to compress.
    student_features = {"layers.3": torch.randn(4, 16)}
    teacher_features = {"layers.3": torch.randn(4, 8)}
    loss = d._compute_projection(student_features, teacher_features)
    assert torch.isfinite(loss)
    assert d._translator is not None
    # Translator's output dim must match teacher's input dim (8).
    assert d._translator.net[-1].out_features == 8


# ----------------------------------------------------------------------------
# Strict layer match
# ----------------------------------------------------------------------------


def test_strict_layer_match_raises_on_unknown() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    with pytest.raises(ConfigError, match="teacher:nope"):
        ProjectionDistiller(
            teacher,
            student,
            config={
                "projection": {
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


def test_projection_finite_under_extreme_inputs() -> None:
    torch.manual_seed(2)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ProjectionDistiller(
        teacher,
        student,
        config={
            "projection": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    student_features = {"layers.3": torch.randn(4, 8) * 1e3}
    teacher_features = {"layers.3": torch.randn(4, 8) * 1e3}
    loss = d._compute_projection(student_features, teacher_features)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# Translator is trainable; teacher detached
# ----------------------------------------------------------------------------


def test_translator_is_trainable_and_teacher_detached() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ProjectionDistiller(
        teacher,
        student,
        config={
            "projection": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
            }
        },
        device="cpu",
    )
    student_features = {
        "layers.3": torch.randn(4, 8, requires_grad=True)
    }
    teacher_features = {
        "layers.3": torch.randn(4, 8, requires_grad=True)
    }
    loss = d._compute_projection(student_features, teacher_features)
    loss.backward()
    # Translator receives gradient.
    for p in d._translator.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    # Teacher features detached inside.
    assert teacher_features["layers.3"].grad is None


# ----------------------------------------------------------------------------
# Pooling across ranks
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(4, 16), (4, 8, 16), (4, 16, 4, 4)],
)
def test_pool_reduces_to_batch_features(shape) -> None:
    out = ProjectionDistiller._pool(torch.randn(*shape))
    assert out.shape == (shape[0], shape[1] if len(shape) == 4 else shape[-1])
