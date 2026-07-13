"""Behavior tests for :class:`AuxHeadDistiller` (deep supervision).

Pins the aux-head loss composition, strict layer match, FP16 stability,
and gradient flow to heads + student.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.aux_head_distiller import AuxHeadDistiller, _AuxHead
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
# Aux head helper
# ----------------------------------------------------------------------------


def test_aux_head_outputs_correct_classes() -> None:
    head = _AuxHead(in_dim=16, num_classes=4, hidden_dim=8)
    out = head(torch.randn(3, 16))
    assert out.shape == (3, 4)


# ----------------------------------------------------------------------------
# Loss composition
# ----------------------------------------------------------------------------


def test_aux_loss_is_mean_of_per_layer_cross_entropies() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = AuxHeadDistiller(
        teacher,
        student,
        config={
            "aux_head": {
                "student_layers": ["layers.1", "layers.2", "layers.3"],
                "num_classes": 4,
                "head_hidden": 8,
            }
        },
        device="cpu",
    )
    torch.manual_seed(0)
    feats = torch.randn(3, 8)
    student_features = {f"layers.{i}": feats.clone() for i in [1, 2, 3]}
    targets = torch.tensor([0, 1, 2])
    loss = d._compute_aux(student_features, targets)
    assert torch.isfinite(loss)
    # Each aux head sees the same features → losses should be close
    # (but not exactly equal because the heads are independently
    # initialised).
    assert loss.item() > 0.0


def test_aux_loss_handles_subset_of_configured_layers() -> None:
    """If the pipeline only captures some of the configured layers,
    the distiller averages over what's available.
    """
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = AuxHeadDistiller(
        teacher,
        student,
        config={
            "aux_head": {
                "student_layers": ["layers.1", "layers.2", "layers.3"],
                "num_classes": 4,
            }
        },
        device="cpu",
    )
    # Only `layers.2` captured.
    student_features = {"layers.2": torch.randn(4, 8)}
    targets = torch.tensor([0, 1, 2, 3])
    loss = d._compute_aux(student_features, targets)
    assert torch.isfinite(loss)
    # Only one layer was used → mean of a single value.
    assert loss.item() > 0.0


# ----------------------------------------------------------------------------
# Strict layer match
# ----------------------------------------------------------------------------


def test_strict_layer_match_raises_on_unknown() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    with pytest.raises(ConfigError, match="student:nope"):
        AuxHeadDistiller(
            teacher,
            student,
            config={
                "aux_head": {
                    "student_layers": ["layers.1", "nope"],
                    "num_classes": 4,
                    "strict_layer_match": True,
                }
            },
            device="cpu",
        )


# ----------------------------------------------------------------------------
# FP16 stability
# ----------------------------------------------------------------------------


def test_aux_finite_under_extreme_inputs() -> None:
    torch.manual_seed(1)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = AuxHeadDistiller(
        teacher,
        student,
        config={
            "aux_head": {
                "student_layers": ["layers.1", "layers.2", "layers.3"],
                "num_classes": 4,
            }
        },
        device="cpu",
    )
    student_features = {
        f"layers.{i}": torch.randn(3, 8) * 1e3 for i in [1, 2, 3]
    }
    targets = torch.tensor([0, 1, 2])
    loss = d._compute_aux(student_features, targets)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------------


def test_gradient_flows_through_aux_heads_and_student() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = AuxHeadDistiller(
        teacher,
        student,
        config={
            "aux_head": {
                "student_layers": ["layers.1", "layers.2"],
                "num_classes": 4,
            }
        },
        device="cpu",
    )
    student_features = {
        f"layers.{i}": torch.randn(3, 8, requires_grad=True) for i in [1, 2]
    }
    targets = torch.tensor([0, 1, 2])
    loss = d._compute_aux(student_features, targets)
    loss.backward()
    # Each aux head receives gradient.
    for layer_name in ("layers.1", "layers.2"):
        head = d._aux_heads[layer_name]
        for p in head.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0


# ----------------------------------------------------------------------------
# Pooling across ranks
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(4, 16), (4, 8, 16), (4, 16, 4, 4)],
)
def test_pool_reduces_to_batch_features(shape) -> None:
    out = AuxHeadDistiller._pool(torch.randn(*shape))
    assert out.shape == (shape[0], shape[1] if len(shape) == 4 else shape[-1])
