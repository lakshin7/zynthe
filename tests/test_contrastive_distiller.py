"""Behavior tests for :class:`ContrastiveDistiller` (CRD).

Pins the math (InfoNCE reference value), shape handling (3-D /
4-D / 2-D pooling), and the strict_layer_match behavior added in
Phase 0 (regression check).

References: Tian, Krishnan, Isola, "Contrastive Representation
Distillation", ICLR 2020.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.contrastive_distiller import ContrastiveDistiller
from zynthe.core.utils.exceptions import ConfigError


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _TinyHFMod(nn.Module):
    """Minimal module with an `embed` + a single `layers.3` Linear + a head."""

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


def _info_nce_reference(z_s: torch.Tensor, z_t: torch.Tensor, temperature: float) -> torch.Tensor:
    """Hand-evaluated InfoNCE used as the gold-standard reference for the
    contrastive loss.  z_s, z_t are L2-normalised projections.
    """
    B = z_s.shape[0]
    if B < 2:
        return torch.zeros((), device=z_s.device)
    sim_pos = (z_s * z_t).sum(dim=-1, keepdim=True)
    sim_neg = z_s @ z_t.T
    diag_mask = torch.eye(B, device=z_s.device, dtype=torch.bool)
    sim_neg = sim_neg.masked_fill(diag_mask, float("-inf"))
    logits = torch.cat([sim_pos, sim_neg], dim=1) / max(temperature, 1e-6)
    labels = torch.zeros(B, dtype=torch.long, device=z_s.device)
    return F.cross_entropy(logits, labels)


# ----------------------------------------------------------------------------
# Pooling
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(4, 16), (4, 8, 16), (4, 16, 4, 4)],
)
def test_pool_reduces_to_batch_features(shape) -> None:
    """_pool maps (B, C) / (B, T, C) / (B, C, H, W) -> (B, C)."""
    out = ContrastiveDistiller._pool(torch.randn(*shape))
    assert out.shape == (shape[0], shape[-1])


# ----------------------------------------------------------------------------
# Projection head — produces L2-normalised outputs
# ----------------------------------------------------------------------------


def test_projection_head_outputs_are_l2_normalised() -> None:
    from zynthe.core.distillers.contrastive_distiller import _ProjectionHead

    head = _ProjectionHead(in_dim=16, projection_dim=8)
    z = head(torch.randn(4, 16))
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


# ----------------------------------------------------------------------------
# InfoNCE reference value
# ----------------------------------------------------------------------------


def test_crd_matches_info_nce_reference() -> None:
    """Hand-evaluated InfoNCE matches the distiller's loss within atol."""
    torch.manual_seed(0)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ContrastiveDistiller(
        teacher,
        student,
        config={
            "contrastive": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
                "projection_dim": 16,
                "temperature": 0.1,
                "memory_bank_size": 0,
            }
        },
        device="cpu",
    )

    # Build a synthetic feature dict.
    torch.manual_seed(1)
    feats = torch.randn(4, 8)
    student_features = {"layers.3": feats.clone()}
    teacher_features = {"layers.3": feats.clone() + 0.1}

    # Reference using the distiller's own head path: forward through
    # both heads manually so we can pin the loss exactly.
    from zynthe.core.distillers.contrastive_distiller import _ProjectionHead

    head_s = _ProjectionHead(in_dim=8, projection_dim=16).to("cpu")
    head_t = _ProjectionHead(in_dim=8, projection_dim=16).to("cpu")
    z_s = head_s(student_features["layers.3"])
    z_t = head_t(teacher_features["layers.3"])
    expected = _info_nce_reference(z_s, z_t, temperature=0.1)

    # The distiller's own projection heads (lazy-created).
    d._student_head = head_s
    d._teacher_head = head_t
    actual = d._compute_crd(student_features, teacher_features)
    assert torch.allclose(actual, expected, atol=1e-5)


# ----------------------------------------------------------------------------
# 2-D feature path
# ----------------------------------------------------------------------------


def test_crd_runs_on_2d_features() -> None:
    """Features already (B, C) skip the pooling path and compute CRD."""
    torch.manual_seed(2)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ContrastiveDistiller(
        teacher,
        student,
        config={
            "contrastive": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
                "projection_dim": 16,
                "temperature": 0.07,
            }
        },
        device="cpu",
    )

    student_features = {"layers.3": torch.randn(3, 8)}
    teacher_features = {"layers.3": torch.randn(3, 8)}
    loss = d._compute_crd(student_features, teacher_features)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# Zero gradient on batch size 1 (degenerate case)
# ----------------------------------------------------------------------------


def test_crd_returns_zero_on_batch_size_1() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ContrastiveDistiller(
        teacher,
        student,
        config={
            "contrastive": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
                "projection_dim": 16,
            }
        },
        device="cpu",
    )
    student_features = {"layers.3": torch.randn(1, 8)}
    teacher_features = {"layers.3": torch.randn(1, 8)}
    loss = d._compute_crd(student_features, teacher_features)
    assert loss.item() == 0.0


# ----------------------------------------------------------------------------
# Projection heads are learnable, base backbone is frozen
# ----------------------------------------------------------------------------


def test_projection_heads_are_trainable_after_init() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ContrastiveDistiller(
        teacher,
        student,
        config={
            "contrastive": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
                "projection_dim": 16,
            }
        },
        device="cpu",
    )

    # Force head creation.
    student_features = {"layers.3": torch.randn(4, 8)}
    teacher_features = {"layers.3": torch.randn(4, 8)}
    d._compute_crd(student_features, teacher_features)

    assert d._student_head is not None
    assert d._teacher_head is not None
    # Student head params trainable.
    assert all(p.requires_grad for p in d._student_head.parameters())
    # Teacher head params also trainable (we explicitly override the
    # base class' teacher freeze for the projection head).
    assert all(p.requires_grad for p in d._teacher_head.parameters())
    # But the teacher's actual backbone is still frozen.
    for p in teacher.layers[0].parameters():
        assert not p.requires_grad


# ----------------------------------------------------------------------------
# Memory bank updates
# ----------------------------------------------------------------------------


def test_memory_bank_grows_and_evicts() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ContrastiveDistiller(
        teacher,
        student,
        config={
            "contrastive": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
                "projection_dim": 16,
                "memory_bank_size": 4,  # small bank
            }
        },
        device="cpu",
    )
    student_features = {"layers.3": torch.randn(2, 8)}
    teacher_features = {"layers.3": torch.randn(2, 8)}

    # Need heads first.
    d._compute_crd(student_features, teacher_features)
    bank = d._memory_bank
    assert bank is not None
    assert bank.shape[0] == 2

    # Add more — bank should grow but cap at memory_bank_size.
    for _ in range(5):
        d._update_memory_bank(torch.randn(2, 16))
    assert d._memory_bank.shape[0] == 4  # capped


# ----------------------------------------------------------------------------
# Strict layer match (Phase 0 regression check)
# ----------------------------------------------------------------------------


def test_strict_layer_match_raises_on_unknown() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    with pytest.raises(ConfigError, match="teacher:nope"):
        ContrastiveDistiller(
            teacher,
            student,
            config={
                "contrastive": {
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


def test_crd_is_finite_under_extreme_inputs() -> None:
    """Very large feature values still produce a finite scalar loss."""
    torch.manual_seed(3)
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ContrastiveDistiller(
        teacher,
        student,
        config={
            "contrastive": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
                "projection_dim": 16,
                "temperature": 0.07,
            }
        },
        device="cpu",
    )
    student_features = {"layers.3": torch.randn(4, 8) * 200}
    teacher_features = {"layers.3": torch.randn(4, 8) * 200}
    loss = d._compute_crd(student_features, teacher_features)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# Gradient flows through student head (only)
# ----------------------------------------------------------------------------


def test_gradient_flows_through_student_projection_only() -> None:
    teacher = _TinyHFMod(num_classes=4, hidden=8)
    student = _TinyHFMod(num_classes=4, hidden=8)
    d = ContrastiveDistiller(
        teacher,
        student,
        config={
            "contrastive": {
                "student_layers": ["layers.3"],
                "teacher_layers": ["layers.3"],
                "projection_dim": 16,
                "memory_bank_size": 0,
            }
        },
        device="cpu",
    )

    student_features = {"layers.3": torch.randn(4, 8, requires_grad=True)}
    teacher_features = {"layers.3": torch.randn(4, 8)}

    loss = d._compute_crd(student_features, teacher_features)
    loss.backward()

    # Student projection head's parameters received gradient.
    for p in d._student_head.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    # The student's underlying layer should NOT have gradient (teacher
    # features are detached inside the path).
    assert student_features["layers.3"].grad is None
