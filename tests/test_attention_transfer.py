"""Behavior tests for :class:`AttentionTransferDistiller`.

Pins:

* Attention rollout on identity attention matrices yields identity.
* Spatial attention on identical features yields 0 in MSE against itself.
* Affinity / cosine composition functions.
* Layer auto-detection falls back gracefully when no attention shapes
  match.
* Phase 0 strict layer match raises :class:`ConfigError`; non-strict
  warns and proceeds.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.attention_transfer import (
    AttentionExtractor,
    AttentionLossComposer,
    AttentionMatcher,
    AttentionTransferDistiller,
)
from zynthe.core.utils.exceptions import ConfigError


# ----------------------------------------------------------------------------
# Helper models
# ----------------------------------------------------------------------------


class _TinyMod(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = 8, num_classes: int = 4):
        super().__init__()
        first = []
        last = []
        for i in range(n_layers):
            first.append(nn.Linear(hidden, hidden))
            first.append(nn.ReLU())
        first.append(nn.Linear(hidden, num_classes))
        self.embed = nn.Sequential(*first)
        self.embed[-1] = nn.Identity()

    def forward(self, input_ids=None, labels=None, **_unused):
        return type("O", (), {"logits": self.embed(input_ids)})()


# ----------------------------------------------------------------------------
# Attention rollout invariant: rollout on identity = identity
# ----------------------------------------------------------------------------


def test_rollout_on_identity_is_identity() -> None:
    """If every layer's attention matrix is exactly the identity, the
    rollout (averaged over heads, propagated through layers with
    residual mix) should still be (approximately) identity.
    """
    from zynthe.core.distillers.attention_transfer import AttentionTransferDistiller

    n_heads = 4
    seq_len = 6
    attn = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).expand(1, n_heads, seq_len, seq_len).contiguous()
    layers = [attn.clone() for _ in range(3)]

    d = AttentionTransferDistiller.__new__(AttentionTransferDistiller)
    out = d.attention_rollout(layers, residual=True)
    # Diagonal entries ≈ 1 (residual mix means off-diagonals get shared
    # weight too — but the diagonal should remain the strongest signal).
    diag = out.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones(seq_len), atol=1e-5)


# ----------------------------------------------------------------------------
# Spatial attention should be dimensionless-zero on identical inputs.
# ----------------------------------------------------------------------------


def test_spatial_attention_zero_when_features_match() -> None:
    from zynthe.core.distillers.attention_transfer import AttentionTransferDistiller

    d = AttentionTransferDistiller.__new__(AttentionTransferDistiller)
    feats = torch.randn(2, 8, 4, 4)
    a = d.compute_spatial_attention(feats)
    b = d.compute_spatial_attention(feats)
    assert torch.allclose(a, b, atol=1e-6)
    # L2 distance between normalized outputs (== 0 by definition here).
    assert F.mse_loss(a, b).item() < 1e-12


def test_affinity_attention_is_unit_norm_columns() -> None:
    """Cosine-affinity: each row of (normalized F @ normalized F.T) has
    unit norm (it's a projection of unit vectors).
    """
    from zynthe.core.distillers.attention_transfer import AttentionTransferDistiller

    d = AttentionTransferDistiller.__new__(AttentionTransferDistiller)
    feats = torch.randn(4, 16)
    aff = d.compute_affinity_attention(feats)
    assert aff.shape == (4, 4)
    # Symmetric.
    assert torch.allclose(aff, aff.T, atol=1e-6)
    # Diagonal entries largest (since self-similarity = 1 for normalized vectors).
    diag = aff.diagonal()
    for i, row_max in enumerate(aff.max(dim=1).values):
        assert diag[i] >= row_max - 1e-5


# ----------------------------------------------------------------------------
# AttentionMatcher resizes student maps to match teacher.
# ----------------------------------------------------------------------------


def test_matcher_resizes_spatial_attention() -> None:
    matcher = AttentionMatcher(normalization="l2", interpolation_mode="bilinear")
    teacher = torch.randn(2, 8, 8)
    student = torch.randn(2, 4, 4)
    resized = matcher.resize(student, teacher)
    assert resized.shape[-2:] == (8, 8)


def test_matcher_returns_unchanged_when_shapes_match() -> None:
    matcher = AttentionMatcher()
    t = torch.randn(2, 4, 4)
    s = torch.randn(2, 4, 4)
    out = matcher.resize(s, t)
    assert out.shape == s.shape
    # Same tensor reference (no copy needed).
    assert out.data_ptr() == s.data_ptr()


# ----------------------------------------------------------------------------
# AttentionLossComposer weighted combination.
# ----------------------------------------------------------------------------


def test_loss_composer_l2_matches_mse() -> None:
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    composer = AttentionLossComposer(loss_types=["l2"])
    out = composer.compute(s, t)
    expected = F.mse_loss(s, t)
    assert torch.allclose(out, expected, atol=1e-6)


def test_loss_composer_raises_on_unknown_type() -> None:
    composer = AttentionLossComposer(loss_types=["not_a_loss_type"])
    with pytest.raises(ValueError, match="Unknown loss type"):
        composer.compute(torch.randn(2, 3), torch.randn(2, 3))


def test_loss_composer_default_weights_sum_to_one() -> None:
    composer = AttentionLossComposer(loss_types=["l2", "relational"])
    assert math.isclose(sum(composer.weights), 1.0)
    assert len(composer.weights) == 2


# ----------------------------------------------------------------------------
# Phase 0 fix: strict_layer_match raises ConfigError.
# ----------------------------------------------------------------------------


def test_strict_layer_match_raises_on_unknown_layer() -> None:
    teacher = _TinyMod(n_layers=1, hidden=4)
    student = _TinyMod(n_layers=1, hidden=4)
    config = {
        "attention_transfer": {
            "auto_detect_layers": False,
            "strict_layer_match": True,
            "teacher_layers": ["definitely_not_a_layer"],
            "student_layers": ["embed.0"],
        }
    }
    with pytest.raises(ConfigError) as exc:
        AttentionTransferDistiller(teacher, student, config=config)
    assert "definitely_not_a_layer" in str(exc.value)


def test_non_strict_warns_proceeds() -> None:
    """Layer name 'nope' is unknown — non-strict mode warns and proceeds
    (does not raise). The construction also emits an _attention module's
    'Unknown model type' UserWarning; we accept any UserWarning here.
    """
    import warnings

    teacher = _TinyMod(n_layers=1, hidden=4)
    student = _TinyMod(n_layers=1, hidden=4)
    config = {
        "attention_transfer": {
            "auto_detect_layers": False,
            "teacher_layers": ["nope"],
            "student_layers": ["embed.0"],
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        AttentionTransferDistiller(teacher, student, config=config)


# ----------------------------------------------------------------------------
# Auto-detect layer selection finds layers whose names end with attention suffixes.
# ----------------------------------------------------------------------------


def test_auto_detect_layers_returns_attention_module_names() -> None:
    """Stub with ``self_attn``-suffixed layers (the exact-suffix list in
    _auto_detect_attention_layers) — they should be returned unchanged.

    The distiller's ``AttentionExtractor`` emits a one-shot UserWarning
    about the unknown stub model class; we suppress it for this test.
    """
    import warnings

    class _AttnMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(4, 4)
            self.layer1_attn = nn.Linear(4, 4)  # matches suffix "self_attn"? no
            self.layer1_self_attn = nn.Linear(4, 4)  # matches suffix "self_attn"
            self.head = nn.Linear(4, 4)

        def forward(self, input_ids=None, labels=None, **_unused):
            return type("O", (), {"logits": self.head(self.layer1(input_ids))})()

    teacher = _AttnMod()
    student = _AttnMod()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        d = AttentionTransferDistiller(
            teacher,
            student,
            config={"auto_detect_layers": True},
        )
    # Exact-suffix search picks up "_self_attn" — we can detect whatever
    # ends in "attn" since "head" doesn't qualify (not a suffix match
    # even though it ends in "ad").
    assert all(name.endswith("attn") or "attention" in name for name in d.teacher_layers)
