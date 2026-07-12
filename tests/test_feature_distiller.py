"""Behavior tests for :class:`FeatureDistiller`.

Pins the closed-form losses that live on
:class:`zynthe.core.distillers.feature_distiller.FeatureMetrics` (L2,
cosine, Gram, CKA, AB, contrastive), and the new (Phase 0) strict layer
match behavior that must raise :class:`ConfigError` when a configured
layer name is unknown.
"""

from __future__ import annotations

import math

import pytest
import torch

from zynthe.core.distillers.feature_distiller import (
    FeatureDistiller,
    FeatureLossComposer,
    FeatureMetrics,
)
from zynthe.core.utils.exceptions import ConfigError


# ----------------------------------------------------------------------------
# Helper: torch reference for each metric
# ----------------------------------------------------------------------------


def _ref_l2(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(f_s, f_t)


def _ref_cosine(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
    """Match :meth:`FeatureMetrics.cosine_similarity_loss`.

    The implementation flattens spatial dims then takes ``cosine_similarity``
    along the channel axis at every spatial position, yielding
    ``[B, H*W]``. A mean over spatial positions then batch is what it
    averages to compute the loss.
    """
    t = f_t.flatten(2)  # [B, C, H*W]
    s = f_s.flatten(2)
    cos = torch.nn.functional.cosine_similarity(t, s, dim=1).mean(dim=1)
    return (1.0 - cos).mean()


def _ref_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    feats = x.view(b, c, h * w)
    return torch.bmm(feats, feats.transpose(1, 2)) / (c * h * w)


def _ref_gram_loss(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(_ref_gram_matrix(f_s), _ref_gram_matrix(f_t))


def _ref_cka(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
    t = f_t.flatten(1)
    s = f_s.flatten(1)
    t_c = t - t.mean(dim=0, keepdim=True)
    s_c = s - s.mean(dim=0, keepdim=True)
    k_t = t_c @ t_c.T
    k_s = s_c @ s_c.T
    hsic_ts = (k_t * k_s).sum()
    hsic_tt = (k_t * k_t).sum()
    hsic_ss = (k_s * k_s).sum()
    return 1 - hsic_ts / (torch.sqrt(hsic_tt * hsic_ss) + 1e-10)


def _ref_contrastive(
    f_t: torch.Tensor, f_s: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    t = torch.nn.functional.normalize(f_t.flatten(2).mean(dim=2), dim=1)
    s = torch.nn.functional.normalize(f_s.flatten(2).mean(dim=2), dim=1)
    logits = torch.mm(s, t.T) / temperature
    labels = torch.arange(logits.size(0))
    return torch.nn.functional.cross_entropy(logits, labels)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture()
def fmap_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two 4-D feature maps of identical shape so we can test matching losses."""
    torch.manual_seed(0)
    f_t = torch.randn(4, 8, 6, 6)
    f_s = torch.randn(4, 8, 6, 6)
    return f_t, f_s


@pytest.fixture()
def stub_teacher_student():
    """A minimal teacher/student pair so FeatureDistiller can be constructed
    without doing anything heavyweight — it doesn't need to do a real forward
    pass for these tests; we call compute_loss directly.
    """
    import torch.nn as nn

    class _Stub(nn.Module):
        def forward(self, x, labels=None):
            return type("O", (), {"logits": torch.zeros(1, 2)})()

    return _Stub(), _Stub()


# ----------------------------------------------------------------------------
# Metric closed-form tests
# ----------------------------------------------------------------------------


def test_l2_metric_matches_reference(fmap_pair) -> None:
    f_t, f_s = fmap_pair
    expected = _ref_l2(f_t, f_s)
    actual = FeatureMetrics.l2_distance(f_t, f_s)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_cosine_metric_matches_reference(fmap_pair) -> None:
    f_t, f_s = fmap_pair
    expected = _ref_cosine(f_t, f_s)
    actual = FeatureMetrics.cosine_similarity_loss(f_t, f_s)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_gram_loss_matches_reference(fmap_pair) -> None:
    f_t, f_s = fmap_pair
    expected = _ref_gram_loss(f_t, f_s)
    actual = FeatureMetrics.gram_loss(f_t, f_s)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_cka_metric_matches_reference(fmap_pair) -> None:
    f_t, f_s = fmap_pair
    expected = _ref_cka(f_t, f_s)
    actual = FeatureMetrics.centered_kernel_alignment(f_t, f_s)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_cka_is_zero_when_student_matches_teacher(fmap_pair) -> None:
    f_t, _ = fmap_pair
    actual = FeatureMetrics.centered_kernel_alignment(f_t, f_t)
    assert actual.abs().item() < 1e-5


def test_contrastive_matches_reference(fmap_pair) -> None:
    f_t, f_s = fmap_pair
    expected = _ref_contrastive(f_t, f_s, temperature=0.07)
    actual = FeatureMetrics.contrastive_loss(f_t, f_s, temperature=0.07)
    # CE returns a 0-D scalar in both implementations.
    assert torch.allclose(actual, expected, atol=1e-4)


# ----------------------------------------------------------------------------
# FeatureLossComposer: weighted combination honors weights
# ----------------------------------------------------------------------------


def test_composer_combines_metrics_with_weights(fmap_pair) -> None:
    f_t, f_s = fmap_pair
    composer = FeatureLossComposer(
        metrics=["l2", "cosine"], weights={"l2": 0.7, "cosine": 0.3}
    )
    total, loss_dict = composer(f_t, f_s)
    expected = 0.7 * FeatureMetrics.l2_distance(f_t, f_s) + 0.3 * FeatureMetrics.cosine_similarity_loss(
        f_t, f_s
    )
    assert torch.allclose(total, expected, atol=1e-5)
    assert "feat_l2" in loss_dict
    assert "feat_cosine" in loss_dict
    assert "feat_total" in loss_dict
    assert loss_dict["feat_total"] == pytest.approx(total.item(), abs=1e-5)


def test_composer_skips_unknown_metric_with_warning(fmap_pair, caplog) -> None:
    from zynthe.core.distillers.feature_distiller import FeatureLossComposer

    f_t, f_s = fmap_pair
    composer = FeatureLossComposer(metrics=["l2", "totally_made_up"])
    with pytest.warns(UserWarning, match="Unknown metric"):
        total, _ = composer(f_t, f_s)
    assert torch.allclose(
        total, FeatureMetrics.l2_distance(f_t, f_s), atol=1e-6
    )


# ----------------------------------------------------------------------------
# Phase 0 fix: strict_layer_match raises ConfigError on missing layers
# ----------------------------------------------------------------------------


class _TinyMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def test_strict_layer_match_raises_on_missing_teacher(stub_teacher_student) -> None:
    teacher, student = stub_teacher_student
    config = {
        "feature_distillation": {
            "enabled": True,
            "strict_layer_match": True,
            "layers": [
                {"teacher": "encoder.layer.99", "student": "lin"},
            ],
        }
    }
    with pytest.raises(ConfigError) as exc:
        FeatureDistiller(teacher, student, config=config, device="cpu")
    assert "teacher:encoder.layer.99" in str(exc.value)


def test_strict_layer_match_raises_on_missing_student(stub_teacher_student) -> None:
    teacher, student = stub_teacher_student
    config = {
        "feature_distillation": {
            "enabled": True,
            "strict_layer_match": True,
            "layers": [
                {"teacher": "lin", "student": "nope.nope"},
            ],
        }
    }
    with pytest.raises(ConfigError) as exc:
        FeatureDistiller(teacher, student, config=config, device="cpu")
    assert "student:nope.nope" in str(exc.value)


def test_non_strict_layer_match_warns_skips(stub_teacher_student) -> None:
    teacher, student = stub_teacher_student
    config = {
        "feature_distillation": {
            "enabled": True,
            "layers": [
                {"teacher": "encoder.layer.99", "student": "lin"},
            ],
        }
    }
    with pytest.warns(UserWarning, match="skipping unmatched"):
        FeatureDistiller(teacher, student, config=config, device="cpu")
