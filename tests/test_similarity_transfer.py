"""Behavior tests for :class:`SimilarityTransfer`.

Pins:

* Cosine pairwise similarity matrix equals
  :math:`F_{\\mathrm{norm}} \\cdot F_{\\mathrm{norm}}^T` and is symmetric
  with diagonal 1.
* Euclidean distance similarity = ``exp(-d^2 / T)``.
* Loss when student matches teacher is ~0.
* Cross-modal similarity loss is symmetric.
* Progressive layer schedule returns the right number of layers per
  epoch.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.similarity_transfer import SimilarityTransfer


# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Stub(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.lin = nn.Linear(3, num_classes)

    def forward(self, input_ids=None, labels=None, **_unused):
        return _Out(logits=self.lin(input_ids))


# ----------------------------------------------------------------------------
# Pairwise similarity — closed form
# ----------------------------------------------------------------------------


@pytest.fixture()
def distiller() -> SimilarityTransfer:
    teacher = _Stub()
    student = _Stub()
    return SimilarityTransfer(
        teacher,
        student,
        config={
            "layer": -1,
            "similarity_metric": "cosine",
            "weight": 0.5,
            "normalize": True,
            "total_epochs": 5,
        },
        device="cpu",
    )


def test_cosine_pairwise_similarity_symmetric_with_unit_diag(distiller) -> None:
    torch.manual_seed(0)
    features = torch.randn(6, 16)
    sim = distiller.compute_similarity_matrix(features, metric="cosine")
    # cos sim from L2-normalized features.
    feats_norm = F.normalize(features, p=2, dim=-1)
    expected = feats_norm @ feats_norm.T
    assert torch.allclose(sim, expected, atol=1e-6)
    # Symmetric.
    assert torch.allclose(sim, sim.T, atol=1e-6)
    # Diagonal ≈ 1.
    assert torch.allclose(sim.diagonal(), torch.ones(6), atol=1e-5)


def test_euclidean_pairwise_similarity(distiller) -> None:
    torch.manual_seed(0)
    features = torch.randn(5, 8)
    sim = distiller.compute_similarity_matrix(features, metric="euclidean")
    # T is the distiller's temperature — default 1.0.
    d_sq = torch.cdist(features, features, p=2).pow(2)
    expected = torch.exp(-d_sq / distiller.temperature)
    assert torch.allclose(sim, expected, atol=1e-6)


def test_unknown_metric_raises_valueerror(distiller) -> None:
    with pytest.raises(ValueError, match="Unknown similarity metric"):
        distiller.compute_similarity_matrix(torch.randn(2, 3), metric="not_a_metric")


# ----------------------------------------------------------------------------
# Loss math
# ----------------------------------------------------------------------------


def test_similarity_loss_zero_when_student_matches_teacher(distiller) -> None:
    torch.manual_seed(0)
    f_t = torch.randn(4, 16)
    f_s = f_t.clone()
    loss = distiller.compute_similarity_loss(f_s, f_t)
    assert loss.item() < 1e-10


def test_cross_modality_loss_symmetric() -> None:
    """Cross-modal similarity loss: teacher/student matrices for two
    modalities are interchangeable.
    """
    torch.manual_seed(0)
    teacher_a = torch.randn(4, 8)
    teacher_b = torch.randn(4, 8)
    student_a = torch.randn(4, 8)
    student_b = torch.randn(4, 8)

    distiller = SimilarityTransfer(
        _Stub(),
        _Stub(),
        config={"layer": -1, "similarity_metric": "cosine"},
        device="cpu",
    )
    loss_forward = distiller.compute_cross_modality_loss(
        teacher_a, teacher_b, student_a, student_b
    )
    loss_backward = distiller.compute_cross_modality_loss(
        teacher_b, teacher_a, student_b, student_a
    )
    assert torch.allclose(loss_forward, loss_backward, atol=1e-6)


# ----------------------------------------------------------------------------
# Hidden-state shorthand parser
# ----------------------------------------------------------------------------


def test_infer_auto_layers_returns_hidden_shorthand(distiller) -> None:
    layers = distiller._infer_auto_layers("last", count=3)
    assert layers == ["hidden:-1", "hidden:-2", "hidden:-3"]


def test_infer_auto_layers_first_strategy(distiller) -> None:
    layers = distiller._infer_auto_layers("first", count=2)
    assert layers == ["hidden:0", "hidden:1"]


def test_infer_auto_layers_unknown_falls_back_to_last(distiller) -> None:
    layers = distiller._infer_auto_layers("not_a_strategy", count=2)
    # Falls back to 'last' (uses -N-1).
    assert layers == ["hidden:-1", "hidden:-2"]


# ----------------------------------------------------------------------------
# Progressive layer schedule
# ----------------------------------------------------------------------------


def test_progressive_schedule_adds_layers_per_epoch() -> None:
    """With progressive=True and progressive_epochs=2, the schedule
    returns 1 layer during the first 2 epochs, then 2 for epochs 3-4,
    then 3 thereafter (cap at total count).
    """
    config = {
        "similarity_transfer": {
            "layer": -1,
            "similarity_metric": "cosine",
            "progressive": True,
            "progressive_epochs": 2,
            "layers": ["a", "b", "c"],
            "total_epochs": 5,
        }
    }
    d = SimilarityTransfer(_Stub(), _Stub(), config=config, device="cpu")
    # Epoch 1: 1 layer.
    assert d.get_progressive_layers(1) == ["a"]
    # Epoch 2: 1 layer (within the first 2 epochs).
    assert d.get_progressive_layers(2) == ["a"]
    # Epoch 3: 2 layers.
    assert d.get_progressive_layers(3) == ["a", "b"]
    # Epoch 4: 2 layers (still within the 3-4 block).
    assert d.get_progressive_layers(4) == ["a", "b"]
    # Epoch 5: 3 layers.
    assert d.get_progressive_layers(5) == ["a", "b", "c"]


def test_progressive_disabled_returns_all_layers() -> None:
    config = {
        "similarity_transfer": {
            "layer": -1,
            "similarity_metric": "cosine",
            "progressive": False,
            "layers": ["a", "b", "c"],
        }
    }
    d = SimilarityTransfer(_Stub(), _Stub(), config=config, device="cpu")
    assert d.get_progressive_layers(1) == ["a", "b", "c"]
    assert d.get_progressive_layers(3) == ["a", "b", "c"]
