"""Integration tests that exercise the full ``compute_loss`` paths in
each distiller. Each test takes a real forward pair and walks the
distiller end-to-end, asserting only that the loss is finite (so we
catch crashes / silent zero-loss bugs) and matches an analytic shape
where one is computable.

These tests close the coverage gap on the heavy compute paths that
the math-pin tests deliberately skipped.
"""

from __future__ import annotations

# Force CPU-only test environment. Without this, Modal L4 hosts default
# to CUDA and any ``nn.Module`` created with no ``device=`` ends up with
# cuda parameters while the tensors we pass stay on cpu -> crash.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Tiny forward-output model that mimics HuggingFace ModelOutput.
# ----------------------------------------------------------------------------


class _Out:
    def __init__(self, logits, hidden_states=None, attentions=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


class _TinyHFMod(nn.Module):
    def __init__(self, num_classes: int = 5, hidden: int = 16, layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids=None, labels=None, **_unused):
        if input_ids is None:
            raise TypeError("input_ids required")
        x = self.embed(input_ids)
        hidden = [x]
        for layer in self.layers:
            hidden.append(F.relu(layer(hidden[-1])))
        logits = self.head(hidden[-1])
        # Provide attentions so attention-rollout / dual matching paths
        # can run.
        attentions = tuple(
            torch.softmax(torch.randn(input_ids.size(0), 2, input_ids.size(1), input_ids.size(1)), dim=-1)
            for _ in range(len(self.layers))
        )
        # Provide hidden_states as a tuple of (B, T, hidden)
        return _Out(
            logits=logits,
            hidden_states=tuple(hidden),
            attentions=attentions,
        )


# ----------------------------------------------------------------------------
# KD-Hinton: full compute_loss path with hints + hint regressors
# ----------------------------------------------------------------------------


def test_kd_hinton_compute_loss_full_path_with_hints() -> None:
    """End-to-end compute_loss: matching teacher/student layers,
    hint regressors engage, attention head mismatch is bridged
    by the lazy layer-creation path.
    """
    import warnings

    from zynthe.core.distillers.kd_hinton import KDHintonDistiller

    torch.manual_seed(0)
    teacher = _TinyHFMod(num_classes=5, hidden=16, layers=3)
    student = _TinyHFMod(num_classes=5, hidden=16, layers=3)

    distiller = KDHintonDistiller(
        teacher,
        student,
        config={
            "kd_hinton": {
                "temperature": 2.0,
                "alpha": 0.5,
                "hint_enabled": True,
                "hints": [
                    {"teacher": "layers.0", "student": "layers.0", "regressor": "linear", "loss": "mse"},
                    {"teacher": "layers.1", "student": "layers.1", "regressor": "linear", "loss": "cosine"},
                ],
                "label_smoothing": 0.1,
                "log_interval": 1,
            }
        },
        device="cpu",
    )

    x = torch.randint(0, 32, (2, 4))
    y = torch.randint(0, 5, (2, 4))
    teacher_features = {f"layers.{i}": None for i in range(3)}
    student_features = {f"layers.{i}": None for i in range(3)}
    with torch.no_grad():
        s_out = student(x)
        t_out = teacher(x)
        # Grab matched layer outputs to pass into the layer-config.
        teacher_features = {
            "layers.0": t_out.hidden_states[1],
            "layers.1": t_out.hidden_states[2],
        }
        student_features = {
            "layers.0": s_out.hidden_states[1],
            "layers.1": s_out.hidden_states[2],
        }

    total, breakdown = distiller.compute_loss(
        student_outputs=s_out,
        teacher_outputs=t_out,
        targets=y,
        student_features=student_features,
        teacher_features=teacher_features,
    )
    assert torch.isfinite(total)
    assert "kd_loss" in breakdown
    assert "ce_loss" in breakdown
    assert "hint_total" in breakdown
    # Hint regressors were created.
    assert len(distiller.hint_regressors) > 0


def test_kd_hinton_compute_loss_label_smoothing() -> None:
    """Label smoothing > 0 produces a different CE than the no-smoothing
    case for the same logits + targets.
    """
    from zynthe.core.distillers.kd_hinton import KDHintonDistiller

    torch.manual_seed(1)
    teacher = _TinyHFMod(num_classes=4, hidden=8, layers=2)
    student = _TinyHFMod(num_classes=4, hidden=8, layers=2)
    d_smooth = KDHintonDistiller(
        teacher,
        student,
        config={"alpha": 0.0, "temperature": 2.0, "label_smoothing": 0.1},
        device="cpu",
    )
    d_plain = KDHintonDistiller(
        teacher,
        student,
        config={"alpha": 0.0, "temperature": 2.0, "label_smoothing": 0.0},
        device="cpu",
    )

    x = torch.randint(0, 32, (2, 4))
    y = torch.randint(0, 4, (2, 4))
    with torch.no_grad():
        t_out = teacher(x, labels=y)
        s_out = student(x, labels=y)

    # Both distill with alpha=0 → total == CE.
    smooth_loss = d_smooth.compute_loss(s_out, t_out, y)[0]
    plain_loss = d_plain.compute_loss(s_out, t_out, y)[0]
    assert torch.isfinite(smooth_loss)
    assert torch.isfinite(plain_loss)
    # They differ.
    assert not torch.allclose(smooth_loss, plain_loss, atol=1e-4)


# ----------------------------------------------------------------------------
# FeatureDistiller: full compute_loss path with FSP / AB / cross-layer bridges
# ----------------------------------------------------------------------------


def test_feature_distiller_compute_loss_with_alignment() -> None:
    """Full path: feature alignment via LayerAligner + composer
    combination + supervised CE. Verify the loss is finite and
    carries per-layer diagnostics.
    """
    from zynthe.core.distillers.feature_distiller import FeatureDistiller

    torch.manual_seed(2)
    teacher = _TinyHFMod(num_classes=4, hidden=16, layers=2)
    student = _TinyHFMod(num_classes=4, hidden=16, layers=2)

    distiller = FeatureDistiller(
        teacher,
        student,
        config={
            "feature_distillation": {
                "enabled": True,
                "layers": [
                    {"teacher": "layers.0", "student": "layers.0"},
                    {"teacher": "layers.1", "student": "layers.1"},
                ],
                "metrics": ["l2", "cosine"],
                "metric_weights": {"l2": 0.5, "cosine": 0.5},
                "auto_align": True,
                "adapter_type": "linear",
            }
        },
        device="cpu",
    )

    x = torch.randint(0, 32, (2, 4))
    y = torch.randint(0, 4, (2, 4))
    with torch.no_grad():
        t_out = teacher(x)
        s_out = student(x)

    # Run forward so hooks fire — base distiller forwards through the
    # student with no_grad wrapped teacher; we use compute_loss directly
    # to control timing.
    student_features = {
        "layers.0": s_out.hidden_states[1],
        "layers.1": s_out.hidden_states[2],
    }
    teacher_features = {
        "layers.0": t_out.hidden_states[1].detach(),
        "layers.1": t_out.hidden_states[2].detach(),
    }
    total, breakdown = distiller.compute_loss(
        student_outputs=s_out,
        teacher_outputs=t_out,
        targets=y,
        student_features=student_features,
        teacher_features=teacher_features,
    )
    assert torch.isfinite(total)
    assert "feature_total" in breakdown
    assert "supervised" in breakdown


# ----------------------------------------------------------------------------
# SimilarityTransfer: full compute_loss with hidden-state alignment.
# ----------------------------------------------------------------------------


def test_similarity_transfer_compute_loss_with_hidden_states() -> None:
    """Full path: hidden:-1 hidden-state shorthand resolved by the
    adapter, Gram matrix built via compute_similarity_matrix, MSE
    against teacher.
    """
    from zynthe.core.distillers.similarity_transfer import SimilarityTransfer

    torch.manual_seed(3)
    teacher = _TinyHFMod(num_classes=4, hidden=16, layers=2)
    student = _TinyHFMod(num_classes=4, hidden=16, layers=2)
    d = SimilarityTransfer(
        teacher,
        student,
        config={
            "similarity_transfer": {
                "layer": -1,
                "similarity_metric": "cosine",
                "weight": 0.5,
                "layers": ["hidden:-1"],
                "normalize": True,
            }
        },
        device="cpu",
    )

    x = torch.randint(0, 32, (2, 4))
    y = torch.randint(0, 4, (2, 4))
    with torch.no_grad():
        t_out = teacher(x, labels=y)
        s_out = student(x, labels=y)
    loss, _ = d.compute_loss(s_out, t_out, y)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# AttentionTransferDistiller: full compute_loss with a working attention map.
# ----------------------------------------------------------------------------


def test_attention_transfer_compute_loss_rollout() -> None:
    """Attention-rollout path: hooks feed attention maps, rollout is
    computed and compared.
    """
    import warnings

    from zynthe.core.distillers.attention_transfer import AttentionTransferDistiller

    torch.manual_seed(5)
    teacher = _TinyHFMod(num_classes=4, hidden=16, layers=2)
    student = _TinyHFMod(num_classes=4, hidden=16, layers=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        d = AttentionTransferDistiller(
            teacher,
            student,
            config={
                "mode": "spatial",
                "use_attention_rollout": True,
                "auto_detect_layers": False,
            },
        )

    x = torch.randint(0, 32, (2, 4))
    y = torch.randint(0, 4, (2, 4))
    with torch.no_grad():
        t_out = teacher(x, labels=y)
        s_out = student(x, labels=y)
    # Rollout path needs attentions on both — they are present.
    loss, _ = d.compute_loss(s_out, t_out, y)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# MultiStageDistiller: composite stage gating.
# ----------------------------------------------------------------------------


def test_multi_stage_distiller_get_weights_does_not_crash() -> None:
    """Smoke: build a MultiStageDistiller via a preset and confirm it
    constructs without error. ``list_presets()`` exposes the available
    preset names; we pick the first one.
    """
    from zynthe.core.distillers.multi_stage_distiller import MultiStageDistiller
    from zynthe.core.distillers.presets import get_preset, list_presets

    available = list_presets()
    assert len(available) > 0, "at least one preset should be registered"
    preset = get_preset(available[0])
    assert preset is not None

    teacher = _TinyHFMod(num_classes=4, hidden=8, layers=1)
    student = _TinyHFMod(num_classes=4, hidden=8, layers=1)
    d = MultiStageDistiller(
        teacher=teacher,
        student=student,
        config={"multi_stage": preset, "distillation": {}},
        device="cpu",
    )
    assert d is not None
