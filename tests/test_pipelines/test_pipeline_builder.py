"""Behavior tests for :class:`PipelineBuilder`.

Pins:

* ``add_distiller`` without ``add_stage`` auto-creates a single stage.
* 1 stage + 1 distiller ⇒ :class:`SingleDistillerPipeline`.
* 2+ stages (or 2+ distillers in one stage) ⇒
  :class:`MultiStagePipeline`.
* ``MultiStagePipeline.setup`` normalizes stage weights when
  ``normalize_weights=True``.
* ``from_config`` accepts both legacy (single distiller) and the new
  pipeline-shape (``stages:[...]``) configs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from zynthe.core.distillers.kd_hinton import KDHintonDistiller
from zynthe.core.distillers.similarity_transfer import SimilarityTransfer
from zynthe.core.pipelines import PipelineBuilder
from zynthe.core.pipelines.multi_stage_pipeline import MultiStagePipeline
from zynthe.core.pipelines.single_distiller_pipeline import SingleDistillerPipeline
from zynthe.core.distillers.multi_stage_distiller import DistillerRegistry


# ----------------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------------


class _Stub(nn.Module):
    def forward(self, input_ids=None, labels=None, **_unused):
        return type("O", (), {"logits": torch.zeros(1, 2)})()


@pytest.fixture()
def teacher_student():
    return _Stub(), _Stub()


# ----------------------------------------------------------------------------
# Routing — 1 vs N distillers
# ----------------------------------------------------------------------------


def test_builder_one_distiller_returns_single_pipeline(teacher_student) -> None:
    teacher, student = teacher_student
    pipeline = (
        PipelineBuilder()
        .add_distiller("kd_hinton", temperature=2.0)
        .build(teacher, student, device="cpu")
    )
    assert isinstance(pipeline, SingleDistillerPipeline)


def test_builder_two_stages_returns_multi_pipeline(teacher_student) -> None:
    teacher, student = teacher_student

    pipeline = (
        PipelineBuilder()
        .add_stage("logit", weight=0.6)
        .add_distiller("kd_hinton", temperature=2.0)
        .add_stage("feature", weight=0.4)
        .add_distiller("feature")
        .build(teacher, student, device="cpu")
    )
    assert isinstance(pipeline, MultiStagePipeline)


def test_builder_two_distillers_one_stage_returns_multi_pipeline(teacher_student) -> None:
    teacher, student = teacher_student
    pipeline = (
        PipelineBuilder()
        .add_distiller("kd_hinton", temperature=2.0)
        .add_distiller("feature")
        .build(teacher, student, device="cpu")
    )
    assert isinstance(pipeline, MultiStagePipeline)


# ----------------------------------------------------------------------------
# Distiller registry
# ----------------------------------------------------------------------------


def test_distiller_registry_can_resolve_known_ids() -> None:
    reg = DistillerRegistry()
    assert reg.get("kd_hinton") is KDHintonDistiller
    assert reg.get("feature") is not None
    assert reg.get("similarity") is SimilarityTransfer


def test_distiller_registry_unknown_raises_valueerror() -> None:
    reg = DistillerRegistry()
    with pytest.raises(ValueError, match="Unknown distiller"):
        reg.get("not_a_distiller_v0_2_6")


# ----------------------------------------------------------------------------
# Multi-stage weight normalization
# ----------------------------------------------------------------------------


def test_multi_stage_normalizes_weights(teacher_student) -> None:
    teacher, student = teacher_student
    pipeline = (
        PipelineBuilder()
        .add_stage("a", weight=0.7)
        .add_distiller("kd_hinton", temperature=2.0)
        .add_stage("b", weight=0.3)
        .add_distiller("feature")
        .with_mode("parallel")
        .build(teacher, student, device="cpu")
    )
    # Call setup explicitly to trigger normalization (SingleDistillerPipeline
    # auto-setup happens on first __call__; MultiStagePipeline does the same).
    pipeline.setup()
    weights = [stage.weight for stage in pipeline.stages]
    total = sum(weights)
    assert abs(total - 1.0) < 1e-6
    # Relative ratios preserved.
    assert abs(weights[0] / weights[1] - 0.7 / 0.3) < 1e-6


def test_multi_stage_normalize_disabled(teacher_student) -> None:
    teacher, student = teacher_student
    pipeline = (
        PipelineBuilder()
        .normalize_weights(False)
        .add_stage("a", weight=0.7)
        .add_distiller("kd_hinton", temperature=2.0)
        .add_stage("b", weight=0.3)
        .add_distiller("feature")
        .with_mode("parallel")
        .build(teacher, student, device="cpu")
    )
    pipeline.setup()
    weights = [stage.weight for stage in pipeline.stages]
    assert abs(weights[0] - 0.7) < 1e-6
    assert abs(weights[1] - 0.3) < 1e-6


# ----------------------------------------------------------------------------
# from_config — both legacy and new formats
# ----------------------------------------------------------------------------


def test_from_config_legacy_single_distiller(teacher_student) -> None:
    teacher, student = teacher_student
    config = {
        "distillation": {
            "method": "kd_hinton",
            "config": {"temperature": 2.0, "alpha": 0.7},
        }
    }
    pipeline = PipelineBuilder.from_config(config, teacher, student, device="cpu")
    assert isinstance(pipeline, SingleDistillerPipeline)


def test_from_config_new_pipeline_shape(teacher_student) -> None:
    teacher, student = teacher_student
    config = {
        "distillation": {
            "pipeline": {
                "type": "multi_stage",
                "mode": "sequential",
                "stages": [
                    {
                        "name": "logit_stage",
                        "weight": 0.6,
                        "distillers": [
                            {
                                "type": "kd_hinton",
                                "config": {"temperature": 2.0},
                            }
                        ],
                    },
                    {
                        "name": "feature_stage",
                        "weight": 0.4,
                        "distillers": [
                            {"type": "feature", "config": {}},
                        ],
                    },
                ],
            }
        }
    }
    pipeline = PipelineBuilder.from_config(config, teacher, student, device="cpu")
    assert isinstance(pipeline, MultiStagePipeline)
    assert len(pipeline.stages) == 2
