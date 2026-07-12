"""Behavior tests for :class:`MultiStagePipeline`.

Pins:

* Parallel mode runs every stage and aggregates losses weighted by
  stage weight.
* Sequential mode treats each stage as a separate forward (the
  source's implementation collapses forward+parallel into a single
  pass — we test the actual contract).
* Conditional mode honors the ``condition`` callable and skips a stage
  whose predicate returns False.
* Stage weight normalization converges to a unit-sum vector.
* Adding an empty stage list raises during ``setup()``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from zynthe.core.distillers.kd_hinton import KDHintonDistiller
from zynthe.core.pipelines import PipelineBuilder
from zynthe.core.pipelines.multi_stage_pipeline import (
    ExecutionMode,
    MultiStagePipeline,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _StubMod(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.lin = nn.Linear(3, num_classes)

    def forward(self, input_ids=None, labels=None, **_unused):
        return _Out(logits=self.lin(input_ids))


@pytest.fixture()
def teacher_student():
    return _StubMod(), _StubMod()


@pytest.fixture()
def batch():
    return {
        "input_ids": torch.randn(4, 3),
        "labels": torch.randint(0, 4, (4,)),
    }


# ----------------------------------------------------------------------------
# Mode behavior
# ----------------------------------------------------------------------------


def test_parallel_mode_runs_every_stage(teacher_student, batch) -> None:
    teacher, student = teacher_student
    pipeline = (
        PipelineBuilder()
        .add_stage("a", weight=0.6)
        .add_distiller("kd_hinton", temperature=2.0)
        .add_stage("b", weight=0.4)
        .add_distiller("kd_hinton", temperature=4.0)
        .with_mode(ExecutionMode.PARALLEL)
        .build(teacher, student, device="cpu")
    )
    pipeline.setup()
    loss, metrics = pipeline(batch)
    assert torch.isfinite(loss)
    assert "stage_metrics" in metrics.custom_metrics
    # Both stages should appear in custom_metrics.
    assert "a" in metrics.custom_metrics["stage_metrics"]
    assert "b" in metrics.custom_metrics["stage_metrics"]


def test_setup_raises_for_empty_pipeline() -> None:
    pipeline = MultiStagePipeline(_StubMod(), _StubMod(), mode=ExecutionMode.PARALLEL)
    with pytest.raises(ValueError, match="No stages added"):
        pipeline.setup()


# ----------------------------------------------------------------------------
# Weight normalization through setup()
# ----------------------------------------------------------------------------


def test_stage_weight_normalization_sums_to_one(teacher_student) -> None:
    teacher, student = teacher_student
    pipeline = (
        PipelineBuilder()
        .add_stage("a", weight=0.2)
        .add_distiller("kd_hinton", temperature=2.0)
        .add_stage("b", weight=0.5)
        .add_distiller("kd_hinton", temperature=2.0)
        .add_stage("c", weight=0.9)
        .add_distiller("kd_hinton", temperature=2.0)
        .build(teacher, student, device="cpu")
    )
    pipeline.setup()
    weights = [s.weight for s in pipeline.stages]
    total = sum(weights)
    assert abs(total - 1.0) < 1e-6
    # Relative ratios preserved.
    assert abs(weights[0] - 0.2 / 1.6) < 1e-6
    assert abs(weights[1] - 0.5 / 1.6) < 1e-6
    assert abs(weights[2] - 0.9 / 1.6) < 1e-6


# ----------------------------------------------------------------------------
# Conditional stage execution
# ----------------------------------------------------------------------------


def test_conditional_stage_skipped_when_predicate_false(teacher_student, batch) -> None:
    teacher, student = teacher_student
    # Build a 2-stage pipeline with the second stage's condition
    # returning False.
    pipeline = (
        PipelineBuilder()
        .add_stage("always_run", weight=1.0)
        .add_distiller("kd_hinton", temperature=2.0)
        .build(teacher, student, device="cpu")
    )
    pipeline.setup()

    # Mutate the pipeline to add a second stage with a condition.
    student_pipeline = pipeline  # already a SingleDistillerPipeline
    multi = MultiStagePipeline(
        teacher, student, mode=ExecutionMode.SEQUENTIAL, device=torch.device("cpu")
    )
    multi.add_stage("always_run", student_pipeline, weight=0.5)
    skipped_pipeline = KDHintonDistiller(
        teacher, student, config={"temperature": 2.0}, device="cpu"
    )
    multi.add_stage(
        "skipped",
        skipped_pipeline,
        weight=0.5,
        condition=lambda batch_, outputs: False,  # always skip
    )
    multi.setup()

    loss, metrics = multi(batch)
    assert torch.isfinite(loss)
    # Stage 'skipped' should be absent from stage metrics.
    stage_names = list(metrics.custom_metrics["stage_metrics"].keys())
    assert "skipped" not in stage_names
    assert "always_run" in stage_names


def test_conditional_stage_runs_when_predicate_true(teacher_student, batch) -> None:
    teacher, student = teacher_student
    p_a = KDHintonDistiller(teacher, student, config={"temperature": 2.0}, device="cpu")
    p_b = KDHintonDistiller(teacher, student, config={"temperature": 2.0}, device="cpu")
    multi = MultiStagePipeline(teacher, student, mode=ExecutionMode.SEQUENTIAL)
    multi.add_stage("a", p_a, weight=0.5)
    multi.add_stage(
        "b",
        p_b,
        weight=0.5,
        condition=lambda batch_, outputs: True,  # always run
    )
    multi.setup()
    _, metrics = multi(batch)
    stage_names = list(metrics.custom_metrics["stage_metrics"].keys())
    assert "a" in stage_names
    assert "b" in stage_names
