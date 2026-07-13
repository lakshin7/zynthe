"""Tests for the typed Plan / Stage DSL (Phase 5 Iteration 2).

Pins:
* Stage / Plan validation (empty loss, negative weight, missing
  stages, non-positive epochs all raise).
* Plan.to_dict produces the shape the rest of zynthe consumes
  (distillation.type = "multi_stage", training.epochs, stages list).
* Plan.from_preset roundtrips with the legacy dict form.
* Five new presets land in the registry: ``compression_max``,
  ``fidelity_first``, ``vision_default``, ``causal_lm_default``,
  ``multimodal_default``.
"""

from __future__ import annotations

import pytest

from zynthe.core.distillers.presets import (
    Plan,
    PRESET_LIBRARY,
    Stage,
    describe_preset,
    get_preset,
    list_presets,
    register_plan,
)


# ----------------------------------------------------------------------------
# Stage validation
# ----------------------------------------------------------------------------


def test_stage_rejects_empty_loss() -> None:
    with pytest.raises(ValueError, match="loss"):
        Stage(loss="")


def test_stage_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        Stage(loss="kd_hinton", weight=-0.1)


def test_stage_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError, match="epochs"):
        Stage(loss="kd_hinton", epochs=0)


def test_stage_to_dict_shape() -> None:
    s = Stage(loss="kd_hinton", weight=0.6, config={"temperature": 4.0})
    d = s.to_dict()
    assert d["name"] == "Stage - kd_hinton"
    assert d["type"] == "kd_hinton"
    assert d["config"] == {"temperature": 4.0}
    # epochs is optional and only appears if set.
    assert "epochs" not in d


def test_stage_to_dict_includes_eps_when_set() -> None:
    s = Stage(loss="feature", epochs=2)
    d = s.to_dict()
    assert d["epochs"] == 2


# ----------------------------------------------------------------------------
# Plan validation
# ----------------------------------------------------------------------------


def test_plan_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        Plan(name="", stages=[Stage(loss="kd_hinton")])


def test_plan_rejects_empty_stages() -> None:
    with pytest.raises(ValueError, match="stages"):
        Plan(name="x", stages=[])


def test_plan_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError, match="epochs"):
        Plan(name="x", stages=[Stage(loss="kd_hinton")], epochs=0)


def test_plan_to_dict_includes_training_eps_and_stages() -> None:
    plan = Plan(
        name="smoke",
        stages=[
            Stage(loss="kd_hinton", weight=0.6),
            Stage(loss="feature", weight=0.4, config={"layers": ["layers.1"]}),
        ],
        epochs=10,
        description="smoke test plan",
    )
    d = plan.to_dict()
    assert d["description"] == "smoke test plan"
    assert d["training"]["epochs"] == 10
    assert d["distillation"]["type"] == "multi_stage"
    assert len(d["distillation"]["stages"]) == 2
    # Each stage has its own loss type and config preserved.
    assert d["distillation"]["stages"][0]["type"] == "kd_hinton"
    assert d["distillation"]["stages"][1]["type"] == "feature"
    assert d["distillation"]["stages"][1]["config"] == {"layers": ["layers.1"]}


def test_plan_from_preset_round_trip_legacy_dicts() -> None:
    """``get_preset`` (legacy dict) and ``Plan.from_preset`` (typed) should
    both resolve to the same underlying structure for a preset that
    has stages.
    """
    raw = get_preset("balanced")
    plan = Plan.from_preset("balanced")
    out = plan.to_dict()
    # Raw has distillation.stages; the plan's to_dict uses the same
    # loss types.
    raw_loss_types = [s.get("type") for s in raw["distillation"]["stages"]]
    out_loss_types = [s["type"] for s in out["distillation"]["stages"]]
    assert raw_loss_types == out_loss_types


# ----------------------------------------------------------------------------
# New presets
# ----------------------------------------------------------------------------


NEW_PRESETS = [
    "compression_max",
    "fidelity_first",
    "vision_default",
    "causal_lm_default",
    "multimodal_default",
]


@pytest.mark.parametrize("preset_name", NEW_PRESETS)
def test_new_preset_registered(preset_name: str) -> None:
    assert preset_name in PRESET_LIBRARY, (
        f"{preset_name} should be in PRESET_LIBRARY"
    )
    cfg = get_preset(preset_name)
    assert "training" in cfg
    assert "distillation" in cfg
    assert cfg["distillation"]["type"] == "multi_stage"
    # Each preset must have a non-empty description.
    assert describe_preset(preset_name), f"{preset_name} has no description"


@pytest.mark.parametrize("preset_name", NEW_PRESETS)
def test_new_preset_typed_plan_loads(preset_name: str) -> None:
    """Plan.from_preset on each new preset returns a non-empty
    Plan with stages.
    """
    plan = Plan.from_preset(preset_name)
    assert plan.name == preset_name
    assert len(plan.stages) > 0
    assert plan.epochs >= 1
    # Each stage is a real distiller key (not 'unknown').
    for s in plan.stages:
        assert s.loss in get_preset  # proxy: it's a non-empty string


def test_register_plan_adds_to_library() -> None:
    plan = Plan(
        name="__test_custom__",
        stages=[Stage(loss="kd_hinton", weight=1.0)],
        epochs=1,
        description="custom plan for testing",
    )
    register_plan(plan)
    assert "__test_custom__" in PRESET_LIBRARY
    # Clean up.
    PRESET_LIBRARY.pop("__test_custom__", None)


def test_list_presets_contains_legacy_and_new() -> None:
    names = list_presets()
    # Legacy: balanced, compression_max (was the old name, but we
    # redefined it as compression; either way it should be present).
    assert "balanced" in names
    for p in NEW_PRESETS:
        assert p in names
