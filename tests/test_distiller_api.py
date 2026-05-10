"""Tests for the public Distiller and DistillationToolkit API.

Covers:
- Goal enum backward compatibility with strings
- DistillationConfig dataclass defaults and override generation
- Input validation (non-Module raises TypeError, bad goal raises ValueError)
- All valid goal strings resolve to correct presets
- Device auto-correction between teacher and student
- Preset plan structure validation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from zynthe.core.distillers.toolkit import (
    DistillationToolkit,
    Distiller,
    Goal,
    DistillationConfig,
    _validate_model,
    _validate_goal,
)
from zynthe.core.distillers.presets import list_presets, get_preset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal torch Module for testing."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture()
def teacher_student():
    torch.manual_seed(0)
    return TinyModel(), TinyModel()


# ---------------------------------------------------------------------------
# Goal enum
# ---------------------------------------------------------------------------

class TestGoal:
    """Tests for the Goal str-enum."""

    def test_goal_is_string(self):
        """Goal members should be equal to their string value."""
        assert Goal.VISION == "vision"
        assert Goal.TEXT == "text"
        assert Goal.CLIP == "clip"

    def test_goal_usable_as_dict_key(self):
        """Goal should work as dict lookup key alongside raw strings."""
        mapping = {"vision": "vision_transformer"}
        assert mapping[Goal.VISION] == "vision_transformer"

    def test_all_goals_have_preset_mapping(self):
        """Every Goal enum member should map to a valid preset."""
        for goal in Goal:
            preset = DistillationToolkit.GOAL_TO_PRESET.get(goal.value)
            assert preset is not None, f"Goal.{goal.name} ({goal.value}) has no preset mapping"


# ---------------------------------------------------------------------------
# DistillationConfig
# ---------------------------------------------------------------------------

class TestDistillationConfig:
    """Tests for the DistillationConfig dataclass."""

    def test_default_values(self):
        cfg = DistillationConfig()
        assert cfg.temperature == 4.0
        assert cfg.alpha == 0.85
        assert cfg.epochs is None

    def test_to_overrides_empty_when_defaults(self):
        """Default config should produce no overrides."""
        cfg = DistillationConfig()
        assert cfg.to_overrides() == {}

    def test_to_overrides_with_epochs(self):
        cfg = DistillationConfig(epochs=10)
        overrides = cfg.to_overrides()
        assert overrides["training"]["epochs"] == 10
        assert overrides["train"]["epochs"] == 10

    def test_to_overrides_with_mixed_precision(self):
        cfg = DistillationConfig(mixed_precision=True, batch_size=32)
        overrides = cfg.to_overrides()
        assert overrides["train"]["mixed_precision"] is True
        assert overrides["train"]["batch_size"] == 32


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for model and goal validation."""

    def test_validate_model_rejects_non_module(self):
        with pytest.raises(TypeError, match="must be a torch.nn.Module"):
            _validate_model("not a model", "teacher")

    def test_validate_model_accepts_module(self):
        _validate_model(TinyModel(), "teacher")  # should not raise

    def test_validate_goal_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown distillation goal"):
            _validate_goal("nonexistent_goal_xyz")

    def test_validate_goal_accepts_valid(self):
        _validate_goal("vision")  # should not raise
        _validate_goal("text")
        _validate_goal("clip")

    def test_toolkit_rejects_non_module_teacher(self):
        with pytest.raises(TypeError):
            DistillationToolkit("not a model", TinyModel())

    def test_toolkit_rejects_non_module_student(self):
        with pytest.raises(TypeError):
            DistillationToolkit(TinyModel(), 42)

    def test_distiller_rejects_bad_goal(self):
        with pytest.raises(ValueError, match="Unknown distillation goal"):
            Distiller(TinyModel(), TinyModel(), goal="definitely_not_a_goal")


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------

class TestPresetResolution:
    """Tests for goal → preset resolution."""

    def test_all_string_goals_resolve(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        for goal_str, expected_preset in DistillationToolkit.GOAL_TO_PRESET.items():
            resolved = toolkit.resolve_preset(goal_str, None)
            assert resolved == expected_preset, (
                f"goal={goal_str!r} resolved to {resolved!r}, expected {expected_preset!r}"
            )

    def test_explicit_preset_overrides_goal(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        assert toolkit.resolve_preset("vision", "quick_start") == "quick_start"

    def test_none_goal_returns_default(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student, default_preset="compression_max")
        assert toolkit.resolve_preset(None, None) == "compression_max"

    def test_goal_enum_resolves(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        assert toolkit.resolve_preset(Goal.VISION, None) == "vision_transformer"
        assert toolkit.resolve_preset(Goal.CLIP, None) == "multimodal"


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------

class TestPlanBuilding:
    """Tests for build_plan output structure."""

    def test_plan_has_stages(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        plan = toolkit.build_plan(goal="balanced")
        assert "distillation" in plan
        assert "stages" in plan["distillation"]
        assert len(plan["distillation"]["stages"]) > 0

    def test_plan_metadata_contains_preset(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        plan = toolkit.build_plan(goal="vision")
        assert plan["metadata"]["preset"] == "vision_transformer"

    def test_vision_preset_has_attention_stage(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        plan = toolkit.build_plan(goal="vision")
        types = [s["type"] for s in plan["distillation"]["stages"]]
        assert "attention" in types

    def test_multimodal_preset_has_feature_stage(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        plan = toolkit.build_plan(goal="multimodal")
        types = [s["type"] for s in plan["distillation"]["stages"]]
        assert "feature" in types

    def test_causal_lm_preset_task_type(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        plan = toolkit.build_plan(goal="causal_lm")
        assert plan["distillation"].get("task_type") == "causal_lm"

    def test_config_overrides_applied(self, teacher_student):
        teacher, student = teacher_student
        cfg = DistillationConfig(epochs=42)
        toolkit = DistillationToolkit(teacher, student, config=cfg)
        plan = toolkit.build_plan(goal="quick")
        train_cfg = plan.get("train", plan.get("training", {}))
        assert train_cfg.get("epochs") == 42

    def test_dry_run_returns_plan(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student)
        result = toolkit.run(dry_run=True)
        assert isinstance(result, dict)
        assert "distillation" in result


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------

class TestDeviceHandling:
    """Tests for device auto-detection and alignment."""

    def test_resolve_device_cpu(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student, device="cpu")
        assert toolkit._resolve_device() == "cpu"

    def test_models_on_same_device(self, teacher_student):
        teacher, student = teacher_student
        toolkit = DistillationToolkit(teacher, student, device="cpu")
        t_device = next(toolkit.teacher.parameters()).device
        s_device = next(toolkit.student.parameters()).device
        assert t_device == s_device
