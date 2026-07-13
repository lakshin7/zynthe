"""Tests for :class:`RationaleDistiller` and :class:`RationaleDataset`.

Reference: Hsieh et al. 2023, "Distilling Step-by-Step".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zynthe.core.distillers.rationale_distiller import RationaleDistiller
from zynthe.data.rationale_dataset import RationaleDataset


# ----------------------------------------------------------------------------
# Distiller math
# ----------------------------------------------------------------------------


class _TwoHeadMod(nn.Module):
    """Stand-in student/teacher: a shared embed + a label head + a
    rationale head.  We only test the distiller's loss math, so the
    specifics of the heads are not important — we feed pre-computed
    logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.label_head = nn.Linear(8, 4)
        self.rationale_head = nn.Linear(8, 6)


def _ref_loss(
    label_logits: torch.Tensor,
    rationale_logits: torch.Tensor,
    label_ids: torch.Tensor,
    rationale_ids: torch.Tensor,
    ignore_index: int = -100,
    label_weight: float = 1.0,
    rationale_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-evaluated reference loss for the multi-task setup."""
    label_loss = F.cross_entropy(
        label_logits.reshape(-1, label_logits.size(-1)),
        label_ids.reshape(-1),
        ignore_index=ignore_index,
    )
    rationale_loss = F.cross_entropy(
        rationale_logits.reshape(-1, rationale_logits.size(-1)),
        rationale_ids.reshape(-1),
        ignore_index=ignore_index,
    )
    total = label_weight * label_loss + rationale_weight * rationale_loss
    return total, label_loss, rationale_loss


def test_loss_matches_closed_form_reference() -> None:
    """`total = w_l * CE_label + w_r * CE_rationale` matches hand-eval."""
    torch.manual_seed(0)
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = RationaleDistiller(
        teacher,
        student,
        config={
            "rationale": {
                "label_weight": 0.7,
                "rationale_weight": 1.3,
                "ignore_index": -100,
            }
        },
        device="cpu",
    )
    label_logits = torch.randn(2, 4)
    rationale_logits = torch.randn(2, 5, 6)
    label_ids = torch.tensor([0, 1])
    rationale_ids = torch.randint(0, 6, (2, 5))

    expected_total, expected_label, expected_rationale = _ref_loss(
        label_logits,
        rationale_logits,
        label_ids,
        rationale_ids,
        ignore_index=-100,
        label_weight=0.7,
        rationale_weight=1.3,
    )
    total, breakdown = d.compute_loss(
        student_outputs={"label_logits": label_logits, "rationale_logits": rationale_logits},
        targets={"label_ids": label_ids, "rationale_ids": rationale_ids},
    )
    assert torch.allclose(total, expected_total, atol=1e-6)
    assert math.isclose(breakdown["label"], expected_label.item(), abs_tol=1e-6)
    assert math.isclose(breakdown["rationale"], expected_rationale.item(), abs_tol=1e-6)
    assert math.isclose(breakdown["total"], expected_total.item(), abs_tol=1e-6)


def test_loss_changes_with_weights() -> None:
    """Different (label_weight, rationale_weight) produce different totals."""
    torch.manual_seed(1)
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d_a = RationaleDistiller(
        teacher, student, config={"rationale": {"label_weight": 1.0, "rationale_weight": 1.0}}
    )
    d_b = RationaleDistiller(
        teacher, student, config={"rationale": {"label_weight": 0.0, "rationale_weight": 1.0}}
    )
    label_logits = torch.randn(2, 4)
    rationale_logits = torch.randn(2, 5, 6)
    student_outputs = {"label_logits": label_logits, "rationale_logits": rationale_logits}
    targets = {
        "label_ids": torch.tensor([0, 1]),
        "rationale_ids": torch.randint(0, 6, (2, 5)),
    }
    total_a, _ = d_a.compute_loss(student_outputs=student_outputs, targets=targets)
    total_b, _ = d_b.compute_loss(student_outputs=student_outputs, targets=targets)
    assert not torch.allclose(total_a, total_b, atol=1e-4)


def test_zero_rationale_weight_drops_rationale_contribution() -> None:
    """Setting ``rationale_weight=0`` makes the loss independent of
    rationale_logits, so the total is just ``label_weight * CE_label``.
    """
    torch.manual_seed(2)
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = RationaleDistiller(
        teacher, student, config={"rationale": {"label_weight": 1.0, "rationale_weight": 0.0}}
    )
    label_logits = torch.randn(2, 4)
    rationale_logits_a = torch.randn(2, 5, 6)
    rationale_logits_b = torch.randn(2, 5, 6)
    targets = {
        "label_ids": torch.tensor([0, 1]),
        "rationale_ids": torch.randint(0, 6, (2, 5)),
    }
    total_a, _ = d.compute_loss(
        student_outputs={"label_logits": label_logits, "rationale_logits": rationale_logits_a},
        targets=targets,
    )
    total_b, _ = d.compute_loss(
        student_outputs={"label_logits": label_logits, "rationale_logits": rationale_logits_b},
        targets=targets,
    )
    # rationale_weight = 0 → loss independent of rationale_logits.
    assert torch.allclose(total_a, total_b, atol=1e-6)


# ----------------------------------------------------------------------------
# Input contract enforcement
# ----------------------------------------------------------------------------


def test_compute_loss_rejects_non_dict_student_outputs() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = RationaleDistiller(teacher, student)
    with pytest.raises(TypeError, match="label_logits"):
        d.compute_loss(
            student_outputs="not a dict",  # type: ignore[arg-type]
            targets={"label_ids": torch.tensor([0]), "rationale_ids": torch.tensor([[0]])},
        )


def test_compute_loss_rejects_non_dict_targets() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = RationaleDistiller(teacher, student)
    with pytest.raises(TypeError, match="label_ids"):
        d.compute_loss(
            student_outputs={"label_logits": torch.randn(1, 4), "rationale_logits": torch.randn(1, 5, 6)},
            targets="not a dict",  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------------
# ignore_index behavior
# ----------------------------------------------------------------------------


def test_ignore_index_excludes_label_tokens() -> None:
    torch.manual_seed(3)
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = RationaleDistiller(
        teacher, student, config={"rationale": {"ignore_index": -100}}
    )
    label_logits = torch.randn(1, 4)
    rationale_logits = torch.randn(1, 5, 6)
    rationale_ids = torch.randint(0, 6, (1, 5))

    # All label_ids == -100 → label loss == 0.
    targets = {
        "label_ids": torch.tensor([-100]),
        "rationale_ids": rationale_ids,
    }
    total, breakdown = d.compute_loss(
        student_outputs={"label_logits": label_logits, "rationale_logits": rationale_logits},
        targets=targets,
    )
    assert breakdown["label"] == 0.0
    # Total = 0 * 1.0 + rationale_loss * 1.0 (default).
    assert torch.allclose(total, breakdown["rationale"], atol=1e-6)


# ----------------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------------


def test_gradient_flows_through_both_label_and_rationale_logits() -> None:
    teacher = _TwoHeadMod()
    student = _TwoHeadMod()
    d = RationaleDistiller(teacher, student)
    label_logits = torch.randn(2, 4, requires_grad=True)
    rationale_logits = torch.randn(2, 5, 6, requires_grad=True)
    targets = {
        "label_ids": torch.tensor([0, 1]),
        "rationale_ids": torch.randint(0, 6, (2, 5)),
    }
    total, _ = d.compute_loss(
        student_outputs={"label_logits": label_logits, "rationale_logits": rationale_logits},
        targets=targets,
    )
    total.backward()
    assert label_logits.grad is not None and label_logits.grad.abs().sum() > 0
    assert rationale_logits.grad is not None and rationale_logits.grad.abs().sum() > 0


# ----------------------------------------------------------------------------
# RationaleDataset
# ----------------------------------------------------------------------------


def test_dataset_roundtrip(tmp_path: Path) -> None:
    """JSONL -> dataset -> record roundtrip preserves input/label/rationale."""
    path = tmp_path / "r.jsonl"
    records = [
        {"input": "2 + 2", "label": "4", "rationale": "Two plus two equals four."},
        {"input": "5 - 3", "label": "2", "rationale": "Five minus three equals two."},
    ]
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds = RationaleDataset(path)
    assert len(ds) == 2
    for i, r in enumerate(records):
        got = ds[i]
        assert got["input"] == r["input"]
        assert got["label"] == r["label"]
        assert got["rationale"] == r["rationale"]


def test_dataset_rejects_missing_keys_when_required(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"input": "x", "label": "y"}) + "\n")  # no rationale
    with pytest.raises(ValueError, match="missing one of"):
        RationaleDataset(path, required=True)


def test_dataset_skips_missing_keys_when_not_required(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"input": "x", "label": "y", "rationale": "z"}) + "\n")
        f.write(json.dumps({"input": "x", "label": "y"}) + "\n")  # no rationale
        f.write(json.dumps({"input": "x", "label": "y", "rationale": "w"}) + "\n")
    ds = RationaleDataset(path, required=False)
    assert len(ds) == 2  # middle one skipped


def test_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RationaleDataset(tmp_path / "nope.jsonl")


# ----------------------------------------------------------------------------
# Registry wiring
# ----------------------------------------------------------------------------


def test_rationale_distiller_registered() -> None:
    from zynthe.core.distillers.multi_stage_distiller import DistillerRegistry

    reg = DistillerRegistry()
    assert reg.get("rationale") is RationaleDistiller
    assert reg.get("distill_step_by_step") is RationaleDistiller
