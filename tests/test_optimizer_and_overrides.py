from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.cli_helpers import parse_overrides
from training.optimizer import OptimizerFactory


def test_parse_overrides_parses_scientific_notation_float() -> None:
    overrides = parse_overrides(["train.learning_rate=3e-5", "train.weight_decay=1e-2"])

    train_cfg = overrides["train"]
    assert train_cfg["learning_rate"] == pytest.approx(3e-5)
    assert train_cfg["weight_decay"] == pytest.approx(1e-2)


def test_optimizer_accepts_learning_rate_string_in_scientific_notation() -> None:
    model = nn.Linear(8, 2)

    optimizer = OptimizerFactory.get_optimizer(
        model,
        {
            "optimizer": "adamw",
            "learning_rate": "5e-5",
            "weight_decay": "1e-2",
        },
        phase="distillation",
    )

    assert isinstance(optimizer, torch.optim.Optimizer)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(5e-5)
