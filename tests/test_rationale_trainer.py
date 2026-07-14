"""Tests for the multi-task T5 trainer (Phase 6 Iteration 2).

Pins:
* from_pretrained loads model + tokenizer and wraps them.
* _encode puts inputs on the model's device.
* _teacher_forcing_targets shifts right correctly (T5 convention).
* forward_label / forward_rationale produce (1, T, V) logits each.
* forward_both returns a RationaleDistiller-shaped dict.
* train_step runs the two forward passes and calls the distiller
  with the right dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make sure the src/ layout is importable when pytest runs this file
# in isolation (Modal's pytest discovery doesn't always pick up
# pyproject's pythonpath).
_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from zynthe.core.training.rationale_trainer import MultiTaskT5Trainer  # noqa: E402


# We use a tiny seq2seq model for the test so it loads fast and the
# smoke covers the contract, not a particular T5 variant.
TEST_MODEL = "patrickvonplaten/t5-tiny-random"


@pytest.fixture(scope="module")
def trainer() -> MultiTaskT5Trainer:
    return MultiTaskT5Trainer.from_pretrained(TEST_MODEL, device="cpu")


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------


def test_from_pretrained_returns_wrapped_trainer(trainer) -> None:
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.label_prefix == "label: "
    assert trainer.rationale_prefix == "rationale: "


def test_from_pretrained_respects_custom_prefixes() -> None:
    t = MultiTaskT5Trainer.from_pretrained(
        TEST_MODEL,
        label_prefix="[L] ",
        rationale_prefix="[R] ",
        device="cpu",
    )
    assert t.label_prefix == "[L] "
    assert t.rationale_prefix == "[R] "


# ----------------------------------------------------------------------------
# Tokenisation helpers
# ----------------------------------------------------------------------------


def test_encode_uses_default_max_length(trainer) -> None:
    enc = trainer._encode("hello world")
    assert "input_ids" in enc
    assert "attention_mask" in enc
    # Both tensors are on the trainer's device.
    assert enc["input_ids"].device == trainer.device


def test_encode_respects_max_length(trainer) -> None:
    enc = trainer._encode("hello world", max_length=16)
    assert enc["input_ids"].shape[1] == 16


def test_teacher_forcing_shifts_right(trainer) -> None:
    """T5's shift-right: prepend decoder_start_token, drop the last col."""
    target = torch.tensor([[5, 6, 7, 8, 9]])
    shifted = trainer._teacher_forcing_targets(target)
    # First col becomes the decoder start token.
    assert int(shifted[0, 0]) == trainer.label_decoder_start_token_id
    # Subsequent cols come from the original target's prefix.
    assert shifted[0, 1:].tolist() == [5, 6, 7, 8]


# ----------------------------------------------------------------------------
# Forward
# ----------------------------------------------------------------------------


def test_forward_label_returns_logits(trainer) -> None:
    with torch.no_grad():
        logits = trainer.forward_label("a sentence")
    assert logits.dim() == 3
    # T5's vocab size is in the 30k+ range; we just check it's > 100.
    assert logits.shape[-1] > 100


def test_forward_rationale_returns_logits(trainer) -> None:
    with torch.no_grad():
        logits = trainer.forward_rationale("a sentence")
    assert logits.dim() == 3
    assert logits.shape[-1] > 100


def test_forward_label_and_rationale_have_same_vocab(trainer) -> None:
    """Both views must agree on vocab size so the distiller's CE lines up."""
    with torch.no_grad():
        a = trainer.forward_label("same input")
        b = trainer.forward_rationale("same input")
    assert a.shape[-1] == b.shape[-1]


def test_forward_both_returns_distiller_dict(trainer) -> None:
    with torch.no_grad():
        out = trainer.forward_both("a sentence", max_length=32)
    assert set(out.keys()) == {"label_logits", "rationale_logits"}
    assert out["label_logits"].dim() == 3
    assert out["rationale_logits"].dim() == 3


def test_label_prefix_actually_prepends(trainer) -> None:
    """Tokenising with the prefix differs from tokenising without it."""
    bare = trainer._encode("a sentence", max_length=32)
    with_prefix = trainer._encode(trainer.label_prefix + "a sentence", max_length=32)
    # The two input_ids sequences should differ.
    assert not torch.equal(bare["input_ids"], with_prefix["input_ids"])


def test_label_and_rationale_logits_differ(trainer) -> None:
    """The two views should produce different logit distributions."""
    with torch.no_grad():
        a = trainer.forward_label("a sentence")
        b = trainer.forward_rationale("a sentence")
    assert not torch.allclose(a.float(), b.float(), atol=1e-3)


# ----------------------------------------------------------------------------
# Multi-task training step
# ----------------------------------------------------------------------------


def test_train_step_returns_loss_and_breakdown(trainer) -> None:
    """End-to-end: one multi-task step with a stub distiller that just
    returns label + rationale CE.
    """
    class _StubDistiller:
        def compute_loss(self, student_outputs, targets):
            # Reshape to (B*T, V) vs (B*T,) and use ignore_index=-100
            # to handle the masked positions.
            label_loss = torch.nn.functional.cross_entropy(
                student_outputs["label_logits"].float().view(-1, student_outputs["label_logits"].size(-1)),
                targets["label_ids"].view(-1),
                ignore_index=-100,
            )
            rationale_loss = torch.nn.functional.cross_entropy(
                student_outputs["rationale_logits"].float().view(-1, student_outputs["rationale_logits"].size(-1)),
                targets["rationale_ids"].view(-1),
                ignore_index=-100,
            )
            total = label_loss + rationale_loss
            return total, {
                "label": label_loss.item(),
                "rationale": rationale_loss.item(),
                "total": total.item(),
            }

    tokenizer = trainer.tokenizer
    label_target = tokenizer(
        "positive", return_tensors="pt", padding="max_length", max_length=8, truncation=True
    )["input_ids"]
    rationale_target = tokenizer(
        "a delight of wit and warmth",
        return_tensors="pt",
        padding="max_length",
        max_length=16,
        truncation=True,
    )["input_ids"]
    batch = {
        "input": "a delightful film",
        "label_ids": label_target,
        "rationale_ids": rationale_target,
        "max_length": 32,
    }
    optim = torch.optim.SGD(trainer.model.parameters(), lr=1e-3)
    loss, breakdown = trainer.train_step(batch, distiller=_StubDistiller(), optimizer=optim)
    assert torch.isfinite(loss)
    assert "label" in breakdown
    assert "rationale" in breakdown
    assert "total" in breakdown
    assert breakdown["label"] > 0
    assert breakdown["rationale"] > 0


def test_train_step_calls_distiller_compute_loss(trainer) -> None:
    """The trainer's train_step must pass the distiller a dict the
    distiller can consume (label_logits, rationale_logits, label_ids,
    rationale_ids).  The distiller returns a loss that depends on
    the inputs so backward() works.
    """
    captured: dict = {}

    class _CapturingDistiller:
        def compute_loss(self, student_outputs, targets):
            captured["student_outputs"] = student_outputs
            captured["targets"] = targets
            # Use the logit-sum so the loss is a real scalar with grad_fn.
            loss = student_outputs["label_logits"].float().sum() * 1e-6
            return loss, {"total": loss.item()}

    tokenizer = trainer.tokenizer
    batch = {
        "input": "a sample",
        "label_ids": tokenizer(
            "pos", return_tensors="pt", padding="max_length", max_length=4, truncation=True
        )["input_ids"],
        "rationale_ids": tokenizer(
            "because", return_tensors="pt", padding="max_length", max_length=4, truncation=True
        )["input_ids"],
    }
    optim = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
    trainer.train_step(batch, distiller=_CapturingDistiller(), optimizer=optim)
    assert "label_logits" in captured["student_outputs"]
    assert "rationale_logits" in captured["student_outputs"]
    assert "label_ids" in captured["targets"]
    assert "rationale_ids" in captured["targets"]

    tokenizer = trainer.tokenizer
    batch = {
        "input": "a sample",
        "label_ids": tokenizer(
            "pos", return_tensors="pt", padding="max_length", max_length=4, truncation=True
        )["input_ids"],
        "rationale_ids": tokenizer(
            "because", return_tensors="pt", padding="max_length", max_length=4, truncation=True
        )["input_ids"],
    }
    optim = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
    trainer.train_step(batch, distiller=_CapturingDistiller(), optimizer=optim)
    assert "label_logits" in captured["student_outputs"]
    assert "rationale_logits" in captured["student_outputs"]
    assert "label_ids" in captured["targets"]
    assert "rationale_ids" in captured["targets"]
