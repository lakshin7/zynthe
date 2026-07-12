"""Behavior tests for the causal-LM distillation engine.

The :class:`CausalLMDistillationEngine` is the numerical core used by
:class:`SafeCausalLMTrainer`. It implements the Hinton KD loss on
token-shifted sequences and returns a structured
:class:`DistillationLossOutput`.

These tests pin:

* Token-shift contract (predict next token, drop the last logit).
* :math:`L = α T² KL(q_s ‖ q_t) + (1 - α) CE(s, y)` numerically on
  hand-evaluated fixtures.
* Ignore-index masking drops pad tokens, not their loss.
* Zero-valid-tokens path returns a warning rather than NaN.
* Loss components remain finite under fp16-class overflow inputs.
* Output dataclass fields are populated and the loss has the right
  shape.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from zynthe.core.distillers.causal_lm.distillation import (
    CausalLMDistillationEngine,
    DistillationConfig,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _ref_causal_lm_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    alpha: float,
    ignore_index: int,
    shift: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-evaluated reference to compare against the engine."""
    if shift:
        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    flat_s = student_logits.view(-1, student_logits.size(-1))
    flat_t = teacher_logits.view(-1, teacher_logits.size(-1))
    flat_y = labels.reshape(-1)
    valid = flat_y != ignore_index
    flat_s = flat_s[valid]
    flat_t = flat_t[valid]
    flat_y = flat_y[valid]

    t = max(temperature, 1e-6)
    s_log = F.log_softmax(flat_s / t, dim=-1)
    t_prob = F.softmax(flat_t / t, dim=-1)
    kd = F.kl_div(s_log, t_prob, reduction="batchmean") * (t * t)

    ce = F.cross_entropy(flat_s, flat_y, reduction="mean")
    total = alpha * kd + (1.0 - alpha) * ce
    return total, kd, ce


@pytest.fixture()
def engine() -> CausalLMDistillationEngine:
    return CausalLMDistillationEngine(
        DistillationConfig(temperature=2.0, alpha=0.7, ignore_index=-100, use_ce=True)
    )


# ----------------------------------------------------------------------------
# Extract logits (output-shape polymorphism)
# ----------------------------------------------------------------------------


def test_extract_logits_from_dict() -> None:
    out = CausalLMDistillationEngine.extract_logits({"logits": torch.zeros(1, 2, 3)})
    assert out.shape == (1, 2, 3)


def test_extract_logits_from_object_with_logits_attr() -> None:
    class _O:
        logits = torch.zeros(1, 2, 3)

    out = CausalLMDistillationEngine.extract_logits(_O())
    assert out.shape == (1, 2, 3)


def test_extract_logits_from_tuple_uses_first() -> None:
    out = CausalLMDistillationEngine.extract_logits((torch.zeros(1, 2, 3), "ignored"))
    assert out.shape == (1, 2, 3)


# ----------------------------------------------------------------------------
# Token shift + logit-clip contract
# ----------------------------------------------------------------------------


def test_compute_total_loss_shifts_targets() -> None:
    """Validate that, after the internal shift, the loss is computed
    against ``labels[:, 1:]`` (predict-the-next-token).
    """
    engine = CausalLMDistillationEngine(
        DistillationConfig(temperature=1.0, alpha=0.0, ignore_index=-100, use_ce=True)
    )
    torch.manual_seed(0)
    student_logits = torch.randn(2, 5, 7)
    teacher_logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))

    # alpha=0 -> loss == CE on the shifted (predict next token) labels.
    out = engine.compute_total_loss(
        student_outputs=student_logits,
        teacher_outputs=teacher_logits,
        labels=labels,
    )
    expected_ce, _, _ = _ref_causal_lm_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=1.0,
        alpha=0.0,
        ignore_index=-100,
    )
    assert torch.allclose(out.total, expected_ce, atol=1e-5)


@pytest.mark.parametrize(
    "alpha",
    [0.0, 0.5, 1.0],
)
def test_compute_total_loss_matches_reference(alpha: float) -> None:
    """Closed-form check across a few α values, fp32 only."""
    torch.manual_seed(1)
    student = torch.randn(2, 4, 6)
    teacher = torch.randn(2, 4, 6)
    labels = torch.randint(0, 6, (2, 4))
    # Drop a token row so ignore_index filtering kicks in.
    labels[0, 2] = -100

    engine = CausalLMDistillationEngine(
        DistillationConfig(temperature=2.0, alpha=alpha, ignore_index=-100, use_ce=True)
    )
    out = engine.compute_total_loss(
        student_outputs=student,
        teacher_outputs=teacher,
        labels=labels,
    )
    ref_total, ref_kd, ref_ce = _ref_causal_lm_loss(
        student,
        teacher,
        labels,
        temperature=2.0,
        alpha=alpha,
        ignore_index=-100,
    )
    assert torch.allclose(out.total, ref_total, atol=1e-5)
    assert torch.allclose(out.kd, ref_kd, atol=1e-5)
    assert torch.allclose(out.ce, ref_ce, atol=1e-5)


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


def test_empty_batch_after_masking_returns_warning_not_nan() -> None:
    """If every token in the batch is the ignore_index, the engine
    must return a finite-zero output with a warning instead of NaN.
    """
    engine = CausalLMDistillationEngine(
        DistillationConfig(temperature=2.0, alpha=0.7, ignore_index=-100, use_ce=True)
    )
    torch.manual_seed(2)
    student = torch.randn(1, 3, 4)
    teacher = torch.randn(1, 3, 4)
    labels = torch.full((1, 3), -100, dtype=torch.long)
    out = engine.compute_total_loss(
        student_outputs=student,
        teacher_outputs=teacher,
        labels=labels,
    )
    assert out.is_finite
    assert torch.isfinite(out.total)
    assert out.warning is not None and "No valid tokens" in out.warning
    assert out.valid_tokens == 0


def test_logit_clip_keeps_loss_finite_with_extreme_inputs() -> None:
    """Inputs that would normally overflow softmax/KL (logits ~5e3) are
    clipped to ``logit_clip`` (default 80.0). Verify the engine stays
    finite.
    """
    engine = CausalLMDistillationEngine(
        DistillationConfig(temperature=2.0, alpha=0.5, ignore_index=-100, logit_clip=80.0)
    )
    torch.manual_seed(3)
    student = torch.randn(2, 4, 5) * 200.0  # would overflow
    teacher = torch.randn(2, 4, 5) * 200.0
    labels = torch.randint(0, 5, (2, 4))
    out = engine.compute_total_loss(
        student_outputs=student,
        teacher_outputs=teacher,
        labels=labels,
    )
    assert out.is_finite
    assert torch.isfinite(out.total)


def test_use_ce_false_drops_ce_term() -> None:
    """When ``use_ce=False`` the engine returns total == kd (no CE mix)."""
    engine = CausalLMDistillationEngine(
        DistillationConfig(temperature=2.0, alpha=0.7, ignore_index=-100, use_ce=False)
    )
    torch.manual_seed(4)
    student = torch.randn(2, 4, 6)
    teacher = torch.randn(2, 4, 6)
    labels = torch.randint(0, 6, (2, 4))
    out = engine.compute_total_loss(
        student_outputs=student,
        teacher_outputs=teacher,
        labels=labels,
    )
    assert torch.allclose(out.total, out.kd, atol=1e-6)
    assert torch.allclose(out.ce, torch.zeros((), dtype=out.kd.dtype), atol=1e-6)


def test_shift_disabled_keeps_full_sequence() -> None:
    """With shift_labels=False the engine uses positions 1..T for both
    logits and labels (predicts current token, not next).
    """
    engine = CausalLMDistillationEngine(
        DistillationConfig(
            temperature=1.0, alpha=0.0, ignore_index=-100, use_ce=True, shift_labels=False
        )
    )
    torch.manual_seed(5)
    student = torch.randn(2, 5, 7)
    teacher = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    out = engine.compute_total_loss(
        student_outputs=student,
        teacher_outputs=teacher,
        labels=labels,
    )
    expected_ce, _, _ = _ref_causal_lm_loss(
        student,
        teacher,
        labels,
        temperature=1.0,
        alpha=0.0,
        ignore_index=-100,
        shift=False,
    )
    assert torch.allclose(out.total, expected_ce, atol=1e-5)


# ----------------------------------------------------------------------------
# Output dataclass
# ----------------------------------------------------------------------------


def test_output_dataclass_has_required_fields() -> None:
    engine = CausalLMDistillationEngine(DistillationConfig())
    torch.manual_seed(6)
    student = torch.randn(1, 4, 5)
    teacher = torch.randn(1, 4, 5)
    labels = torch.randint(0, 5, (1, 4))
    out = engine.compute_total_loss(
        student_outputs=student,
        teacher_outputs=teacher,
        labels=labels,
    )
    # 4 labels → 3 positions after shift → 3 valid tokens.
    assert out.valid_tokens == 3
    assert out.is_finite
    assert out.warning is None
    assert out.total.dim() == 0  # scalar tensor
