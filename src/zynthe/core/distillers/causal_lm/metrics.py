"""LM-specific metrics for stable Causal-LM distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


@dataclass
class DistillationHealthMetrics:
    """Health counters for fault-tolerant training loops."""

    nan_events: int = 0
    inf_events: int = 0
    invalid_logits: int = 0
    skipped_steps: int = 0
    overflow_events: int = 0
    grad_explosion_events: int = 0
    frozen_loss_steps: int = 0
    fallback_checkpoint_loads: int = 0
    zero_grad_steps: int = 0
    bad_grad_steps: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "nan_events": self.nan_events,
            "inf_events": self.inf_events,
            "invalid_logits": self.invalid_logits,
            "skipped_steps": self.skipped_steps,
            "overflow_events": self.overflow_events,
            "grad_explosion_events": self.grad_explosion_events,
            "frozen_loss_steps": self.frozen_loss_steps,
            "fallback_checkpoint_loads": self.fallback_checkpoint_loads,
            "zero_grad_steps": self.zero_grad_steps,
            "bad_grad_steps": self.bad_grad_steps,
        }


class MetricStabilityMonitor:
    """Detect exploding perplexity and frozen-loss plateaus."""

    def __init__(
        self,
        *,
        freeze_window: int = 10,
        freeze_tolerance: float = 1e-7,
        perplexity_warn_threshold: float = 1e6,
    ):
        self.freeze_window = max(2, int(freeze_window))
        self.freeze_tolerance = float(freeze_tolerance)
        self.perplexity_warn_threshold = float(perplexity_warn_threshold)
        self.loss_history: List[float] = []

    @staticmethod
    def stable_perplexity(token_loss: float, max_exp_input: float = 20.0) -> float:
        return float(math.exp(min(float(token_loss), float(max_exp_input))))

    def update(self, token_loss: float) -> Dict[str, Any]:
        loss = float(token_loss)
        self.loss_history.append(loss)
        if len(self.loss_history) > self.freeze_window:
            self.loss_history = self.loss_history[-self.freeze_window :]

        frozen = False
        if len(self.loss_history) >= self.freeze_window:
            recent = self.loss_history[-self.freeze_window :]
            frozen = (max(recent) - min(recent)) <= self.freeze_tolerance

        perplexity = self.stable_perplexity(loss)
        perplexity_exploded = perplexity >= self.perplexity_warn_threshold

        return {
            "token_loss": loss,
            "perplexity": perplexity,
            "frozen_loss": frozen,
            "perplexity_exploded": perplexity_exploded,
        }


class TokenMetricsAccumulator:
    """Batch-safe token metrics: NLL, perplexity, and token accuracy."""

    def __init__(self, ignore_index: int = -100, shift_labels: bool = True):
        self.ignore_index = int(ignore_index)
        self.shift_labels = bool(shift_labels)
        self.reset()

    def reset(self) -> None:
        self.total_nll = 0.0
        self.total_tokens = 0
        self.total_correct = 0

    def update_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.dim() != 3 or labels.dim() < 2:
            return

        if self.shift_labels:
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

        flat_logits = logits.view(-1, logits.size(-1)).float()
        flat_labels = labels.reshape(-1)
        valid = flat_labels != self.ignore_index
        if not valid.any():
            return

        valid_logits = flat_logits[valid]
        valid_labels = flat_labels[valid]

        # Sum reduction keeps exact token accounting across variable batch sizes.
        nll = F.cross_entropy(valid_logits, valid_labels, reduction="sum")
        self.total_nll += float(nll.item())
        token_count = int(valid_labels.numel())
        self.total_tokens += token_count

        preds = valid_logits.argmax(dim=-1)
        self.total_correct += int((preds == valid_labels).sum().item())

    def compute(self) -> Dict[str, float]:
        if self.total_tokens <= 0:
            return {
                "token_loss": 0.0,
                "perplexity": float("inf"),
                "token_accuracy": 0.0,
                "valid_tokens": 0.0,
            }

        token_loss = self.total_nll / max(self.total_tokens, 1)
        # Clamp exponent input to avoid inf in exp for unusually high losses.
        ppl = math.exp(min(token_loss, 20.0))
        token_acc = self.total_correct / max(self.total_tokens, 1)
        return {
            "token_loss": float(token_loss),
            "perplexity": float(ppl),
            "token_accuracy": float(token_acc),
            "valid_tokens": float(self.total_tokens),
        }


def compute_distill_alignment(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool = True,
    temperature: float = 2.0,
) -> Dict[str, float]:
    """Compute token-level teacher/student KL and top-1 agreement on valid tokens."""

    if student_logits.dim() != 3 or teacher_logits.dim() != 3 or labels.dim() < 2:
        return {"distill_kl": 0.0, "top1_agreement": 0.0}

    if shift_labels:
        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    flat_s = student_logits.view(-1, student_logits.size(-1)).float()
    flat_t = teacher_logits.view(-1, teacher_logits.size(-1)).float()
    flat_y = labels.reshape(-1)

    valid = flat_y != int(ignore_index)
    if not valid.any():
        return {"distill_kl": 0.0, "top1_agreement": 0.0}

    flat_s = flat_s[valid]
    flat_t = flat_t[valid]

    t = max(float(temperature), 1e-6)
    s_log_probs = F.log_softmax(flat_s / t, dim=-1)
    t_probs = F.softmax(flat_t / t, dim=-1)
    distill_kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (t * t)

    s_argmax = flat_s.argmax(dim=-1)
    t_argmax = flat_t.argmax(dim=-1)
    agreement = (s_argmax == t_argmax).float().mean()

    return {
        "distill_kl": float(distill_kl.item()) if torch.isfinite(distill_kl).all() else 0.0,
        "top1_agreement": float(agreement.item()) if torch.isfinite(agreement).all() else 0.0,
    }


def metric_value(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except Exception:
        return default
