"""Numerically-stable Causal-LM distillation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    """Configuration for token-level Causal-LM distillation."""

    temperature: float = 2.0
    alpha: float = 0.7
    use_ce: bool = True
    ignore_index: int = -100
    shift_labels: bool = True
    logit_clip: Optional[float] = 80.0
    min_valid_tokens: int = 1


@dataclass
class DistillationLossOutput:
    """Container for distillation loss components and health flags."""

    total: torch.Tensor
    kd: torch.Tensor
    ce: torch.Tensor
    valid_tokens: int
    is_finite: bool
    warning: Optional[str] = None


class CausalLMDistillationEngine:
    """Stable KD+CE loss for GPT-style teacher->student training."""

    def __init__(self, config: DistillationConfig):
        self.config = config

    @staticmethod
    def extract_logits(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            return outputs["logits"]
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _clip_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.logit_clip is None:
            return logits
        return torch.clamp(logits, min=-float(self.config.logit_clip), max=float(self.config.logit_clip))

    def _shift(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.shift_labels:
            return (
                student_logits[:, :-1, :].contiguous(),
                teacher_logits[:, :-1, :].contiguous(),
                labels[:, 1:].contiguous(),
            )
        return student_logits, teacher_logits, labels

    @staticmethod
    def _zeros_like(ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=torch.float32)

    def compute_total_loss(
        self,
        *,
        student_outputs: Any,
        teacher_outputs: Any,
        labels: torch.Tensor,
    ) -> DistillationLossOutput:
        student_logits = self.extract_logits(student_outputs)
        teacher_logits = self.extract_logits(teacher_outputs)

        if student_logits.dim() != 3 or teacher_logits.dim() != 3:
            raise ValueError(
                f"Expected [B,T,V] logits. Got student={tuple(student_logits.shape)}, teacher={tuple(teacher_logits.shape)}"
            )

        student_logits, teacher_logits, labels = self._shift(student_logits, teacher_logits, labels)

        # Promote to float32 for numerically stable softmax/log-softmax.
        student_logits = self._clip_logits(student_logits.float())
        teacher_logits = self._clip_logits(teacher_logits.float())

        flat_student = student_logits.view(-1, student_logits.size(-1))
        flat_teacher = teacher_logits.view(-1, teacher_logits.size(-1))
        flat_labels = labels.reshape(-1)
        valid_mask = flat_labels != int(self.config.ignore_index)
        valid_tokens = int(valid_mask.sum().item())

        if valid_tokens < int(self.config.min_valid_tokens):
            zero = self._zeros_like(student_logits)
            return DistillationLossOutput(
                total=zero,
                kd=zero,
                ce=zero,
                valid_tokens=valid_tokens,
                is_finite=True,
                warning="No valid tokens in batch after ignore_index masking.",
            )

        s_valid = flat_student[valid_mask]
        t_valid = flat_teacher[valid_mask]
        y_valid = flat_labels[valid_mask]

        t = max(float(self.config.temperature), 1e-6)
        s_log_probs = F.log_softmax(s_valid / t, dim=-1)
        t_probs = F.softmax(t_valid / t, dim=-1)
        kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (t * t)

        if self.config.use_ce:
            ce_loss = F.cross_entropy(s_valid, y_valid, reduction="mean")
            total_loss = float(self.config.alpha) * kd_loss + (1.0 - float(self.config.alpha)) * ce_loss
        else:
            ce_loss = self._zeros_like(kd_loss)
            total_loss = kd_loss

        finite = bool(
            torch.isfinite(total_loss).all()
            and torch.isfinite(kd_loss).all()
            and torch.isfinite(ce_loss).all()
        )
        if not finite:
            warning = "Non-finite loss detected in distillation output."
        else:
            warning = None

        return DistillationLossOutput(
            total=total_loss,
            kd=kd_loss,
            ce=ce_loss,
            valid_tokens=valid_tokens,
            is_finite=finite,
            warning=warning,
        )
