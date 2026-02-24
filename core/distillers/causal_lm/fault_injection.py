"""Controlled fault injection for Causal-LM training safety validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

LOG = logging.getLogger(__name__)


@dataclass
class FaultInjectionConfig:
    enabled: bool = False
    mode: str = "none"  # one of: none,nan_loss,invalid_logits,exploding_grad,amp_overflow
    trigger_step: int = 1
    scale: float = 1e6


class FaultInjector:
    """Injects synthetic numerical faults at controlled steps."""

    def __init__(self, config: Optional[FaultInjectionConfig] = None):
        self.config = config or FaultInjectionConfig()
        self.activation_count = 0

    @classmethod
    def from_mapping(cls, cfg: Dict[str, Any]) -> "FaultInjector":
        return cls(
            FaultInjectionConfig(
                enabled=bool(cfg.get("enabled", False)),
                mode=str(cfg.get("mode", "none")).lower(),
                trigger_step=int(cfg.get("trigger_step", 1)),
                scale=float(cfg.get("scale", 1e6)),
            )
        )

    def should_trigger(self, step: int) -> bool:
        if not self.config.enabled:
            return False
        return int(step) >= int(self.config.trigger_step) and self.activation_count == 0

    def maybe_inject_loss(self, loss: torch.Tensor, *, step: int) -> torch.Tensor:
        if not self.should_trigger(step):
            return loss

        if self.config.mode == "nan_loss":
            self.activation_count += 1
            LOG.warning("FaultInjector: injecting NaN loss at step=%d", step)
            return loss * torch.tensor(float("nan"), device=loss.device)

        if self.config.mode == "amp_overflow":
            self.activation_count += 1
            LOG.warning("FaultInjector: injecting AMP-overflow style large loss at step=%d", step)
            return loss * float(self.config.scale)

        return loss

    def maybe_inject_logits(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        if not self.should_trigger(step):
            return {"student_logits": student_logits, "teacher_logits": teacher_logits}

        if self.config.mode == "invalid_logits":
            self.activation_count += 1
            LOG.warning("FaultInjector: injecting invalid logits at step=%d", step)
            bad = torch.full_like(student_logits, float("nan"))
            return {"student_logits": bad, "teacher_logits": teacher_logits}

        return {"student_logits": student_logits, "teacher_logits": teacher_logits}

    def maybe_inject_gradients(self, model: torch.nn.Module, *, step: int) -> bool:
        if not self.should_trigger(step):
            return False

        if self.config.mode == "exploding_grad":
            self.activation_count += 1
            LOG.warning("FaultInjector: injecting exploding gradients at step=%d", step)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(float(self.config.scale))
            return True

        return False
