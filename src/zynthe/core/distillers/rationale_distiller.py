"""Rationale Distillation (Distill step-by-step, Hsieh et al. 2023).

Reference: Hsieh, Li, Yeh, Nakhost, Fujii, Ratner, Krishna, Lee,
Pfister. "Distilling Step-by-Step: Outperforming Larger Language
Models with Less Training Data and Smaller Model Sizes", ACL 2023.

The mechanism: train a small student with **multi-task learning**:

    L = w_l * CE( label_logits,  label_ids )
      + w_r * CE( rationale_logits, rationale_ids )

Where ``label_logits`` and ``rationale_logits`` come from a single
student forward pass under two different task prefixes (e.g. ``label:
<input>`` and ``rationale: <input>``).  The student therefore learns
*why* the answer is what it is — the rationales come from a teacher
LLM (e.g. 540B PaLM) generated offline with chain-of-thought
prompting.

This distiller differs from the rest of the zynthe suite:

* It expects the student model to expose two heads (or two task-prefix
  views) and return a dict ``{"label_logits", "rationale_logits"}``.
* Targets are also a dict ``{"label_ids", "rationale_ids"}``.
* The teacher LLM is not used at training time — only the pre-extracted
  rationales are.

Configuration schema::

    rationale:
      label_prefix: "label: "
      rationale_prefix: "rationale: "
      label_weight: 1.0
      rationale_weight: 1.0
      ignore_index: -100
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


class RationaleDistiller(BaseDistiller):
    """Multi-task rationale distillation (Distill step-by-step)."""

    modality_type = "text"

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = {}

        rat_config = config.get("rationale", {}) or {}
        self.label_prefix = str(rat_config.get("label_prefix", "label: "))
        self.rationale_prefix = str(
            rat_config.get("rationale_prefix", "rationale: ")
        )
        self.label_weight = float(rat_config.get("label_weight", 1.0))
        self.rationale_weight = float(rat_config.get("rationale_weight", 1.0))
        self.ignore_index = int(rat_config.get("ignore_index", -100))

        # Phase-0 strict_layer_match — read from config.
        self.strict_layer_match = bool(
            rat_config.get(
                "strict_layer_match",
                config.get("strict_layer_match", False),
            )
        )

        super().__init__(teacher, student, config=config, device=device, **kwargs)

    def _register_hooks(self) -> None:
        # RationaleDistiller does not use feature hooks. The two forward
        # views (label / rationale) are produced by the trainer (or the
        # model's own task-prefix handling) and passed in as separate
        # logit tensors in ``student_outputs``.
        return

    def compute_loss(
        self,
        student_outputs: Any,
        teacher_outputs: Any = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        if not isinstance(student_outputs, dict) or "label_logits" not in student_outputs:
            raise TypeError(
                "RationaleDistiller.compute_loss expects "
                "student_outputs={'label_logits', 'rationale_logits'}, got "
                f"{type(student_outputs).__name__}"
            )
        if not isinstance(targets, dict) or "label_ids" not in targets:
            raise TypeError(
                "RationaleDistiller.compute_loss expects "
                "targets={'label_ids', 'rationale_ids'}, got "
                f"{type(targets).__name__}"
            )

        label_logits = student_outputs["label_logits"]
        rationale_logits = student_outputs["rationale_logits"]
        label_ids = targets["label_ids"]
        rationale_ids = targets["rationale_ids"]

        # Label loss: shape (B, V) vs (B,).
        flat_label_logits = label_logits.reshape(-1, label_logits.size(-1))
        flat_label_ids = label_ids.reshape(-1).to(label_logits.device)
        if (flat_label_ids != self.ignore_index).any():
            label_loss = F.cross_entropy(
                flat_label_logits,
                flat_label_ids,
                ignore_index=self.ignore_index,
            )
        else:
            # Every label token is the ignore index — no label signal
            # in this batch.  Treat as zero loss (rationale term
            # still carries the optimisation signal).
            label_loss = torch.zeros((), device=label_logits.device)

        # Rationale loss: shape (B, R, V) vs (B, R). Reshape to (B*R, V) vs (B*R,).
        flat_rat_logits = rationale_logits.reshape(-1, rationale_logits.size(-1))
        flat_rat_ids = rationale_ids.reshape(-1).to(rationale_logits.device)
        if (flat_rat_ids != self.ignore_index).any():
            rationale_loss = F.cross_entropy(
                flat_rat_logits,
                flat_rat_ids,
                ignore_index=self.ignore_index,
            )
        else:
            rationale_loss = torch.zeros((), device=rationale_logits.device)

        total = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        return total, {
            "label": label_loss.item(),
            "rationale": rationale_loss.item(),
            "total": total.item(),
            "task_type": "rationale_distill_step_by_step",
        }
