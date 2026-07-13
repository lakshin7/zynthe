"""Intermediate Classifier Distillation (Deep Supervision / Aux Heads).

Reference: Zynthé report §215-217.

Attach a small linear classifier to selected intermediate student
layers; sum their cross-entropy losses on the labels:

    L_aux = sum_l CE(g_l(h^s_l), labels)

This is "deep supervision": each aux head gives the student an extra
gradient signal at depth, which the report expects to accelerate
convergence on smaller GLUE-style datasets.

Configuration schema::

    aux_head:
      student_layers: ["layers.1", "layers.2", "layers.3"]
      num_classes: 2
      head_hidden: 128    # hidden dim of each aux head's MLP
      aux_weight: 0.3     # scaling for the summed aux loss
      supervised_weight: 1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


class _AuxHead(nn.Module):
    """Auxiliary classifier: feature → hidden → num_classes."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AuxHeadDistiller(BaseDistiller):
    """Deep-supervision distillation with intermediate classifiers."""

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

        aux_config = config.get("aux_head", {}) or {}
        self.aux_layers = aux_config.get(
            "student_layers", ["layers.1", "layers.2", "layers.3"]
        )
        self.num_classes = int(aux_config.get("num_classes", 2))
        self.head_hidden = int(aux_config.get("head_hidden", 128))
        self.aux_weight = float(aux_config.get("aux_weight", 0.3))
        self.supervised_weight = float(aux_config.get("supervised_weight", 1.0))

        self.strict_layer_match = bool(
            aux_config.get(
                "strict_layer_match",
                config.get("strict_layer_match", False),
            )
        )

        # Lazily-created aux heads keyed by layer name.
        self._aux_heads: Dict[str, _AuxHead] = {}

        super().__init__(teacher, student, config=config, device=device, **kwargs)

    def _register_hooks(self) -> None:
        if not self.aux_layers:
            return

        student_modules = dict(self.student.named_modules())

        missing: List[str] = []
        for s_layer in self.aux_layers:
            if s_layer not in student_modules:
                missing.append(f"student:{s_layer}")
            else:
                self._hook_handles.append(
                    student_modules[s_layer].register_forward_hook(
                        self._get_student_hook(s_layer)
                    )
                )

        if missing and self.strict_layer_match:
            from zynthe.core.utils import ConfigError, format_missing_layers

            raise ConfigError(
                "AuxHeadDistiller could not find requested layers",
                context={
                    "missing": missing,
                    "missing_summary": format_missing_layers(missing),
                    "hint": (
                        "Use named_modules() to find valid layer names, or "
                        "set strict_layer_match=False."
                    ),
                },
            )
        elif missing:
            import warnings

            warnings.warn(
                f"[AuxHeadDistiller] skipping unmatched layers: {missing}. "
                f"Set strict_layer_match=True to raise instead.",
                stacklevel=2,
            )

    def compute_loss(
        self,
        student_outputs: Any,
        teacher_outputs: Any,
        targets: Optional[torch.Tensor] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        total_loss = torch.zeros((), device=self.device)
        loss_dict: Dict[str, Any] = {}

        if targets is not None:
            student_logits = self._extract_logits_tensor(student_outputs)
            ce = F.cross_entropy(student_logits, targets)
            total_loss = total_loss + self.supervised_weight * ce
            loss_dict["supervised"] = ce.item()
        else:
            loss_dict["task_type"] = "classification (aux-only)"

        # Aux heads — always run, even without `targets` (we'd need
        # labels for them, but the distiller is intentionally
        # supervised-only).
        if student_features and targets is not None:
            aux_loss = self._compute_aux(student_features, targets)
            if torch.isfinite(aux_loss):
                total_loss = total_loss + self.aux_weight * aux_loss
                loss_dict["aux"] = aux_loss.item()
                loss_dict["task_type"] = "classification"
            else:
                loss_dict["aux"] = float("nan")

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def _compute_aux(
        self,
        student_features: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.zeros((), device=self.device)
        n = 0
        for layer_name in self.aux_layers:
            if layer_name not in student_features:
                continue
            feat = student_features[layer_name]
            pooled = self._pool(feat)
            if layer_name not in self._aux_heads:
                head = _AuxHead(
                    in_dim=pooled.shape[-1],
                    num_classes=self.num_classes,
                    hidden_dim=self.head_hidden,
                ).to(self.device)
                self._aux_heads[layer_name] = head
                self.add_module(f"aux_head_{layer_name.replace('.', '_')}", head)
            logits = self._aux_heads[layer_name](pooled)
            loss = F.cross_entropy(logits, targets)
            total = total + loss
            n += 1
        if n == 0:
            return torch.zeros((), device=self.device)
        return total / n  # mean over aux heads

    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 2:
            return feat
        if feat.dim() == 3:
            return feat.mean(dim=1)
        if feat.dim() == 4:
            return feat.mean(dim=(2, 3))
        B = feat.shape[0]
        return feat.reshape(B, -1)

    def remove_hooks(self) -> None:
        super().remove_hooks()
        self._aux_heads = {}
