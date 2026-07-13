"""Relational Knowledge Distillation (PKT).

Reference: Park, Kim, Lee, Lee, "Relational Knowledge
Distillation", CVPR 2019.

While contrastive distillation (CRD) matches *individual* student /
teacher features to a prototype, PKT matches the *second-order
structure* of the batch — i.e. the cosine-similarity matrix between
samples in the batch:

    L_pair = (1 / N^2) * sum_{i,j}
                  ( cos(z^s_i, z^s_j) - cos(z^t_i, z^t_j) )^2

where ``z_*`` are the (pooled, L2-normalised) student / teacher
features, and the sum runs over all pairs in the batch.  The cosine
distance per pair replaces the raw dot-product because it is bounded
in [-1, 1] and therefore more stable under different feature scales.

Configuration schema::

    relational:
      student_layers: ["layers.3"]
      teacher_layers: ["encoder.layer.3"]
      normalize: true        # L2-normalise before computing cosine
      relational_weight: 1.0
      supervised_weight: 1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


def _pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise cosine-similarity matrix for a batch of
    L2-normalised features.

    Args:
        features: ``(B, C)`` tensor.

    Returns:
        ``(B, B)`` symmetric matrix with 1's on the diagonal.
    """
    if features.dim() != 2:
        raise ValueError(
            f"Expected 2-D features (B, C); got shape {tuple(features.shape)}"
        )
    feats_norm = F.normalize(features, p=2, dim=-1)
    return feats_norm @ feats_norm.T


class RelationalDistiller(BaseDistiller):
    """PKT-style relational distillation."""

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

        rel_config = config.get("relational", {}) or {}
        self.rel_layers = rel_config.get("student_layers") or rel_config.get(
            "layers", ["layers.3"]
        )
        self.teacher_layers = rel_config.get("teacher_layers") or list(
            self.rel_layers
        )
        self.normalize = bool(rel_config.get("normalize", True))
        self.relational_weight = float(rel_config.get("relational_weight", 1.0))
        self.supervised_weight = float(rel_config.get("supervised_weight", 1.0))

        self.strict_layer_match = bool(
            rel_config.get(
                "strict_layer_match",
                config.get("strict_layer_match", False),
            )
        )

        super().__init__(teacher, student, config=config, device=device, **kwargs)

    def _register_hooks(self) -> None:
        if not self.rel_layers:
            return

        teacher_modules = dict(self.teacher.named_modules())
        student_modules = dict(self.student.named_modules())

        missing: List[str] = []
        for s_layer, t_layer in zip(self.rel_layers, self.teacher_layers):
            t_ok = t_layer in teacher_modules
            s_ok = s_layer in student_modules
            if not t_ok:
                missing.append(f"teacher:{t_layer}")
            if not s_ok:
                missing.append(f"student:{s_layer}")
            if t_ok:
                self._hook_handles.append(
                    teacher_modules[t_layer].register_forward_hook(
                        self._get_teacher_hook(t_layer)
                    )
                )
            if s_ok:
                self._hook_handles.append(
                    student_modules[s_layer].register_forward_hook(
                        self._get_student_hook(s_layer)
                    )
                )

        if missing and self.strict_layer_match:
            from zynthe.core.utils import ConfigError, format_missing_layers

            raise ConfigError(
                "RelationalDistiller could not find requested layers",
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
                f"[RelationalDistiller] skipping unmatched layers: {missing}. "
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
            loss_dict["task_type"] = "classification"
            ce = F.cross_entropy(student_logits, targets)
            total_loss = total_loss + self.supervised_weight * ce
            loss_dict["supervised"] = ce.item()

        if student_features and teacher_features:
            rel_loss = self._compute_pair(student_features, teacher_features)
            if torch.isfinite(rel_loss):
                total_loss = total_loss + self.relational_weight * rel_loss
                loss_dict["relational"] = rel_loss.item()
            else:
                loss_dict["relational"] = float("nan")

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def _compute_pair(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        s_layer = next(iter(student_features))
        try:
            i = self.rel_layers.index(s_layer)
            t_layer = self.teacher_layers[i]
        except ValueError:
            t_layer = next(iter(teacher_features))

        s_feat = self._pool(student_features[s_layer])
        t_feat = self._pool(teacher_features[t_layer])

        if s_feat.shape[-1] != t_feat.shape[-1]:
            # Different feature widths: project student linearly to
            # match teacher.  Lazy because the dimension only matters
            # at runtime (not config-time) and we want this distiller to
            # be config-light.
            min_dim = min(s_feat.shape[-1], t_feat.shape[-1])
            s_feat = s_feat[..., :min_dim]
            t_feat = t_feat[..., :min_dim]

        # Teacher detached — we don't propagate gradients through it
        # via the relational loss (the teacher backbone is frozen).
        if self.normalize:
            s_sim = _pairwise_cosine(s_feat)
            t_sim = _pairwise_cosine(t_feat.detach())
        else:
            s_sim = s_feat @ s_feat.T
            t_sim = t_feat.detach() @ t_feat.detach().T

        # MSE between the two matrices — symmetric, captures pairwise
        # distance disagreement.
        loss = F.mse_loss(s_sim, t_sim)
        return loss

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
