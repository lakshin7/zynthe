"""Attention-to-Embedding (Translator Projection) Distillation.

Reference: Zynthé report §213.

The student projects its output through a learnable translator MLP
that maps student features into the teacher's hidden dimension.  The
projection is then compared (MSE) to the teacher's corresponding
hidden state.

    L_proj = MSE( g_s(h^s), h^t )

where ``g_s`` is a learnable projection (the translator).  This is a
lighter-weight alternative to hint-based distillation: the translator
injects a small "interpretability adapter" between student and teacher
spaces.

Configuration schema::

    projection:
      student_layers: ["layers.3"]
      teacher_layers: ["encoder.layer.3"]
      projection_hidden: 512   # translator hidden dim
      projection_weight: 1.0
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


class _Translator(nn.Module):
    """Two-layer MLP translator: in_dim -> hidden -> out_dim.

    Storing as a Module so the optimiser finds it via
    ``student.parameters()``.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionDistiller(BaseDistiller):
    """Translator-projection distillation."""

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

        proj_config = config.get("projection", {}) or {}
        self.proj_layers = proj_config.get("student_layers") or proj_config.get(
            "layers", ["layers.3"]
        )
        self.teacher_layers = proj_config.get("teacher_layers") or list(
            self.proj_layers
        )
        self.projection_hidden = int(proj_config.get("projection_hidden", 512))
        self.projection_weight = float(proj_config.get("projection_weight", 1.0))
        self.supervised_weight = float(proj_config.get("supervised_weight", 1.0))

        self.strict_layer_match = bool(
            proj_config.get(
                "strict_layer_match",
                config.get("strict_layer_match", False),
            )
        )

        self._translator: Optional[_Translator] = None

        super().__init__(teacher, student, config=config, device=device, **kwargs)

    def _register_hooks(self) -> None:
        if not self.proj_layers:
            return

        teacher_modules = dict(self.teacher.named_modules())
        student_modules = dict(self.student.named_modules())

        missing: List[str] = []
        for s_layer, t_layer in zip(self.proj_layers, self.teacher_layers):
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
                "ProjectionDistiller could not find requested layers",
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
                f"[ProjectionDistiller] skipping unmatched layers: {missing}. "
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
            proj_loss = self._compute_projection(
                student_features, teacher_features
            )
            if torch.isfinite(proj_loss):
                total_loss = total_loss + self.projection_weight * proj_loss
                loss_dict["projection"] = proj_loss.item()
            else:
                loss_dict["projection"] = float("nan")

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def _compute_projection(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        s_layer = next(iter(student_features))
        try:
            i = self.proj_layers.index(s_layer)
            t_layer = self.teacher_layers[i]
        except ValueError:
            t_layer = next(iter(teacher_features))

        s_feat = self._pool(student_features[s_layer])
        t_feat = self._pool(teacher_features[t_layer])

        if self._translator is None:
            self._translator = _Translator(
                in_dim=s_feat.shape[-1],
                out_dim=t_feat.shape[-1],
                hidden_dim=self.projection_hidden,
            ).to(self.device)
            self.add_module("_translator", self._translator)

        translated = self._translator(s_feat)
        # Teacher detached — translator only learns to match a fixed
        # target.
        target = t_feat.detach()
        return F.mse_loss(translated, target)

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
        self._translator = None
