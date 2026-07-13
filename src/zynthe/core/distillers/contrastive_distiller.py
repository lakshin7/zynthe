"""Contrastive Representation Distillation (CRD).

Reference: Tian, Krishnan, Isola, "Contrastive Representation
Distillation", ICLR 2020.

The student projects its features through a learnable head
``g_s(h^s)``; the teacher uses a separate head ``g_t(h^t)``.
Both projections are L2-normalised and compared with cosine
similarity.  The CRD loss is an InfoNCE-style NLL of the positive
pair (student/teacher of the same sample) against all negatives:

    L_CRD = - log[exp(sim(z^s_i, z^t_i)/tau)
                   / (exp(sim(z^s_i, z^t_i)/tau)
                      + sum_{j!=i} exp(sim(sim)^i, z^t_j)/tau)
                      + M memory-bank terms)]

Configuration schema::

    contrastive:
      projection_dim: 128      # both heads map to this dim
      temperature: 0.07
      memory_bank_size: 256    # 0 = no bank (in-batch negatives only)
      student_layers: ["layers.3"]
      teacher_layers: ["encoder.layer.3"]
      supervised_weight: 1.0
      contrastive_weight: 1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


class _ProjectionHead(nn.Module):
    """Two-layer MLP: feature_dim -> hidden -> projection_dim, then L2 norm."""

    def __init__(
        self,
        in_dim: int,
        projection_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(projection_dim, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class ContrastiveDistiller(BaseDistiller):
    """CRD-style contrastive distillation."""

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

        crd_config = config.get("contrastive", {}) or {}
        self.crd_layers = crd_config.get("student_layers") or crd_config.get(
            "layers", ["layers.3"]
        )
        self.teacher_layers = crd_config.get("teacher_layers") or list(
            self.crd_layers
        )
        self.projection_dim = int(crd_config.get("projection_dim", 128))
        self.temperature = float(crd_config.get("temperature", 0.07))
        self.memory_bank_size = int(crd_config.get("memory_bank_size", 0))
        self.contrastive_weight = float(crd_config.get("contrastive_weight", 1.0))
        self.supervised_weight = float(crd_config.get("supervised_weight", 1.0))

        # Phase-0 strict_layer_match flag is honoured by BaseDistiller
        # but isn't an attribute on the class — we read it lazily from
        # config when registering hooks.
        self.strict_layer_match = bool(
            crd_config.get(
                "strict_layer_match",
                config.get("strict_layer_match", False),
            )
        )

        self._student_head: Optional[_ProjectionHead] = None
        self._teacher_head: Optional[_ProjectionHead] = None
        self._memory_bank: Optional[torch.Tensor] = None

        super().__init__(teacher, student, config=config, device=device, **kwargs)

        if self._teacher_head is not None:
            for p in self._teacher_head.parameters():
                p.requires_grad = True

    def _register_hooks(self) -> None:
        if not self.crd_layers:
            return

        teacher_modules = dict(self.teacher.named_modules())
        student_modules = dict(self.student.named_modules())

        missing: List[str] = []
        for s_layer, t_layer in zip(self.crd_layers, self.teacher_layers):
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
                "ContrastiveDistiller could not find requested layers",
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
                f"[ContrastiveDistiller] skipping unmatched layers: {missing}. "
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
            crd_loss = self._compute_crd(student_features, teacher_features)
            if torch.isfinite(crd_loss):
                total_loss = total_loss + self.contrastive_weight * crd_loss
                loss_dict["contrastive"] = crd_loss.item()
            else:
                loss_dict["contrastive"] = float("nan")

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def _compute_crd(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        s_layer = next(iter(student_features))
        try:
            i = self.crd_layers.index(s_layer)
            t_layer = self.teacher_layers[i]
        except ValueError:
            t_layer = next(iter(teacher_features))

        s_feat = student_features[s_layer]
        t_feat = teacher_features[t_layer]

        s_pool = self._pool(s_feat)
        t_pool = self._pool(t_feat)
        if s_pool.shape[-1] != t_pool.shape[-1]:
            t_pool, s_pool = self._align_shapes(t_pool, s_pool)

        if self._student_head is None:
            self._student_head = _ProjectionHead(
                in_dim=s_pool.shape[-1],
                projection_dim=self.projection_dim,
            ).to(self.device)
            self.add_module("_student_head", self._student_head)
        if self._teacher_head is None:
            self._teacher_head = _ProjectionHead(
                in_dim=t_pool.shape[-1],
                projection_dim=self.projection_dim,
            ).to(self.device)
            self.add_module("_teacher_head", self._teacher_head)

        z_s = self._student_head(s_pool)
        with torch.no_grad():
            z_t = self._teacher_head(t_pool).detach()

        if self.training and self.memory_bank_size > 0:
            self._update_memory_bank(z_t)

        B = z_s.shape[0]
        if B < 2:
            return torch.zeros((), device=self.device)

        sim_pos = (z_s * z_t).sum(dim=-1, keepdim=True)
        negatives = z_t
        if self._memory_bank is not None:
            negatives = torch.cat(
                [negatives, self._memory_bank.to(z_t.device)], dim=0
            )
        sim_neg = z_s @ negatives.T  # [B, B + M]
        # Mask only the in-batch diagonal of the *in-batch portion*
        # (columns 0..B-1). The memory-bank columns (B..B+M-1) stay
        # negative samples.
        diag_mask = torch.zeros(B, negatives.shape[0], device=z_s.device, dtype=torch.bool)
        diag_mask[:, :B] = torch.eye(B, device=z_s.device, dtype=torch.bool)
        sim_neg = sim_neg.masked_fill(diag_mask, float("-inf"))

        logits = torch.cat([sim_pos, sim_neg], dim=1) / max(self.temperature, 1e-6)
        labels = torch.zeros(B, dtype=torch.long, device=z_s.device)
        loss = F.cross_entropy(logits, labels)
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

    @staticmethod
    def _align_shapes(
        t_pool: torch.Tensor, s_pool: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if t_pool.shape[-1] == s_pool.shape[-1]:
            return t_pool, s_pool
        min_dim = min(t_pool.shape[-1], s_pool.shape[-1])
        return t_pool[..., :min_dim], s_pool[..., :min_dim]

    def _update_memory_bank(self, z_t: torch.Tensor) -> None:
        bank_size = self.memory_bank_size
        if bank_size <= 0:
            return
        if self._memory_bank is None or self._memory_bank.shape[0] < bank_size:
            new = z_t.detach().clone()
            if self._memory_bank is None:
                self._memory_bank = new
            else:
                self._memory_bank = torch.cat(
                    [self._memory_bank, new], dim=0
                )[-bank_size:]
            return

        combined = torch.cat([self._memory_bank, z_t.detach()], dim=0)
        if combined.shape[0] > bank_size:
            combined = combined[-bank_size:]
        self._memory_bank = combined

    def remove_hooks(self) -> None:
        super().remove_hooks()
        self._student_head = None
        self._teacher_head = None
        self._memory_bank = None
