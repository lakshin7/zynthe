"""
Vision Model Adapter
======================

Handles pure vision models:

- **CNNs**: ResNet, EfficientNet, ConvNeXt, RegNet
- **Transformers**: ViT, DeiT, Swin, BEiT, CvT

Batch keys: ``pixel_values``, ``labels``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_adapter import ModelAdapter
from core.utils.device_utils import normalize_model_output


class VisionModelAdapter(ModelAdapter):
    """Adapter for vision-only models (CNNs and ViTs)."""

    modality = "vision"

    # Module name patterns for hookable layers.
    _LAYER_PATTERNS = [
        # ViT / DeiT / BEiT
        re.compile(r"^(vit\.encoder\.layer\.\d+)$"),
        re.compile(r"^(encoder\.layer\.\d+)$"),
        re.compile(r"^(deit\.encoder\.layer\.\d+)$"),
        re.compile(r"^(beit\.encoder\.layer\.\d+)$"),
        # Swin
        re.compile(r"^(swin\.encoder\.layers\.\d+\.blocks\.\d+)$"),
        # ResNet
        re.compile(r"^(resnet\.(layer[1-4]))$"),
        re.compile(r"^(layer[1-4])$"),
        # EfficientNet
        re.compile(r"^(efficientnet\.encoder\.blocks\.\d+)$"),
        re.compile(r"^(features\.\d+)$"),
        # ConvNeXt
        re.compile(r"^(convnext\.encoder\.stages\.\d+)$"),
    ]

    def __init__(self) -> None:
        self._projections: nn.ModuleDict = nn.ModuleDict()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: Dict[str, Any],
        model: nn.Module,
    ) -> Dict[str, Any]:
        """Keep only vision-relevant keys."""
        allowed = {"pixel_values", "labels", "return_dict",
                    "output_attentions", "output_hidden_states"}
        out = {k: v for k, v in batch.items() if k in allowed}

        # Some HF vision models want `pixel_values`, others take raw images
        if "pixel_values" not in out and "image" in batch:
            out["pixel_values"] = batch["image"]

        return out

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise vision output.

        Adds ``patch_embeddings`` when the model is a ViT and the
        hidden states include patch tokens.
        """
        result = normalize_model_output(raw_output)

        # Extra: extract CLS token from ViT hidden states
        if result.get("hidden_states") and isinstance(result["hidden_states"], (tuple, list)):
            last_hidden = result["hidden_states"][-1]
            if isinstance(last_hidden, torch.Tensor) and last_hidden.dim() == 3:
                # (batch, seq_len, hidden) — first token is CLS
                result["cls_token"] = last_hidden[:, 0, :]
                result["patch_embeddings"] = last_hidden[:, 1:, :]

        return result

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Discover conv stages or ViT encoder blocks."""
        layers: List[str] = []
        for name, _ in model.named_modules():
            for pattern in self._LAYER_PATTERNS:
                if pattern.match(name):
                    layers.append(name)
                    break
        return sorted(layers)

    def align_dimensions(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Align spatial + channel dimensions.

        - **Spatial**: bilinear interpolation to match (H, W).
        - **Channel**: learned 1×1 conv (CNN) or linear projection (ViT).
        """
        aligned_teacher: Dict[str, torch.Tensor] = {}
        aligned_student: Dict[str, torch.Tensor] = {}

        for key in teacher_features:
            t_feat = teacher_features[key]
            if key not in student_features:
                aligned_teacher[key] = t_feat
                continue

            s_feat = student_features[key]
            aligned_teacher[key] = t_feat

            # 4-D feature maps (B, C, H, W)
            if t_feat.dim() == 4 and s_feat.dim() == 4:
                # Spatial alignment
                if t_feat.shape[2:] != s_feat.shape[2:]:
                    s_feat = F.interpolate(
                        s_feat, size=t_feat.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                # Channel alignment
                if t_feat.shape[1] != s_feat.shape[1]:
                    proj = self._get_or_create_conv_projection(
                        key, s_feat.shape[1], t_feat.shape[1], s_feat.device,
                    )
                    s_feat = proj(s_feat)
                aligned_student[key] = s_feat

            # 3-D sequences (B, seq, hidden) — ViT
            elif t_feat.dim() == 3 and s_feat.dim() == 3:
                # Sequence length alignment (pad/truncate)
                if t_feat.shape[1] != s_feat.shape[1]:
                    s_feat = self._align_seq_length(s_feat, t_feat.shape[1])
                # Hidden dim alignment
                if t_feat.shape[-1] != s_feat.shape[-1]:
                    proj = self._get_or_create_linear_projection(
                        key, s_feat.shape[-1], t_feat.shape[-1], s_feat.device,
                    )
                    s_feat = proj(s_feat)
                aligned_student[key] = s_feat
            else:
                aligned_student[key] = s_feat

        for key in student_features:
            if key not in aligned_student:
                aligned_student[key] = student_features[key]

        return aligned_teacher, aligned_student

    def supports_model(self, model: nn.Module) -> bool:
        """Heuristic: model has a ``pixel_values`` parameter or common vision modules."""
        # Check forward signature
        try:
            import inspect
            sig = inspect.signature(model.forward)
            if "pixel_values" in sig.parameters:
                return True
        except (ValueError, TypeError):
            pass

        # Check for known vision sub-modules
        module_names = {n for n, _ in model.named_modules()}
        vision_markers = {"patch_embed", "patch_embedding", "cls_token",
                          "features", "layer1", "conv1"}
        return bool(module_names & vision_markers)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _align_seq_length(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad or truncate sequence dimension (dim=1) to *target_len*."""
        current = tensor.shape[1]
        if current == target_len:
            return tensor
        if current > target_len:
            return tensor[:, :target_len, :]
        pad = torch.zeros(
            tensor.shape[0], target_len - current, tensor.shape[2],
            device=tensor.device, dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=1)

    def _get_or_create_conv_projection(
        self, key: str, in_ch: int, out_ch: int, device: torch.device,
    ) -> nn.Conv2d:
        """1×1 conv for channel alignment."""
        proj_key = f"conv_{key}_{in_ch}_{out_ch}"
        if proj_key not in self._projections:
            proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight)
            self._projections[proj_key] = proj
        return self._projections[proj_key]

    def _get_or_create_linear_projection(
        self, key: str, in_dim: int, out_dim: int, device: torch.device,
    ) -> nn.Linear:
        """Linear projection for hidden-dim alignment."""
        proj_key = f"lin_{key}_{in_dim}_{out_dim}"
        if proj_key not in self._projections:
            proj = nn.Linear(in_dim, out_dim, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight)
            self._projections[proj_key] = proj
        return self._projections[proj_key]
