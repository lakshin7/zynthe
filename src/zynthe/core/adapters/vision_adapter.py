"""
Vision Model Adapter
======================

Handles vision transformer and CNN-based image classification models:

- **Vision Transformers**: ViT, BEiT, DeiT, Swin Transformer
- **CNNs**: ConvNeXt, ResNet, EfficientNet
- **Hybrid**: CvT, CoAtNet

Batch keys: ``pixel_values``, ``labels``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_adapter import ModelAdapter
from zynthe.core.utils.device_utils import normalize_model_output


class VisionModelAdapter(ModelAdapter):
    """Adapter for vision classification / feature-extraction models."""

    modality = "vision"

    # Module name patterns for hookable transformer/CNN layers.
    _LAYER_PATTERNS = [
        re.compile(r"^vit\.encoder\.layer\.(\d+)$"),          # ViT (HF)
        re.compile(r"^beit\.encoder\.layer\.(\d+)$"),          # BEiT
        re.compile(r"^deit\.encoder\.layer\.(\d+)$"),          # DeiT
        re.compile(r"^swin\.encoder\.layers\.(\d+)$"),         # Swin
        re.compile(r"^convnext\.encoder\.stages\.(\d+)$"),     # ConvNeXt
        re.compile(r"^encoder\.layer\.(\d+)$"),                # generic ViT
        re.compile(r"^model\.encoder\.layers\.(\d+)$"),        # wrapped ViT
        re.compile(r"^features\.(\d+)$"),                      # EfficientNet / MobileNet
        re.compile(r"^layer(\d+)$"),                           # ResNet
    ]

    # Keys accepted by vision model forward methods.
    _COMMON_KEYS = frozenset({
        "pixel_values",
        "labels",
        "head_mask",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "interpolate_pos_encoding",
        "bool_masked_pos",
    })

    def __init__(self) -> None:
        self._forward_params_cache: Dict[int, frozenset] = {}
        self._projections: nn.ModuleDict = nn.ModuleDict()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: Dict[str, Any],
        model: nn.Module,
    ) -> Dict[str, Any]:
        """Filter batch to vision-model-compatible keys."""
        allowed = self._get_forward_params(model)
        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise vision model output into a standard dict."""
        result = normalize_model_output(raw_output)

        # Vision-specific fields
        for attr in ("pooler_output", "last_hidden_state", "cls_token"):
            val = (
                raw_output.get(attr) if isinstance(raw_output, dict)
                else getattr(raw_output, attr, None)
            )
            if val is not None:
                result[attr] = val

        return result

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Discover vision encoder layers by name pattern matching."""
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
        """Project hidden dimensions via learned linear layers.

        For vision models, feature tensors are typically
        ``(B, num_patches, hidden_dim)`` or ``(B, C, H, W)`` for CNNs.
        We project along the last dimension.
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

            if t_feat.ndim == 4 and s_feat.ndim == 4:
                aligned = s_feat
                if aligned.shape[-2:] != t_feat.shape[-2:]:
                    aligned = F.interpolate(
                        aligned,
                        size=t_feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if aligned.shape[1] != t_feat.shape[1]:
                    proj = self._get_or_create_channel_projection(
                        key, aligned.shape[1], t_feat.shape[1], aligned.device,
                    )
                    aligned = proj(aligned)
                aligned_student[key] = aligned
            elif t_feat.shape[-1] != s_feat.shape[-1]:
                proj = self._get_or_create_projection(
                    key, s_feat.shape[-1], t_feat.shape[-1], s_feat.device,
                )
                aligned_student[key] = proj(s_feat)
            else:
                aligned_student[key] = s_feat

        # Keep student-only keys as-is
        for key in student_features:
            if key not in aligned_student:
                aligned_student[key] = student_features[key]

        return aligned_teacher, aligned_student

    def supports_model(self, model: nn.Module) -> bool:
        """Heuristic: model accepts ``pixel_values`` but not ``input_ids``."""
        params = self._get_forward_params(model)
        has_pixels = "pixel_values" in params
        has_text = "input_ids" in params
        # Pure vision = has pixel_values but NOT input_ids
        return has_pixels and not has_text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_forward_params(self, model: nn.Module) -> frozenset:
        """Cache-friendly forward-parameter discovery."""
        model_id = id(model)
        if model_id not in self._forward_params_cache:
            try:
                sig = inspect.signature(model.forward)
                self._forward_params_cache[model_id] = frozenset(
                    sig.parameters.keys()
                )
            except (ValueError, TypeError):
                self._forward_params_cache[model_id] = self._COMMON_KEYS
        return self._forward_params_cache[model_id]

    def _get_or_create_projection(
        self, key: str, in_dim: int, out_dim: int, device: torch.device,
    ) -> nn.Linear:
        """Lazily create a linear projection for dimension alignment."""
        proj_key = f"proj_{key}_{in_dim}_{out_dim}"
        if proj_key not in self._projections:
            proj = nn.Linear(in_dim, out_dim, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight)
            self._projections[proj_key] = proj
        return self._projections[proj_key]

    def _get_or_create_channel_projection(
        self, key: str, in_channels: int, out_channels: int, device: torch.device,
    ) -> nn.Conv2d:
        """Lazily create a 1x1 conv projection for channel alignment."""
        proj_key = f"conv_{key}_{in_channels}_{out_channels}"
        if proj_key not in self._projections:
            proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight)
            self._projections[proj_key] = proj
        return self._projections[proj_key]
