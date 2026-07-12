"""Diffusion model adapter — UNet-style noise predictors (Stable
Diffusion / SDXL unets).  Module-name based detection on
``down_blocks`` / ``mid_block`` / ``up_blocks``.

We don't import ``diffusers`` directly — the adapter operates on the
``nn.Module`` interface only.  Diffusers models expose those submodule
names regardless of which optional extras are installed.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from zynthe.core.utils.device_utils import normalize_model_output

from .base_adapter import ModelAdapter


class DiffusionAdapter(ModelAdapter):
    """Adapter for diffusion UNets and similar denoising models."""

    modality = "diffusion"

    #: Module patterns of UNet-style blocks (Stable Diffusion layout).
    _BLOCK_PATTERNS = [
        re.compile(r"^down_blocks\.\d+\.?(\w+)?$"),
        re.compile(r"^up_blocks\.\d+\.?(\w+)?$"),
        re.compile(r"^mid_block$"),
    ]

    #: Forward-kwarg hints accepted by diffusion UNets (sample, timestep,
    #: encoder_hidden_states for class-conditioned variants).
    _COMMON_KEYS = frozenset(
        {
            "sample",
            "timestep",
            "encoder_hidden_states",
            "added_cond_kwargs",
            "return_dict",
            "cross_attention_kwargs",
        }
    )

    def __init__(self) -> None:
        self._forward_params_cache: Dict[int, frozenset] = {}

    def supports_model(self, model: nn.Module) -> bool:
        """UNet has at least one ``down_blocks.i`` module and ``up_blocks.i``
        module (SD-style).
        """
        module_names = {n for n, _ in model.named_modules()}
        has_down = any(name.startswith("down_blocks.") for name in module_names)
        has_up = any(name.startswith("up_blocks.") for name in module_names)
        has_mid = any(name.startswith("mid_block") for name in module_names)
        return (has_down and has_up) or has_mid

    def prepare_batch(
        self,
        batch: Dict[str, Any],
        model: nn.Module,
    ) -> Dict[str, Any]:
        allowed = self._get_forward_params(model)
        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        # Diffusion UNets return a ``UNet2DConditionOutput`` dataclass.
        # We normalize through the generic helper; downstream code reads
        # ``.sample`` (the predicted noise).
        if hasattr(raw_output, "sample"):
            sample = raw_output.sample
            if isinstance(sample, torch.Tensor):
                return {
                    "logits": sample,
                    "hidden_states": getattr(raw_output, "down_block_res_samples", None),
                    "attentions": None,
                    "loss": None,
                }
        return normalize_model_output(raw_output)

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        layers: List[str] = []
        for name, _ in model.named_modules():
            if not name:
                continue
            for pattern in self._BLOCK_PATTERNS:
                if pattern.match(name):
                    layers.append(name)
                    break
        return sorted(layers)

    def align_dimensions(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        import torch.nn.functional as F

        aligned_t = dict(teacher_features)
        aligned_s: Dict[str, torch.Tensor] = {}
        for key, t_feat in teacher_features.items():
            if key not in student_features:
                continue
            s_feat = student_features[key]
            if s_feat.shape == t_feat.shape:
                aligned_s[key] = s_feat
                continue
            if s_feat.dim() >= 2 and t_feat.shape[1:] != s_feat.shape[1:]:
                try:
                    s_feat = F.interpolate(
                        s_feat, size=t_feat.shape[1:], mode="nearest"
                    )
                except Exception:
                    pass
            aligned_s[key] = s_feat
        for key, val in student_features.items():
            if key not in aligned_s:
                aligned_s[key] = val
        return aligned_t, aligned_s

    def _get_forward_params(self, model: nn.Module) -> frozenset:
        model_id = id(model)
        if model_id not in self._forward_params_cache:
            try:
                sig = inspect.signature(model.forward)
                self._forward_params_cache[model_id] = frozenset(sig.parameters.keys())
            except (ValueError, TypeError):
                self._forward_params_cache[model_id] = self._COMMON_KEYS
        return self._forward_params_cache[model_id]
