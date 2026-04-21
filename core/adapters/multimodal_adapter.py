"""
Multimodal Model Adapter
===========================

Handles dual-encoder contrastive models:

- **CLIP** (OpenAI, open_clip)
- **SigLIP**
- **BLIP / BLIP-2** (image-text matching encoder)
- **ALIGN**

These models expose *separate* text and vision encoders plus a
cross-modal projection.  The adapter normalises both branches and
provides per-encoder hookable layer lists.

Batch keys: ``pixel_values``, ``input_ids``, ``attention_mask``,
``labels``, ``return_loss``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base_adapter import ModelAdapter
from core.utils.device_utils import normalize_model_output


class MultimodalModelAdapter(ModelAdapter):
    """Adapter for dual-encoder multimodal models (CLIP, BLIP, etc.)."""

    modality = "multimodal"

    _TEXT_LAYER_PATTERNS = [
        re.compile(r"^text_model\.encoder\.layers?\.\d+$"),
        re.compile(r"^text_encoder\.encoder\.layer\.\d+$"),
    ]

    _VISION_LAYER_PATTERNS = [
        re.compile(r"^vision_model\.encoder\.layers?\.\d+$"),
        re.compile(r"^visual\.transformer\.resblocks\.\d+$"),
        re.compile(r"^vision_encoder\.encoder\.layer\.\d+$"),
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
        """Keep only keys accepted by a multimodal model."""
        allowed = {
            "pixel_values", "input_ids", "attention_mask",
            "return_loss", "labels", "return_dict",
            "output_attentions", "output_hidden_states",
        }

        # Introspect the forward signature for extra keys
        try:
            sig = inspect.signature(model.forward)
            allowed |= set(sig.parameters.keys())
        except (ValueError, TypeError):
            pass

        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise dual-encoder output.

        In addition to the standard ``logits`` / ``hidden_states`` /
        ``attentions``, multimodal models expose:

        - ``image_embeds`` / ``text_embeds``  — projected embeddings
        - ``logits_per_image`` / ``logits_per_text`` — similarity logits
        """
        result = normalize_model_output(raw_output)

        # Dual-encoder specific outputs
        for attr in (
            "image_embeds", "text_embeds",
            "logits_per_image", "logits_per_text",
            "image_features", "text_features",
        ):
            val = (
                raw_output.get(attr) if isinstance(raw_output, dict)
                else getattr(raw_output, attr, None)
            )
            if val is not None:
                result[attr] = val

        # If the model doesn't set ``logits`` but has similarity logits,
        # use ``logits_per_image`` as the canonical logits.
        if result.get("logits") is None and result.get("logits_per_image") is not None:
            result["logits"] = result["logits_per_image"]

        return result

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Return text-encoder and vision-encoder layers separately.

        Layers are sorted and prefixed so the caller can tell them
        apart.
        """
        layers: List[str] = []
        for name, _ in model.named_modules():
            for pattern in self._TEXT_LAYER_PATTERNS + self._VISION_LAYER_PATTERNS:
                if pattern.match(name):
                    layers.append(name)
                    break
        return sorted(layers)

    def align_dimensions(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Align embed dimensions via linear projection.

        Multimodal features are typically 2-D (batch, embed_dim) or
        3-D (batch, seq, embed_dim).  We handle both.
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

            if t_feat.shape[-1] != s_feat.shape[-1]:
                proj = self._get_or_create_projection(
                    key, s_feat.shape[-1], t_feat.shape[-1], s_feat.device,
                )
                aligned_student[key] = proj(s_feat)
            else:
                aligned_student[key] = s_feat

        for key in student_features:
            if key not in aligned_student:
                aligned_student[key] = student_features[key]

        return aligned_teacher, aligned_student

    def supports_model(self, model: nn.Module) -> bool:
        """Heuristic: model has *both* ``text_model`` and ``vision_model``."""
        module_names = {n for n, _ in model.named_modules()}
        has_text = any(
            n.startswith("text_model") or n.startswith("text_encoder")
            for n in module_names
        )
        has_vision = any(
            n.startswith("vision_model") or n.startswith("visual")
            or n.startswith("vision_encoder")
            for n in module_names
        )
        return has_text and has_vision

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_projection(
        self, key: str, in_dim: int, out_dim: int, device: torch.device,
    ) -> nn.Linear:
        proj_key = f"proj_{key}_{in_dim}_{out_dim}"
        if proj_key not in self._projections:
            proj = nn.Linear(in_dim, out_dim, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight)
            self._projections[proj_key] = proj
        return self._projections[proj_key]
