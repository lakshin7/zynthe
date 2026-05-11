"""
Multimodal Model Adapter
==========================

Handles dual-encoder and fusion-based multimodal models:

- **Dual Encoders**: CLIP, SigLIP, ALIGN, BLIP
- **Fusion Models**: FLAVA, BridgeTower, X-CLIP
- **Audio-Visual**: (future extension point)

These models process both text and image inputs through separate
encoders and produce aligned embeddings.

Batch keys: ``input_ids``, ``attention_mask``, ``pixel_values``,
``labels``, ``return_loss``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base_adapter import ModelAdapter
from zynthe.core.utils.device_utils import normalize_model_output


class MultimodalModelAdapter(ModelAdapter):
    """Adapter for dual-encoder multimodal models (CLIP, SigLIP, etc.)."""

    modality = "multimodal"

    # Module name patterns for text encoder layers.
    _TEXT_PATTERNS = [
        re.compile(r"^text_model\.encoder\.layers\.(\d+)$"),  # CLIP
        re.compile(r"^text_encoder\.encoder\.layer\.(\d+)$"),  # BLIP
        re.compile(r"^text_model\.transformer\.resblocks\.(\d+)$"),  # OpenCLIP
    ]

    # Module name patterns for vision encoder layers.
    _VISION_PATTERNS = [
        re.compile(r"^vision_model\.encoder\.layers\.(\d+)$"),  # CLIP / SigLIP
        re.compile(r"^visual_encoder\.blocks\.(\d+)$"),  # BLIP
        re.compile(r"^vision_model\.transformer\.resblocks\.(\d+)$"),  # OpenCLIP
    ]

    # Module name patterns for cross-attention / fusion layers.
    _FUSION_PATTERNS = [
        re.compile(r"^cross_attention\.layers\.(\d+)$"),
        re.compile(r"^multimodal_encoder\.layer\.(\d+)$"),
        re.compile(r"^fusion_module\.layers\.(\d+)$"),
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
        """Filter batch for multimodal forward signature.

        Multimodal models typically accept both text and vision inputs.
        We introspect the forward method and fall back to a generous
        allowlist.
        """
        try:
            sig = inspect.signature(model.forward)
            allowed = set(sig.parameters.keys())
        except (ValueError, TypeError):
            allowed = set()

        # Generous fallback for multimodal models
        allowed |= {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "labels",
            "return_loss",
            "return_dict",
            "output_attentions",
            "output_hidden_states",
            "token_type_ids",
            "position_ids",
        }

        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise multimodal output.

        Beyond logits/hidden_states, multimodal models expose
        ``text_embeds``, ``image_embeds``, ``logits_per_image``,
        ``logits_per_text``, etc.
        """
        result = normalize_model_output(raw_output)

        # Multimodal-specific fields
        for attr in (
            "text_embeds",
            "image_embeds",
            "text_model_output",
            "vision_model_output",
            "logits_per_image",
            "logits_per_text",
            "itm_score",
        ):
            val = (
                raw_output.get(attr)
                if isinstance(raw_output, dict)
                else getattr(raw_output, attr, None)
            )
            if val is not None:
                result[attr] = val

        if result.get("logits") is None:
            logits = result.get("logits_per_image")
            result["logits"] = logits if logits is not None else result.get("itm_score")

        return result

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Return layers from both text and vision encoders plus fusion layers.

        Returns:
            Sorted list of hookable module names.
        """
        layers: List[str] = []
        all_patterns = self._TEXT_PATTERNS + self._VISION_PATTERNS + self._FUSION_PATTERNS
        for name, _ in model.named_modules():
            for pattern in all_patterns:
                if pattern.match(name):
                    layers.append(name)
                    break
        return sorted(layers)

    def align_dimensions(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Align features from text and vision encoders.

        Multimodal models often have different hidden sizes for their
        text and vision encoders. We project student features into
        teacher dimensionality per feature key.
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
                    key,
                    s_feat.shape[-1],
                    t_feat.shape[-1],
                    s_feat.device,
                )
                aligned_student[key] = proj(s_feat)
            else:
                aligned_student[key] = s_feat

        for key in student_features:
            if key not in aligned_student:
                aligned_student[key] = student_features[key]

        return aligned_teacher, aligned_student

    def supports_model(self, model: nn.Module) -> bool:
        """Heuristic: model has both text and vision sub-modules but no LM head.

        Distinguishes from VLM (which has an LM head for generation) and
        pure vision (which has no text encoder).
        """
        module_names = {n for n, _ in model.named_modules()}

        has_text_encoder = any("text_model" in n or "text_encoder" in n for n in module_names)
        has_vision_encoder = any(
            "vision_model" in n or "visual_encoder" in n or "vision_tower" in n
            for n in module_names
        )
        has_lm_head = any("language_model" in n or "lm_head" in n for n in module_names)

        # Multimodal = has both encoders but NOT a generative LM head
        return has_text_encoder and has_vision_encoder and not has_lm_head

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_projection(
        self,
        key: str,
        in_dim: int,
        out_dim: int,
        device: torch.device,
    ) -> nn.Linear:
        proj_key = f"proj_{key}_{in_dim}_{out_dim}"
        if proj_key not in self._projections:
            proj = nn.Linear(in_dim, out_dim, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight)
            self._projections[proj_key] = proj
        return self._projections[proj_key]
