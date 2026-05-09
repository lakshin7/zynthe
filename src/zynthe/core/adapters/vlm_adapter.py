"""
Vision-Language Model (VLM) Adapter
=====================================

Handles generative VLMs with a vision encoder feeding into a language
model via a projection layer:

- **LLaVA** / LLaVA-NeXT
- **InternVL** / InternVL-2
- **Qwen-VL** / Qwen2-VL
- **Phi-3-Vision** / Phi-4-Vision
- **mPLUG-Owl** / mPLUG-Owl2

Architecture pattern::

    image → [Vision Encoder] → [Projection] → tokens
                                                ↓
    text → [Tokenizer] → input_ids → [LLM] → logits

The adapter exposes hook points for each sub-component and handles
the heterogeneous I/O shapes.

Batch keys: ``pixel_values``, ``input_ids``, ``attention_mask``,
``labels``, ``image_sizes``, ``images``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base_adapter import ModelAdapter
from zynthe.core.utils.device_utils import normalize_model_output


class VLMModelAdapter(ModelAdapter):
    """Adapter for generative Vision-Language Models."""

    modality = "vlm"

    # Module-name patterns for the three sub-components.
    _VISION_PATTERNS = [
        re.compile(r"^vision_tower\..*?encoder\.layers?\.\d+$"),
        re.compile(r"^vision_model\.encoder\.layers?\.\d+$"),
        re.compile(r"^visual\.blocks\.\d+$"),
        re.compile(r"^model\.vision_tower\..*?layer\.\d+$"),
    ]

    _PROJECTION_PATTERNS = [
        re.compile(r"^multi_modal_projector"),
        re.compile(r"^mm_projector"),
        re.compile(r"^visual_projection"),
        re.compile(r"^vision_proj"),
        re.compile(r"^model\.mm_projector"),
    ]

    _LM_PATTERNS = [
        re.compile(r"^language_model\.model\.layers\.\d+$"),
        re.compile(r"^model\.layers\.\d+$"),
        re.compile(r"^lm\.model\.layers\.\d+$"),
        re.compile(r"^model\.model\.layers\.\d+$"),
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
        """Filter batch for VLM forward signature.

        VLMs typically accept a mix of text and vision keys.  We
        introspect the forward method and fall back to a generous
        allowlist.
        """
        # Try introspection first
        try:
            sig = inspect.signature(model.forward)
            allowed = set(sig.parameters.keys())
        except (ValueError, TypeError):
            allowed = set()

        # Generous fallback for VLMs
        allowed |= {
            "pixel_values", "input_ids", "attention_mask", "labels",
            "images", "image_sizes", "image_token_index",
            "return_dict", "output_attentions", "output_hidden_states",
            "past_key_values", "use_cache",
        }

        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise VLM output.

        Beyond the standard logits/hidden_states/attentions, VLMs may
        expose ``image_features`` or ``vision_outputs``.
        """
        result = normalize_model_output(raw_output)

        # VLM-specific extra fields
        for attr in (
            "image_features",
            "vision_outputs",
            "projected_image_features",
            "language_model_outputs",
        ):
            val = (
                raw_output.get(attr) if isinstance(raw_output, dict)
                else getattr(raw_output, attr, None)
            )
            if val is not None:
                result[attr] = val

        return result

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Return layers from all three VLM sub-components.

        Returns:
            Sorted list with entries from vision tower, projection,
            and language model.
        """
        layers: List[str] = []
        all_patterns = (
            self._VISION_PATTERNS
            + self._PROJECTION_PATTERNS
            + self._LM_PATTERNS
        )
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
        """Align vision + language features.

        Vision encoder features and LM hidden states may live in very
        different dimensionalities.  We handle each category via
        separate projection heads.
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
        """Heuristic: model has a vision tower/encoder AND a language model."""
        module_names = {n for n, _ in model.named_modules()}

        has_vision_tower = any(
            "vision_tower" in n or "vision_model" in n or "visual" in n
            for n in module_names
        )
        has_lm = any(
            "language_model" in n or "lm_head" in n
            for n in module_names
        )
        has_projector = any(
            "projector" in n or "vision_proj" in n
            for n in module_names
        )

        # VLM = vision encoder + LM (projector is common but not required)
        return has_vision_tower and (has_lm or has_projector)

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
