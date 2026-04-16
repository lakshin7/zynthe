"""
Text Model Adapter
====================

Handles standard text encoder / decoder models from HuggingFace:

- **Encoders**: BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa
- **Decoders**: GPT-2, LLaMA, Mistral, Phi, Gemma
- **Encoder-Decoders**: T5, BART, mBART

Batch keys: ``input_ids``, ``attention_mask``, ``labels``,
``token_type_ids``, ``decoder_input_ids``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base_adapter import ModelAdapter
from core.utils.device_utils import normalize_model_output


class TextModelAdapter(ModelAdapter):
    """Adapter for text-only transformer models."""

    modality = "text"

    # Typical forward-signature keys for text models.
    _COMMON_KEYS = frozenset({
        "input_ids",
        "attention_mask",
        "labels",
        "token_type_ids",
        "position_ids",
        "head_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "past_key_values",
        "use_cache",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
    })

    # Module name patterns for hookable transformer layers.
    _LAYER_PATTERNS = [
        re.compile(r"^(encoder\.layer\.\d+)$"),               # BERT-like
        re.compile(r"^(transformer\.h\.\d+)$"),                # GPT-2
        re.compile(r"^(model\.layers\.\d+)$"),                 # LLaMA / Mistral
        re.compile(r"^(roberta\.encoder\.layer\.\d+)$"),       # RoBERTa
        re.compile(r"^(bert\.encoder\.layer\.\d+)$"),          # BERT
        re.compile(r"^(distilbert\.transformer\.layer\.\d+)$"),# DistilBERT
        re.compile(r"^(deberta\.encoder\.layer\.\d+)$"),       # DeBERTa
        re.compile(r"^(decoder\.layers\.\d+)$"),               # T5 / BART decoder
        re.compile(r"^(encoder\.layers\.\d+)$"),               # T5 / BART encoder
    ]

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
        """Filter batch to only keys accepted by *model*.forward()."""
        allowed = self._get_forward_params(model)
        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise HF model output into a standard dict."""
        return normalize_model_output(raw_output)

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Discover transformer block layers by name pattern matching."""
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

        For each matching key pair where the last dimension differs,
        a lazy ``nn.Linear`` projection is created to map the student
        features into the teacher dimension space.
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

        # Keep student-only keys as-is
        for key in student_features:
            if key not in aligned_student:
                aligned_student[key] = student_features[key]

        return aligned_teacher, aligned_student

    def supports_model(self, model: nn.Module) -> bool:
        """Heuristic: model has ``input_ids`` in its forward signature."""
        params = self._get_forward_params(model)
        return "input_ids" in params

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
                # Fallback to common text keys
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
