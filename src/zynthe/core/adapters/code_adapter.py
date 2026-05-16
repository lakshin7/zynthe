"""
Code Model Adapter
====================

Handles code-specific models for distilling programming language
understanding:

- **Encoders**: CodeBERT, GraphCodeBERT, UniXcoder
- **Decoders**: CodeLlama, StarCoder, DeepSeek-Coder, CodeGemma, Codestral
- **Encoder-Decoders**: CodeT5, CodeT5+, PLBART

Code models are structurally similar to text models but benefit from
code-aware batch preparation (e.g. preserving code structure,
handling language-specific tokens).

Batch keys: ``input_ids``, ``attention_mask``, ``labels``,
``decoder_input_ids``, ``token_type_ids``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from zynthe.core.utils.device_utils import normalize_model_output

from .base_adapter import ModelAdapter


class CodeModelAdapter(ModelAdapter):
    """Adapter for code-specific transformer models."""

    modality = "code"

    # Forward signature keys accepted by code models.
    _COMMON_KEYS = frozenset(
        {
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
        }
    )

    # Module name patterns for hookable layers in code models.
    _LAYER_PATTERNS = [
        re.compile(r"^roberta\.encoder\.layer\.(\d+)$"),  # CodeBERT
        re.compile(r"^encoder\.layer\.(\d+)$"),  # generic encoder
        re.compile(r"^model\.layers\.(\d+)$"),  # CodeLlama / DeepSeek
        re.compile(r"^transformer\.h\.(\d+)$"),  # StarCoder (GPT-2 arch)
        re.compile(r"^model\.decoder\.layers\.(\d+)$"),  # CodeT5
        re.compile(r"^decoder\.layers\.(\d+)$"),  # PLBART
        re.compile(r"^encoder\.layers\.(\d+)$"),  # CodeT5 encoder
    ]

    # Known code model name patterns for auto-detection.
    _CODE_MODEL_PATTERNS = [
        "codellama",
        "codebert",
        "codet5",
        "starcoder",
        "starcoderbase",
        "deepseek-coder",
        "deepseek_coder",
        "codegemma",
        "codestral",
        "codegen",
        "unixcoder",
        "graphcodebert",
        "plbart",
        "incoder",
        "santacoder",
        "wizardcoder",
        "phind",
        "magicoder",
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
        """Filter batch to code model-compatible keys."""
        allowed = self._get_forward_params(model)
        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise code model output into a standard dict."""
        return normalize_model_output(raw_output)

    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Discover code model transformer layers by name pattern."""
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

        Identical strategy to text models — project along last dim.
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
        """Heuristic: model name or config contains code-related identifiers.

        Since code models are structurally identical to text models,
        we rely on naming conventions in the model config.
        """
        # Check model config for code model identifiers
        config = getattr(model, "config", None)
        if config is not None:
            model_type = getattr(config, "model_type", "").lower()
            name_or_path = getattr(config, "_name_or_path", "").lower()

            for pattern in self._CODE_MODEL_PATTERNS:
                if pattern in model_type or pattern in name_or_path:
                    return True

        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_forward_params(self, model: nn.Module) -> frozenset:
        """Cache-friendly forward-parameter discovery."""
        model_id = id(model)
        if model_id not in self._forward_params_cache:
            try:
                sig = inspect.signature(model.forward)
                self._forward_params_cache[model_id] = frozenset(sig.parameters.keys())
            except (ValueError, TypeError):
                self._forward_params_cache[model_id] = self._COMMON_KEYS
        return self._forward_params_cache[model_id]

    def _get_or_create_projection(
        self,
        key: str,
        in_dim: int,
        out_dim: int,
        device: torch.device,
    ) -> nn.Linear:
        """Lazily create a linear projection for dimension alignment."""
        proj_key = f"proj_{key}_{in_dim}_{out_dim}".replace(".", "_")
        if proj_key not in self._projections:
            proj = nn.Linear(in_dim, out_dim, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight)
            self._projections[proj_key] = proj
        return self._projections[proj_key]
