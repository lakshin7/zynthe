"""Seq2Seq model adapter — encoder-decoder transformers (T5, BART,
Marian, Flan-T5, mBART).

Distinguishes from VLM (which has ``language_model`` + ``vision_tower``)
and pure encoder / pure decoder models by requiring both
``encoder.*`` and ``decoder.*`` submodule prefixes (or ``*.*`` matching
those tokens) WITHOUT a generative LM head.

Phase-2 scope: T5, BART, Marian, Flan-T5, mBART — anything with an
explicit encoder/decoder split that isn't an LLaMA-style decoder.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from zynthe.core.utils.device_utils import normalize_model_output

from .base_adapter import ModelAdapter


class Seq2SeqAdapter(ModelAdapter):
    """Adapter for encoder-decoder transformers."""

    modality = "seq2seq"

    #: Module-name patterns: encoder/decoder blocks at any depth.
    _BLOCK_PATTERNS = [
        re.compile(r"^encoder\.(layers|blocks|layer)\.\d+$"),
        re.compile(r"^decoder\.(layers|blocks|layer)\.\d+$"),
    ]

    _COMMON_KEYS = frozenset(
        {
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_input_ids",
            "decoder_attention_mask",
            "head_mask",
            "token_type_ids",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        }
    )

    def __init__(self) -> None:
        self._forward_params_cache: Dict[int, frozenset] = {}

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def supports_model(self, model: nn.Module) -> bool:
        """Match a model that has BOTH encoder.* and decoder.* blocks
        AND does NOT have a generative language_model / lm_head (that
        would be a VLM by our taxonomy).
        """
        module_names = {n for n, _ in model.named_modules()}
        has_encoder = any(self._match_encoder(name) for name in module_names)
        has_decoder = any(self._match_decoder(name) for name in module_names)
        has_lm_head = any(
            "language_model" in n or "lm_head" in n for n in module_names
        )
        return has_encoder and has_decoder and not has_lm_head

    @staticmethod
    def _match_encoder(name: str) -> bool:
        return bool(re.match(r"^encoder\.", name))

    @staticmethod
    def _match_decoder(name: str) -> bool:
        return bool(re.match(r"^decoder\.", name))

    # ------------------------------------------------------------------
    # Batch preparation + output extraction
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: Dict[str, Any],
        model: nn.Module,
    ) -> Dict[str, Any]:
        allowed = self._get_forward_params(model)
        return {k: v for k, v in batch.items() if k in allowed}

    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        return normalize_model_output(raw_output)

    # ------------------------------------------------------------------
    # Hookable layers + dimension alignment (default implementations)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_forward_params(self, model: nn.Module) -> frozenset:
        model_id = id(model)
        if model_id not in self._forward_params_cache:
            try:
                sig = inspect.signature(model.forward)
                self._forward_params_cache[model_id] = frozenset(sig.parameters.keys())
            except (ValueError, TypeError):
                self._forward_params_cache[model_id] = self._COMMON_KEYS
        return self._forward_params_cache[model_id]
