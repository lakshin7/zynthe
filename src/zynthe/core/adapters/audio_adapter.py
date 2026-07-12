"""Audio model adapter — Whisper, Wav2Vec2, HuBERT, SeamlessM4T (encoder/decoder).

Distinguishes from the typed adapters by inspecting the forward signature:
* ``input_features`` (Whisper, SeamlessM4T)
* ``input_values`` (Wav2Vec2, HuBERT, SpeechT5)

Phase-2 scope: encoder-only ASR + encoder-decoder speech translation.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from zynthe.core.utils.device_utils import normalize_model_output

from .base_adapter import ModelAdapter


class AudioAdapter(ModelAdapter):
    """Adapter for ASR / speech transformers with audio-style forward kwargs."""

    modality = "audio"

    _COMMON_KEYS = frozenset(
        {
            "input_features",
            "input_values",
            "attention_mask",
            "labels",
            "decoder_input_ids",
            "decoder_attention_mask",
            "sampling_rate",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        }
    )

    #: Module-name patterns for audio transformers.
    _BLOCK_PATTERNS = [
        re.compile(r"^encoder\.layers\.\d+$"),
        re.compile(r"^encoder\.layer\.\d+$"),
        re.compile(r"^encoder\.layers\.0\.self_attn$"),
        re.compile(r"^decoder\.layers\.\d+$"),
        re.compile(r"^decoder\.layer\.\d+$"),
    ]

    def __init__(self) -> None:
        self._forward_params_cache: Dict[int, frozenset] = {}

    # ------------------------------------------------------------------
    # Detection — forward signature based
    # ------------------------------------------------------------------

    def supports_model(self, model: nn.Module) -> bool:
        """Match if the forward signature includes ``input_features``
        (Whisper) or ``input_values`` (Wav2Vec2 / HuBERT).
        """
        params = self._get_forward_params(model)
        return "input_features" in params or "input_values" in params

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
    # Hookable layers + dimension alignment
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
