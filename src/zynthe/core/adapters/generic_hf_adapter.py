"""Generic HuggingFace adapter — universal fallback.

The :class:`AdapterRegistry` resolves the most specific matching
adapter for a given model.  Anything that doesn't match a typed
adapter (Text / Vision / Code / Multimodal / VLM / Seq2Seq / Audio /
Diffusion) lands here.  :class:`GenericHFAdapter` is the safety net
that lets Zynthé claim "any model that has a callable
``forward``" works.

It auto-detects ``input_ids`` and ``pixel_values`` and
``input_features`` / ``input_values`` style signatures from the
forward() signature, normalises that to the standard batch dict, and
falls back to the :func:`normalize_model_output` helper for output
extraction.

It is *not* intended for high-fidelity distillation — for that, write
a dedicated adapter.  But it makes the registry's fallback path
honest: previously we silently routed unknown models through
:class:`TextModelAdapter`, which only sometimes worked.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from zynthe.core.utils.device_utils import normalize_model_output

from .base_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class GenericHFAdapter(ModelAdapter):
    """Catch-all HuggingFace adapter — matches anything with a callable
    forward that exposes at least one of the well-known kwarg names.
    """

    modality = "generic"

    #: Order matters — checked first when no other adapter claims.
    _KNOWN_INPUT_KEYS = frozenset(
        {
            "input_ids",
            "pixel_values",
            "input_features",
            "input_values",
            "inputs",
            "attention_mask",
            "token_type_ids",
            "decoder_input_ids",
        }
    )

    def __init__(self) -> None:
        # Cache introspection per-model to avoid recomputing forward params.
        self._forward_params_cache: dict[int, frozenset] = {}

    # ------------------------------------------------------------------
    # Detection — always claims support if no other adapter did.
    # ------------------------------------------------------------------

    def supports_model(self, model: nn.Module) -> bool:
        """Match anything with a callable forward and at least one
        well-known HF kwarg.  The registry already tries every typed
        adapter first; GenericHFAdapter acts as the last-resort fallback.
        """
        if not hasattr(model, "forward") or not callable(model.forward):
            return False
        params = self._get_forward_params(model)
        # At least one well-known HF input kwarg must be in the signature.
        return bool(params & self._KNOWN_INPUT_KEYS)

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
        """Generic hook discovery: return the first 4 modules that look
        like transformer blocks (``.layers.``, ``.layer.``, ``.encoder.``
        ..) or `.attn`/`.attention` paths.  Anything more specific should
        live in a typed adapter.
        """
        layers: List[str] = []
        for name, _ in model.named_modules():
            lower = name.lower()
            if not name:
                continue
            if (
                "layers." in lower
                or ".layer." in lower
                or ".encoder." in lower
                or ".decoder." in lower
                or lower.endswith(".attn")
                or lower.endswith(".attention")
                or "conv" in lower
            ):
                layers.append(name)
            if len(layers) >= 4:
                break
        return sorted(layers)

    def align_dimensions(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Generic alignment: interpolate student features to teacher
        shape when needed, leave teacher untouched.
        """
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
            if s_feat.dim() >= 2 and t_feat.dim() >= 2 and s_feat.shape[1:] != t_feat.shape[1:]:
                try:
                    s_feat = F.interpolate(
                        s_feat,
                        size=t_feat.shape[1:],
                        mode="nearest",
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
                self._forward_params_cache[model_id] = self._KNOWN_INPUT_KEYS
        return self._forward_params_cache[model_id]
