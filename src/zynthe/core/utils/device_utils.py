"""
Device Utilities — Shared Device Management for Zynthé
========================================================

Centralizes device detection and data movement logic used across
distillers, pipelines, and training components.

Previously duplicated in:
- core/distillers/base_distiller.py
- core/pipelines/base_pipeline.py
- core/pipelines/single_distiller_pipeline.py
- core/pipelines/multi_stage_pipeline.py
"""

from __future__ import annotations

from typing import Any

import torch


def auto_detect_device() -> torch.device:
    """Auto-detect the best available compute device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.

    Returns:
        torch.device for the best available backend.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors to *device*.

    Handles plain tensors, dicts, lists, and tuples. Non-tensor
    leaf values are returned unchanged.

    Args:
        data: Arbitrary data structure potentially containing tensors.
        device: Target device.

    Returns:
        Same structure with all tensors moved to *device*.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


def normalize_model_output(raw_output: Any) -> dict:
    """Normalize a HuggingFace-style model output into a plain dict.

    Handles:
    - ``ModelOutput`` dataclasses (have ``.logits``)
    - Plain dicts
    - Tuples (first element treated as logits)
    - Raw tensors (treated as logits)

    Returns:
        Dict with keys: ``logits``, ``hidden_states``, ``attentions``,
        ``loss``.  Missing values are ``None``.
    """
    result: dict = {
        "logits": None,
        "hidden_states": None,
        "attentions": None,
        "loss": None,
    }

    if isinstance(raw_output, dict):
        result["logits"] = raw_output.get("logits")
        result["hidden_states"] = raw_output.get("hidden_states")
        result["attentions"] = raw_output.get("attentions")
        result["loss"] = raw_output.get("loss")
    elif hasattr(raw_output, "logits"):
        # HuggingFace ModelOutput / dataclass
        result["logits"] = getattr(raw_output, "logits", None)
        result["hidden_states"] = getattr(raw_output, "hidden_states", None)
        result["attentions"] = getattr(raw_output, "attentions", None)
        result["loss"] = getattr(raw_output, "loss", None)
    elif isinstance(raw_output, tuple):
        if raw_output:
            result["logits"] = raw_output[0]
        if len(raw_output) > 1:
            result["hidden_states"] = raw_output[1]
        if len(raw_output) > 2:
            result["attentions"] = raw_output[2]
    elif isinstance(raw_output, torch.Tensor):
        result["logits"] = raw_output
    return result
