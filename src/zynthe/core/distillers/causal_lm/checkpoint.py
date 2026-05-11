"""Fault-tolerant checkpointing for Causal-LM distillation training."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

LOG = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    seed: int = 42
    dataset_hash: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    model_sizes: Dict[str, int] = field(default_factory=dict)
    distillation_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointLoadReport:
    strict_loaded: bool = False
    fallback_used: bool = False
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    shape_mismatch: List[str] = field(default_factory=list)
    loaded_tensors: int = 0
    skipped_tensors: int = 0
    skipped_optimizer: bool = False
    optimizer_restored: bool = False
    optimizer_reset: bool = False
    scheduler_restored: bool = False
    scaler_restored: bool = False
    rng_restored: bool = False
    optimizer_restore_reason: str = ""
    warning: str = ""


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None


def _capture_rng_state() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: Dict[str, Any]) -> bool:
    try:
        if "python" in payload:
            random.setstate(payload["python"])
        if "numpy" in payload:
            np.random.set_state(payload["numpy"])
        if "torch" in payload:
            torch.set_rng_state(payload["torch"])
        if "torch_cuda" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["torch_cuda"])
        return True
    except Exception as exc:
        LOG.warning("Failed to restore RNG state: %s", exc)
        return False


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        torch.save(payload, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
    shutil.move(tmp.name, path)


def save_training_checkpoint(
    *,
    path: str,
    model: Any,
    optimizer: Optional[Any],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    state: TrainingState,
    metadata: CheckpointMeta,
) -> Path:
    payload: Dict[str, Any] = {
        "checkpoint_version": 3,
        "model_state_dict": model.state_dict(),
        "state": asdict(state),
        "metadata": asdict(metadata),
        "rng_state": _capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["optimizer_class"] = type(optimizer).__name__
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler_state_dict"] = scaler.state_dict()

    out_path = Path(path)
    _atomic_torch_save(out_path, payload)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump({"state": asdict(state), "metadata": asdict(metadata)}, handle, indent=2)

    LOG.info("Saved training checkpoint: %s", out_path)
    return out_path


def _safe_state_load(target: Any, state_dict: Dict[str, Any]) -> bool:
    try:
        target.load_state_dict(state_dict)
        return True
    except Exception as exc:
        LOG.warning("Skipping state restore due to incompatibility: %s", exc)
        return False


def _load_compatible_model_weights(
    model: Any, ckpt_state: Dict[str, torch.Tensor]
) -> CheckpointLoadReport:
    report = CheckpointLoadReport(strict_loaded=False, fallback_used=True)
    model_state = model.state_dict()

    compatible: Dict[str, torch.Tensor] = {}
    for key, value in ckpt_state.items():
        if key not in model_state:
            report.unexpected_keys.append(key)
            continue
        target = model_state[key]
        if tuple(target.shape) != tuple(value.shape):
            report.shape_mismatch.append(key)
            continue
        compatible[key] = value

    report.missing_keys = [k for k in model_state.keys() if k not in compatible]
    load_result = model.load_state_dict(compatible, strict=False)
    report.unexpected_keys.extend(list(getattr(load_result, "unexpected_keys", [])))
    report.missing_keys.extend(list(getattr(load_result, "missing_keys", [])))
    report.loaded_tensors = len(compatible)
    report.skipped_tensors = max(len(model_state) - report.loaded_tensors, 0)

    if report.shape_mismatch:
        report.warning = (
            f"Loaded with fallback. Shape mismatch for {len(report.shape_mismatch)} tensors; "
            f"loaded compatible tensors only ({report.loaded_tensors} keys)."
        )
    else:
        report.warning = f"Loaded with fallback using {report.loaded_tensors} compatible tensors."

    return report


def _reset_optimizer_state(optimizer: Any) -> None:
    try:
        optimizer.state.clear()
    except Exception:
        optimizer.state = {}  # type: ignore[assignment]


def _optimizer_state_compatible(
    optimizer: Any,
    checkpoint: Dict[str, Any],
) -> Tuple[bool, str]:
    opt_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(opt_state, dict):
        return False, "missing_optimizer_state"

    saved_class = str(checkpoint.get("optimizer_class", "")).strip()
    current_class = type(optimizer).__name__
    if saved_class and saved_class != current_class:
        return False, "incompatible_optimizer_type"

    saved_groups = opt_state.get("param_groups", [])
    current_groups = optimizer.state_dict().get("param_groups", [])
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        return False, "invalid_optimizer_param_groups"
    if len(saved_groups) != len(current_groups):
        return False, "optimizer_param_group_count_mismatch"

    for saved_g, current_g in zip(saved_groups, current_groups):
        saved_params = saved_g.get("params", []) if isinstance(saved_g, dict) else []
        current_params = current_g.get("params", []) if isinstance(current_g, dict) else []
        if len(saved_params) != len(current_params):
            return False, "optimizer_param_count_mismatch"

    return True, ""


def smart_load_checkpoint(
    *,
    path: str,
    model: Any,
    optimizer: Optional[Any],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    map_location: Optional[Any] = None,
    strict_first: bool = True,
    allow_shape_mismatch_fallback: bool = True,
) -> Tuple[CheckpointLoadReport, TrainingState, Optional[CheckpointMeta]]:
    checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model_state_dict: {path}")

    report = CheckpointLoadReport()
    if strict_first:
        try:
            model.load_state_dict(model_state, strict=True)
            report.strict_loaded = True
            report.loaded_tensors = len(model_state)
            report.skipped_tensors = 0
        except RuntimeError as exc:
            if not allow_shape_mismatch_fallback:
                raise
            LOG.warning("Strict checkpoint load failed, falling back to compatible load: %s", exc)
            report = _load_compatible_model_weights(model, model_state)
    else:
        report = _load_compatible_model_weights(model, model_state)

    # Restore optimizer/scheduler/scaler only on strict load for safety.
    if report.strict_loaded:
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            compatible, reason = _optimizer_state_compatible(optimizer, checkpoint)
            if not compatible:
                restored = False
                report.optimizer_restore_reason = reason
            else:
                restored = _safe_state_load(optimizer, checkpoint["optimizer_state_dict"])
                if not restored:
                    report.optimizer_restore_reason = "incompatible_optimizer_state"

            report.optimizer_restored = restored
            if not restored:
                _reset_optimizer_state(optimizer)
                report.optimizer_reset = True
        elif optimizer is not None:
            report.optimizer_restore_reason = "missing_optimizer_state"
            report.optimizer_reset = True
            _reset_optimizer_state(optimizer)
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            report.scheduler_restored = _safe_state_load(
                scheduler, checkpoint["scheduler_state_dict"]
            )
        if scaler is not None and "scaler_state_dict" in checkpoint:
            report.scaler_restored = _safe_state_load(scaler, checkpoint["scaler_state_dict"])
    else:
        report.skipped_optimizer = True
        report.optimizer_restore_reason = "fallback_model_load"
        if optimizer is not None:
            _reset_optimizer_state(optimizer)
            report.optimizer_reset = True

    rng_payload = checkpoint.get("rng_state")
    if isinstance(rng_payload, dict):
        report.rng_restored = _restore_rng_state(rng_payload)

    state_payload = checkpoint.get("state", {})
    training_state = TrainingState(
        epoch=int(state_payload.get("epoch", 0)),
        global_step=int(state_payload.get("global_step", 0)),
        best_metric=state_payload.get("best_metric", None),
    )

    meta_payload = checkpoint.get("metadata")
    metadata = None
    if isinstance(meta_payload, dict):
        metadata = CheckpointMeta(
            epoch=int(meta_payload.get("epoch", training_state.epoch)),
            global_step=int(meta_payload.get("global_step", training_state.global_step)),
            best_metric=meta_payload.get("best_metric", training_state.best_metric),
            seed=int(meta_payload.get("seed", 42)),
            dataset_hash=str(meta_payload.get("dataset_hash", "")),
            config_snapshot=meta_payload.get("config_snapshot", {}) or {},
            model_sizes=meta_payload.get("model_sizes", {}) or {},
            distillation_params=meta_payload.get("distillation_params", {}) or {},
        )

    LOG.info(
        "Loaded checkpoint %s (strict=%s, fallback=%s, loaded_tensors=%d skipped_tensors=%d)",
        path,
        report.strict_loaded,
        report.fallback_used,
        report.loaded_tensors,
        report.skipped_tensors,
    )
    if report.warning:
        LOG.warning(report.warning)

    return report, training_state, metadata
