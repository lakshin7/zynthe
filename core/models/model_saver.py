"""Enterprise-grade checkpoint and model persistence utilities."""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch


logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Structured metadata saved alongside checkpoints for traceability."""

    stage: Optional[str] = None
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "stage": self.stage,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "metrics": self.metrics,
        }
        if self.extras:
            payload["extras"] = self.extras
        return payload


def _atomic_write(tensor_path: Path, data: Dict[str, Any]) -> None:
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tensor_path.parent, delete=False) as tmp:
        torch.save(data, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
    shutil.move(tmp.name, tensor_path)


def save_checkpoint(
    model: Any,
    optimizer: Optional[Any],
    path: str,
    *,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    metadata: Optional[CheckpointMetadata] = None,
) -> Path:
    """Persist a full training checkpoint with safety guards."""

    checkpoint = {
        "checkpoint_version": 2,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if metadata is not None:
        checkpoint["metadata"] = metadata.to_dict()

    output_path = Path(path)
    _atomic_write(output_path, checkpoint)
    logger.info("Checkpoint saved to %s", output_path)
    return output_path


def load_checkpoint(
    model: Any,
    optimizer: Optional[Any],
    path: str,
    *,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Optional[Any] = None,
    strict: bool = True,
) -> Tuple[Dict[str, Any], Optional[CheckpointMetadata]]:
    """Load checkpoint and restore model/optimizer/scheduler states."""

    safe_map_location = map_location if map_location is not None else "cpu"
    checkpoint = torch.load(path, map_location=safe_map_location, weights_only=False)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if not strict:
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        if missing or unexpected:
            logger.warning(
                "Non-strict checkpoint load: missing_keys=%d unexpected_keys=%d",
                len(missing),
                len(unexpected),
            )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    metadata_payload = checkpoint.get("metadata")
    metadata = None
    if isinstance(metadata_payload, dict):
        metadata = CheckpointMetadata(
            stage=metadata_payload.get("stage"),
            epoch=int(metadata_payload.get("epoch", 0)),
            global_step=int(metadata_payload.get("global_step", 0)),
            best_metric=metadata_payload.get("best_metric"),
            metrics=metadata_payload.get("metrics", {}),
            extras=metadata_payload.get("extras", {}),
        )

    logger.info("Checkpoint loaded from %s", path)
    return checkpoint, metadata


def save_model(
    model: Any,
    path: str,
    tokenizer: Optional[Any] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    use_safetensors: bool = False,
) -> Path:
    """Save a pretrained model (and tokenizer) with optional metadata."""

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir, safe_serialization=use_safetensors)
    else:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    if metadata:
        metadata_path = output_dir / "model_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    logger.info("Model artifacts saved to %s", output_dir)
    return output_dir


def load_model(
    model_class: Any,
    path: str,
    tokenizer_class: Optional[Any] = None,
    *,
    map_location: Optional[str] = None,
):
    """Reload a saved model directory with optional tokenizer."""

    model = model_class.from_pretrained(path)
    if map_location:
        model = model.to(map_location)
    tokenizer = None
    if tokenizer_class is not None:
        tokenizer = tokenizer_class.from_pretrained(path)

    metadata = None
    metadata_path = Path(path) / "model_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        logger.info("Loaded model metadata from %s", metadata_path)

    logger.info("Model loaded from %s", path)
    return model, tokenizer, metadata


def export_torchscript(model: Any, path: str, example_inputs: Any) -> Path:
    """Export the model to TorchScript for production deployments."""

    scripted = torch.jit.trace(model.cpu(), example_inputs)
    scripted_module = scripted[0] if isinstance(scripted, tuple) else scripted
    scripted_module = cast(torch.jit.ScriptModule, scripted_module)
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    scripted_module.save(str(export_path))
    logger.info("TorchScript model exported to %s", export_path)
    return export_path


def export_onnx(model: Any, path: str, example_inputs: Any, *, opset: int = 14) -> Path:
    """Export the model to ONNX."""

    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example_inputs,
        str(export_path),
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    logger.info("ONNX model exported to %s", export_path)
    return export_path