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


def _save_pretrained_compat(obj: Any, path: Path, *, safe_serialization: bool = False) -> None:
    if obj is None or not hasattr(obj, "save_pretrained"):
        return
    try:
        obj.save_pretrained(str(path), safe_serialization=safe_serialization)
    except TypeError:
        obj.save_pretrained(str(path))


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


class ModelSaver:
    """Compatibility facade exposing the high-level save/export workflow."""

    @staticmethod
    def save_training_run(
        model: Any,
        tokenizer: Optional[Any],
        save_dir: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        metrics_history: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        best_metric: Optional[float] = None,
        evaluation_report: Optional[Any] = None,
        use_safetensors: bool = True,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        commit_message: str = "Add model from training run",
    ) -> str:
        """Save a complete training run bundle in a single directory."""

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        _save_pretrained_compat(model, save_path, safe_serialization=use_safetensors)
        _save_pretrained_compat(tokenizer, save_path)

        if config is not None:
            with (save_path / "training_config.json").open("w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)

        if metrics_history is not None:
            with (save_path / "metrics_history.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics_history, handle, indent=2)

        if evaluation_report is not None:
            report_path = save_path / "evaluation_report.json"
            if hasattr(evaluation_report, "save_json"):
                evaluation_report.save_json(report_path)
            else:
                with report_path.open("w", encoding="utf-8") as handle:
                    json.dump(evaluation_report, handle, indent=2)

        if optimizer is not None or scheduler is not None or scaler is not None:
            metadata = CheckpointMetadata(
                epoch=int(epoch or 0),
                global_step=int(global_step or 0),
                best_metric=best_metric,
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                path=str(save_path / "checkpoint.pt"),
                scheduler=scheduler,
                scaler=scaler,
                metadata=metadata,
            )

        manifest = {
            "model_dir": str(save_path),
            "has_config": bool(config is not None),
            "has_metrics_history": bool(metrics_history is not None),
            "has_checkpoint": bool((save_path / "checkpoint.pt").exists()),
            "has_evaluation_report": bool((save_path / "evaluation_report.json").exists()),
            "safe_serialization": bool(use_safetensors),
        }
        with (save_path / "training_run_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        if push_to_hub and repo_id:
            model.push_to_hub(repo_id, commit_message=commit_message)
            if tokenizer is not None and hasattr(tokenizer, "push_to_hub"):
                tokenizer.push_to_hub(repo_id, commit_message=commit_message)

        return str(save_path)

    @staticmethod
    def export_for_deployment(
        model: Any,
        tokenizer: Optional[Any],
        save_dir: str,
        export_format: str = "onnx",
        **kwargs: Any,
    ) -> str:
        """Export a model directory into one or more deployment-friendly formats."""

        requested = [fmt.strip().lower() for fmt in str(export_format).split(",") if fmt.strip()]
        if not requested:
            raise ValueError("No export format requested")

        if len(requested) > 1:
            for single_format in requested:
                ModelSaver.export_for_deployment(
                    model=model,
                    tokenizer=tokenizer,
                    save_dir=save_dir,
                    export_format=single_format,
                    **kwargs,
                )
            return str(Path(save_dir))

        fmt = requested[0]
        save_path = Path(save_dir) / fmt
        save_path.mkdir(parents=True, exist_ok=True)

        if fmt == "onnx":
            try:
                from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification

                temp_dir = save_path / "temp_hf"
                _save_pretrained_compat(model, temp_dir)
                _save_pretrained_compat(tokenizer, temp_dir)

                if hasattr(model, "classifier") or "SequenceClassification" in model.__class__.__name__:
                    ort_model = ORTModelForSequenceClassification.from_pretrained(str(temp_dir), export=True)
                else:
                    ort_model = ORTModelForCausalLM.from_pretrained(str(temp_dir), export=True)

                ort_model.save_pretrained(str(save_path))
                _save_pretrained_compat(tokenizer, save_path)
            except ImportError as exc:
                raise NotImplementedError(
                    "Install optimum for ONNX export: pip install optimum[onnxruntime]"
                ) from exc

        elif fmt == "torchscript":
            example_inputs = kwargs.get("example_inputs")
            if example_inputs is None:
                sample_text = kwargs.get("example_text", "hello world")
                if tokenizer is None:
                    raise ValueError("tokenizer is required when example_inputs is not provided")
                encoded = tokenizer(sample_text, return_tensors="pt")
                input_ids = encoded.get("input_ids")
                attention_mask = encoded.get("attention_mask")

                class _TorchscriptWrapper(torch.nn.Module):
                    def __init__(self, wrapped_model: Any):
                        super().__init__()
                        self.wrapped_model = wrapped_model

                    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
                        outputs = self.wrapped_model(input_ids=ids, attention_mask=mask)
                        return outputs.logits if hasattr(outputs, "logits") else outputs[0]

                wrapper = _TorchscriptWrapper(model.eval())
                example_inputs = (input_ids, attention_mask) if attention_mask is not None else (input_ids,)
                export_torchscript(wrapper, str(save_path / "model.torchscript.pt"), example_inputs)
            else:
                export_torchscript(model.eval(), str(save_path / "model.torchscript.pt"), example_inputs)
            _save_pretrained_compat(tokenizer, save_path)

        elif fmt == "safetensors":
            _save_pretrained_compat(model, save_path, safe_serialization=True)
            _save_pretrained_compat(tokenizer, save_path)

        elif fmt == "gguf":
            _save_pretrained_compat(model, save_path / "hf_source", safe_serialization=True)
            _save_pretrained_compat(tokenizer, save_path / "hf_source")
            with (save_path / "README_GGUF.txt").open("w", encoding="utf-8") as handle:
                handle.write(
                    "GGUF export scaffold generated. Convert hf_source with llama.cpp conversion tools.\n"
                )

        elif fmt == "bitnet":
            _save_pretrained_compat(model, save_path / "hf_source", safe_serialization=True)
            _save_pretrained_compat(tokenizer, save_path / "hf_source")
            with (save_path / "README_BITNET.txt").open("w", encoding="utf-8") as handle:
                handle.write(
                    "BitNet export scaffold generated. Apply BitNet conversion/quantization tooling to hf_source.\n"
                )

        else:
            raise ValueError(f"Unknown export format: {fmt}")

        with (save_path / "export_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump({"format": fmt, "output_dir": str(save_path)}, handle, indent=2)

        return str(save_path)