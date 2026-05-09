"""Post-training quantization utilities."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Type

import torch
from torch import nn

from zynthe.core.models.model_loader import ModelLoader
from zynthe.core.quant.calibration import CalibrationConfig, CalibrationRunner, build_calibration_loader

LOG = logging.getLogger(__name__)


def _resolve_device(device: Any | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device:
        try:
            return torch.device(device)
        except (RuntimeError, TypeError):
            LOG.warning("Unknown device '%s'; falling back to CPU.", device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        mapping = {
            "qint8": torch.qint8,
            "quint8": torch.quint8,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        found = mapping.get(dtype.lower())
        if found is not None:
            return found
    return torch.qint8


def _resolve_target_modules(modules: Optional[Sequence[Any]]) -> Tuple[Type[nn.Module], ...]:
    if not modules:
        return (nn.Linear,)
    resolved: list[Type[nn.Module]] = []
    candidates = modules if isinstance(modules, Sequence) else [modules]
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, nn.Module):
            resolved.append(candidate)
        elif isinstance(candidate, str):
            attr = getattr(nn, candidate, None)
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                resolved.append(attr)
            else:
                LOG.warning("Unknown module '%s' requested for PTQ; skipping.", candidate)
    return tuple(resolved) or (nn.Linear,)


def _default_backend(device: torch.device) -> str:
    current = torch.backends.quantized.engine
    if current:
        return current
    if device.type == "cpu":
        return "fbgemm"
    return "qnnpack"


def _estimate_model_size(model: nn.Module) -> int:
    size = 0
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor):
            size += tensor.numel() * tensor.element_size()
    return int(size)


def _bytes_to_megabytes(value: int) -> float:
    return round(value / (1024 ** 2), 4)


class PTQRunner:
    """High-level orchestrator that applies PTQ using a config dictionary."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.quant_cfg = cfg.get("quantization", {}) or {}
        self.strategy = (
            self.quant_cfg.get("strategy")
            or self.quant_cfg.get("method")
            or self.quant_cfg.get("mode")
            or "dynamic"
        ).lower()
        self.fallback = (self.quant_cfg.get("fallback_strategy") or "float16").lower()
        self.dtype = _resolve_dtype(self.quant_cfg.get("dtype", torch.qint8))
        self.device = _resolve_device(self.quant_cfg.get("device") or cfg.get("runtime", {}).get("device"))
        self.backend = (self.quant_cfg.get("backend") or _default_backend(self.device)).lower()
        self.target_modules = _resolve_target_modules(self.quant_cfg.get("modules"))
        self.calibration_cfg = self._build_calibration_cfg()
        self.export_dir = self._prepare_output_dir()
        self.summary: Dict[str, Any] = {}

    def _build_calibration_cfg(self) -> CalibrationConfig:
        cal_cfg = self.quant_cfg.get("calibration", {}) or {}
        raw_max_samples = cal_cfg.get("max_samples")
        max_samples: Optional[int]
        if raw_max_samples in (None, "auto"):
            max_samples = None
        else:
            try:
                max_samples = int(raw_max_samples) if raw_max_samples is not None else None
            except (TypeError, ValueError):
                LOG.warning("Invalid calibration.max_samples=%s; using default", raw_max_samples)
                max_samples = None

        return CalibrationConfig(
            num_batches=int(cal_cfg.get("num_batches", 32)),
            max_samples=max_samples,
            use_training_split=bool(cal_cfg.get("use_training_split", False)),
            shuffle=bool(cal_cfg.get("shuffle", False)),
        )

    def _prepare_output_dir(self) -> Path:
        root = Path(self.quant_cfg.get("output_dir") or self.cfg.get("output_root", "experiments"))
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        final_dir = root / f"quantized_{timestamp}"
        final_dir.mkdir(parents=True, exist_ok=True)
        return final_dir

    def run(self) -> Dict[str, Any]:
        LOG.info("Starting PTQ runner with strategy '%s'", self.strategy)
        loader = ModelLoader(self.cfg, device=self.device)
        bundle = loader.load(return_bundle=True)
        if isinstance(bundle, tuple):
            _, student, tokenizer = bundle
        else:
            student = bundle.student
            tokenizer = bundle.tokenizer

        baseline_size = _estimate_model_size(student)

        calibration_loader = None
        if self.strategy in {"static", "int8_static", "per_tensor"}:
            try:
                calibration_loader = build_calibration_loader(self.cfg, tokenizer, self.calibration_cfg)
            except Exception as exc:
                LOG.warning("Failed to build calibration loader: %s", exc)

        quantized_model, used_strategy = self.quantize_model(
            model=student,
            strategy=self.strategy,
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            target_modules=self.target_modules,
            fallback=self.fallback,
            calibration_loader=calibration_loader,
            calibration_cfg=self.calibration_cfg,
        )

        quant_size = _estimate_model_size(quantized_model)

        self.summary = {
            "strategy_requested": self.strategy,
            "strategy_used": used_strategy,
            "dtype": str(self.dtype),
            "backend": self.backend,
            "size_before_bytes": baseline_size,
            "size_after_bytes": quant_size,
            "size_before_mb": _bytes_to_megabytes(baseline_size),
            "size_after_mb": _bytes_to_megabytes(quant_size),
            "size_delta_mb": _bytes_to_megabytes(baseline_size - quant_size),
            "export_dir": str(self.export_dir),
        }

        LOG.info(
            "PTQ completed: %s → %s MB (Δ %.2f MB) using strategy '%s'",
            self.summary["size_before_mb"],
            self.summary["size_after_mb"],
            self.summary["size_delta_mb"],
            used_strategy,
        )

        self._export_artifacts(quantized_model, tokenizer)
        self._write_summary()

        return self.summary

    def _export_artifacts(self, model: nn.Module, tokenizer) -> None:
        exported = False
        try:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(self.export_dir)  # type: ignore[attr-defined]
                exported = True
            if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(self.export_dir)  # type: ignore[attr-defined]
        except Exception as exc:
            LOG.warning("save_pretrained export failed: %s", exc)

        if not exported:
            torch.save(model.state_dict(), self.export_dir / "pytorch_model_quantized.bin")

    def _write_summary(self) -> None:
        summary_path = self.export_dir / "quantization_summary.json"
        try:
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(self.summary, handle, indent=2)
        except Exception as exc:
            LOG.warning("Failed to write quantization summary: %s", exc)

    @staticmethod
    def quantize_model(
        model: nn.Module,
        strategy: str,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str = torch.qint8,
        backend: Optional[str] = None,
        target_modules: Optional[Sequence[Any]] = None,
        fallback: Optional[str] = "float16",
        calibration_loader: Optional[Iterable[Dict[str, Any]]] = None,
        calibration_cfg: Optional[CalibrationConfig] = None,
    ) -> Tuple[nn.Module, str]:
        device_obj = _resolve_device(device)
        dtype_obj = _resolve_dtype(dtype)
        modules = _resolve_target_modules(target_modules)
        backend_name = (backend or _default_backend(device_obj)).lower()
        calibration_cfg = calibration_cfg or CalibrationConfig()
        requested_strategy = (strategy or "dynamic").lower()
        fallback_strategy = (fallback or "").lower()

        def _execute(name: str, allow_fallback: bool = True) -> Tuple[nn.Module, str]:
            normalized = name.lower()
            if normalized in {"dynamic", "ptq", "int8"}:
                quantized = PTQRunner._apply_dynamic(model, dtype_obj, modules)
                return quantized, "dynamic"
            if normalized in {"float16", "fp16"}:
                quantized = PTQRunner._apply_float16(model, device_obj)
                return quantized, "float16"
            if normalized in {"static", "int8_static", "per_tensor"}:
                if calibration_loader is None:
                    raise RuntimeError("Static PTQ requires calibration data.")
                quantized = PTQRunner._apply_static(
                    model,
                    backend_name,
                    calibration_loader,
                    calibration_cfg,
                )
                return quantized, "static"
            raise ValueError(f"Unknown PTQ strategy '{name}'")

        try:
            return _execute(requested_strategy, allow_fallback=True)
        except Exception as exc:
            LOG.warning("Quantization strategy '%s' failed: %s", requested_strategy, exc)
            if fallback_strategy and fallback_strategy != requested_strategy:
                LOG.info("Falling back to strategy '%s'", fallback_strategy)
                return _execute(fallback_strategy, allow_fallback=False)
            raise

    @staticmethod
    def _apply_dynamic(
        model: nn.Module,
        dtype: torch.dtype,
        modules: Tuple[Type[nn.Module], ...],
    ) -> nn.Module:
        torch.backends.quantized.engine = torch.backends.quantized.engine or "qnnpack"
        clone = copy.deepcopy(model)
        clone.eval()
        clone_cpu = clone.to(torch.device("cpu"))
        quantized = torch.quantization.quantize_dynamic(
            clone_cpu,
            {module for module in modules},
            dtype=dtype,
        )
        if hasattr(quantized, "to"):
            try:
                return quantized.to(torch.device("cpu"))
            except Exception:
                return quantized
        return quantized

    @staticmethod
    def _apply_float16(model: nn.Module, device: torch.device) -> nn.Module:
        clone = copy.deepcopy(model)
        clone.eval()
        fp16_model = clone.to(torch.float16)
        try:
            return fp16_model.to(device)
        except Exception:
            LOG.warning("Device %s does not support float16 natively; staying on CPU.", device)
            return fp16_model.to(torch.device("cpu"))

    @staticmethod
    def _apply_static(
        model: nn.Module,
        backend: str,
        calibration_loader: Iterable[Dict[str, Any]],
        calibration_cfg: CalibrationConfig,
    ) -> nn.Module:
        torch.backends.quantized.engine = backend
        clone = copy.deepcopy(model)
        clone.eval()
        clone_cpu = clone.to(torch.device("cpu"))
        setattr(clone_cpu, "qconfig", torch.quantization.get_default_qconfig(backend))
        prepared = torch.quantization.prepare(clone_cpu, inplace=False)
        runner = CalibrationRunner(prepared, calibration_loader, torch.device("cpu"), calibration_cfg)
        used = runner.collect()
        LOG.info("Calibrated static PTQ with %d samples", used)
        quantized = torch.quantization.convert(prepared, inplace=False)
        return quantized


def apply_ptq(
    model: nn.Module,
    device: torch.device | str,
    dtype: torch.dtype = torch.qint8,
    mode: Optional[str] = None,
) -> nn.Module:
    """Lightweight helper to quantize an in-memory model (used by training pipeline)."""

    target_device = _resolve_device(device)
    strategy = (mode or "dynamic").lower()
    if strategy in {"dynamic", "ptq", "int8"} and target_device.type != "cpu":
        LOG.info(
            "Dynamic int8 PTQ is CPU-only; switching to float16 for device %s",
            target_device,
        )
        strategy = "float16"

    quantized, used_strategy = PTQRunner.quantize_model(
        model=model,
        strategy=strategy,
        device=target_device,
        dtype=dtype,
        backend=None,
        target_modules=(nn.Linear,),
        fallback="float16",
        calibration_loader=None,
        calibration_cfg=None,
    )
    LOG.debug("apply_ptq used strategy '%s'", used_strategy)
    try:
        return quantized.to(target_device)
    except Exception as exc:
        LOG.debug("Leaving quantized model on CPU (move to %s failed: %s)", target_device, exc)
        return quantized
    return quantized