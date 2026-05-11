# core/config/config_manager.py
"""
ConfigManager for Zynthé (M2-first, L40S-ready).

Key features:
- Load default & user YAML; deep-merge with runtime overrides
- Auto-detect device: prefer MPS (Mac M1/M2) -> CUDA -> CPU
- Apply device-appropriate defaults (batch size / mixed-precision hints)
- Create reproducible experiment directories and save resolved_config.yaml
- Capture lightweight environment metadata (torch, cuda/mps availability, OS)
- Expose simple API to downstream code

Designed to be imported and used by scripts that wire together models/distillers/training.
"""

from __future__ import annotations
import os
import yaml  # type: ignore[import-untyped]
import uuid
import shutil
import logging
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Dict, Optional
import random
import numpy as np
import platform
import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_DEFAULTS_FILENAME = "default.yaml"


def _coerce_float(value: Any, default: float) -> float:
    """Safely coerce config values to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float config value %r; using default=%s", value, default)
        return float(default)


class ConfigError(Exception):
    pass


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating the original."""
    merged = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def _default_config_path() -> str:
    """Resolve packaged defaults with a source-tree fallback for editable installs."""

    candidates = []
    try:
        candidates.append(resources.files("zynthe").joinpath("configs", _DEFAULTS_FILENAME))
    except Exception:
        logger.debug("Failed to resolve packaged config via importlib.resources")

    here = os.path.abspath(os.path.dirname(__file__))
    candidates.extend(
        [
            os.path.join(here, "..", "..", "configs", _DEFAULTS_FILENAME),
            os.path.join(here, "..", "..", "..", "..", "configs", _DEFAULTS_FILENAME),
        ]
    )

    for candidate in candidates:
        candidate_path = os.fspath(candidate)
        if os.path.exists(candidate_path):
            return candidate_path

    return os.fspath(candidates[0]) if candidates else _DEFAULTS_FILENAME


def _get_env_info() -> Dict[str, Any]:
    info = {
        "python": str(platform.python_version()),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": getattr(torch.backends, "mps", None) is not None
        and getattr(torch.backends.mps, "is_available", lambda: False)(),
    }
    try:
        # best-effort: CUDA device names if available
        if torch.cuda.is_available():
            names = []
            for i in range(torch.cuda.device_count()):
                names.append(torch.cuda.get_device_name(i))
            info["cuda_device_names"] = names
    except Exception:
        logger.debug("Failed to query CUDA device names")
    return info


def _get_git_info() -> Dict[str, Optional[str]]:
    """Best-effort minimal git info: commit sha and branch."""
    out: Dict[str, Optional[str]] = {"commit_sha": None, "branch": None}
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        out["commit_sha"] = sha
    except Exception:
        logger.debug("Failed to get git commit SHA")
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        out["branch"] = branch
    except Exception:
        logger.debug("Failed to get git branch")
    return out


class ConfigManager:
    """
    Central configuration manager.
    """

    REQUIRED_SECTIONS = {
        "train": ["epochs", "batch_size", "lr"],
        "model": ["name", "type"],  # student_name and tokenizer_name are optional with defaults
        "distillation": [],  # flexible, so no strict keys
        "data": ["train_path", "val_path"],
    }

    def _validate_minimal(self):
        missing_sections = [s for s in self.REQUIRED_SECTIONS if s not in self.raw_config]
        if missing_sections:
            raise ConfigError(f"Missing required config sections: {missing_sections}")

        # deeper check for critical keys
        for section, required_keys in self.REQUIRED_SECTIONS.items():
            if not required_keys:
                continue
            section_cfg = self.raw_config.get(section, {})
            missing_keys = [k for k in required_keys if k not in section_cfg]
            if missing_keys:
                raise ConfigError(f"Section '{section}' missing required keys: {missing_keys}")

        # Additional model configuration validation and defaults
        model_cfg = self.raw_config.get("model", {})

        # Set student_name default if not specified
        if "student_name" not in model_cfg:
            model_cfg["student_name"] = model_cfg.get("name")
            logger.info(
                f"student_name not specified, defaulting to teacher model: {model_cfg['student_name']}"
            )

        # Set tokenizer_name default if not specified
        if "tokenizer_name" not in model_cfg:
            model_cfg["tokenizer_name"] = model_cfg.get("name")
            logger.info(
                f"tokenizer_name not specified, defaulting to teacher model: {model_cfg['tokenizer_name']}"
            )

        self.raw_config["model"] = model_cfg

    def __init__(
        self,
        config_path: Optional[str] = None,
        defaults_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        experiments_root: str = "experiments",
    ):
        self.config_path = config_path
        env_root = os.environ.get("ZYNTHÉ_EXPERIMENTS_ROOT")
        if env_root:
            experiments_root = env_root
        # Prefer packaged defaults; fall back to the repository config in editable installs.
        if defaults_path is None:
            defaults_path = _default_config_path()
        self.defaults_path = defaults_path
        self.overrides = overrides.copy() if overrides else {}
        self.experiments_root = os.path.abspath(os.path.expanduser(experiments_root))

        self.raw_config: Dict[str, Any] = {}
        self.resolved_config: Dict[str, Any] = {}
        self.experiment_id: str = ""
        self.experiment_dir: str = ""
        self.paths: Dict[str, str] = {}
        self.env_info: Dict[str, Any] = {}
        self.git_info: Dict[str, Optional[str]] = {}

        self._load_and_resolve()

    # -------------------------
    # Loading & merging
    # -------------------------
    def _normalize_legacy_aliases(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize legacy/new key aliases so downstream modules read a stable schema."""
        normalized = cfg.copy()

        # Distillation key normalization
        distill_cfg = normalized.setdefault("distillation", {})
        if not isinstance(distill_cfg, dict):
            distill_cfg = {}
            normalized["distillation"] = distill_cfg

        method = distill_cfg.get("method") or distill_cfg.get("type") or "kd_hinton"
        distill_cfg["method"] = method
        distill_cfg.setdefault("type", method)

        nested_cfg = distill_cfg.get("config")
        if not isinstance(nested_cfg, dict):
            nested_cfg = {}

        for key in (
            "temperature",
            "alpha",
            "label_smoothing",
            "hint_enabled",
            "confidence_scaling",
            "min_confidence",
        ):
            if key in distill_cfg and key not in nested_cfg:
                nested_cfg[key] = distill_cfg[key]

        kd_hinton_cfg = distill_cfg.get("kd_hinton")
        if isinstance(kd_hinton_cfg, dict) and "kd_hinton" not in nested_cfg:
            nested_cfg["kd_hinton"] = kd_hinton_cfg.copy()

        distill_cfg["config"] = nested_cfg

        # Train key normalization
        train_cfg = normalized.setdefault("train", {})
        if not isinstance(train_cfg, dict):
            train_cfg = {}
            normalized["train"] = train_cfg

        if "gradient_accumulation_steps" not in train_cfg and "grad_accum_steps" in train_cfg:
            train_cfg["gradient_accumulation_steps"] = train_cfg["grad_accum_steps"]

        if "use_amp" not in train_cfg and "mixed_precision" in train_cfg:
            train_cfg["use_amp"] = bool(train_cfg["mixed_precision"])

        if "num_epochs" not in train_cfg and "epochs" in train_cfg:
            train_cfg["num_epochs"] = int(train_cfg["epochs"])

        # Multi-platform adapter configuration
        adapter_cfg = normalized.setdefault("adapters", {})
        if not isinstance(adapter_cfg, dict):
            adapter_cfg = {}
            normalized["adapters"] = adapter_cfg
        adapter_cfg.setdefault("auto_detect", False)
        adapter_cfg.setdefault("teacher_type", "auto")
        adapter_cfg.setdefault("student_type", "auto")

        return normalized

    def _load_and_resolve(self):
        defaults = {}
        if os.path.exists(self.defaults_path):
            try:
                defaults = _safe_load_yaml(self.defaults_path)
                logger.info(f"Loaded default config from: {self.defaults_path}")
            except Exception as e:
                raise ConfigError(f"Failed to load defaults from {self.defaults_path}: {e}")
        else:
            logger.warning(
                f"Default config not found at: {self.defaults_path}, using minimal defaults"
            )

        user_cfg = {}
        if self.config_path:
            if not os.path.exists(self.config_path):
                raise ConfigError(f"Config file not found: {self.config_path}")
            try:
                user_cfg = _safe_load_yaml(self.config_path)
                logger.info(f"Loaded user config from: {self.config_path}")
            except Exception as e:
                raise ConfigError(f"Failed to load user config from {self.config_path}: {e}")

        merged = _deep_update(deepcopy(defaults), user_cfg)
        merged = _deep_update(merged, self.overrides or {})
        self.raw_config = self._normalize_legacy_aliases(merged)

        self._validate_minimal()
        self._resolve_runtime()
        self._create_experiment_dirs()
        self._save_resolved_config()

    # -------------------------
    # Runtime resolution
    # -------------------------
    def _resolve_runtime(self):
        # device detection: prefer MPS on Mac M1/M2, then CUDA, else CPU
        prefer_cuda = bool(self.raw_config.get("device", {}).get("prefer_cuda", True))
        prefer_mps = bool(self.raw_config.get("device", {}).get("prefer_mps", False))

        mps_available = (
            getattr(torch.backends, "mps", None) is not None
            and getattr(torch.backends.mps, "is_available", lambda: False)()
        )
        cuda_available = torch.cuda.is_available()

        if prefer_mps and mps_available:
            resolved_device = "mps"
        elif prefer_cuda and cuda_available:
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"

        # sensible M2 defaults: smaller batch, no default AMP (MPS supports float32)
        default_train = {
            "epochs": 3,
            "batch_size": 8 if resolved_device == "mps" else 32,
            "lr": 5e-5,
            "grad_accum_steps": 1,
            "mixed_precision": False if resolved_device == "mps" else True,
            "early_stop_patience": 2,
        }

        train_cfg = self.raw_config.get("train", {})
        merged_train = _deep_update(default_train, train_cfg)

        # ensure integer coercion
        merged_train["epochs"] = int(merged_train.get("epochs", 3))
        merged_train["batch_size"] = int(merged_train.get("batch_size", 8))
        merged_train["grad_accum_steps"] = int(merged_train.get("grad_accum_steps", 1))
        merged_train["early_stop_patience"] = int(merged_train.get("early_stop_patience", 2))
        merged_train["mixed_precision"] = bool(merged_train.get("mixed_precision", False))
        merged_train["lr"] = _coerce_float(merged_train.get("lr", 5e-5), 5e-5)

        # Keep learning_rate/lr aliases in sync and always numeric.
        # Some configs use learning_rate while legacy paths read lr.
        if "learning_rate" in merged_train:
            merged_train["learning_rate"] = _coerce_float(
                merged_train.get("learning_rate"),
                merged_train["lr"],
            )
            merged_train["lr"] = merged_train["learning_rate"]
        else:
            merged_train["learning_rate"] = merged_train["lr"]

        # explainability defaults (SHAP / LIME)
        explain_cfg = self.raw_config.get("explainability", {})
        default_explain = {
            "enable_shap": False,
            "enable_lime": False,
            "shap_samples": 100,
            "lime_samples": 500,
        }
        merged_explain = _deep_update(default_explain, explain_cfg)

        # quantization defaults
        quant_cfg = self.raw_config.get("quantization", {})
        default_quant = {
            "enable": False,
            "mode": "ptq",  # ptq or qat
        }
        merged_quant = _deep_update(default_quant, quant_cfg)

        # similarity transfer default
        similarity_transfer = bool(self.raw_config.get("similarity_transfer", False))

        # final resolved config
        self.resolved_config = dict(self.raw_config)  # shallow copy
        self.resolved_config["train"] = merged_train
        self.resolved_config["explainability"] = merged_explain
        self.resolved_config["quantization"] = merged_quant
        self.resolved_config["similarity_transfer"] = similarity_transfer
        self.resolved_config.setdefault("runtime", {})
        self.resolved_config["runtime"]["device"] = resolved_device
        self.resolved_config["runtime"]["execution_mode"] = "manual_non_agentic"
        self.resolved_config["runtime"]["distiller_default"] = "kd_hinton"

        # seed handling
        seed = int(self.raw_config.get("seed", 42))
        self.set_seed(seed)
        self.resolved_config["runtime"]["seed"] = seed

        # environment metadata
        self.env_info = _get_env_info()
        self.resolved_config["_env"] = self.env_info
        self.git_info = _get_git_info()
        self.resolved_config["_git"] = self.git_info

    def set_seed(self, seed: int):
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -------------------------
    # Experiment directories
    # -------------------------
    def _create_experiment_dirs(self):
        self.paths = {}
        base = self.raw_config.get("output_root", self.experiments_root)
        base = os.path.abspath(os.path.expanduser(base))
        _ensure_dir(base)

        # experiment id uses timestamp + short uuid
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short = uuid.uuid4().hex[:8]
        self.experiment_id = f"{ts}_{short}"
        exp_dir = os.path.join(base, self.experiment_id)
        if os.path.exists(exp_dir):
            # make unique
            exp_dir = f"{exp_dir}_{uuid.uuid4().hex[:6]}"
        _ensure_dir(exp_dir)
        self.experiment_dir = exp_dir

        # standard subdirs
        subdirs = ["checkpoints", "logs", "tensorboard", "snapshots"]
        for s in subdirs:
            p = os.path.join(exp_dir, s)
            _ensure_dir(p)
            self.paths[s] = p

        # resolved config path
        self.paths["resolved_config"] = os.path.join(exp_dir, "resolved_config.yaml")

        # copy user config for traceability (best-effort)
        if self.config_path:
            try:
                shutil.copy(
                    self.config_path, os.path.join(exp_dir, os.path.basename(self.config_path))
                )
            except Exception:
                logger.warning("Failed to copy user config to experiment dir (non-fatal)")

    def _save_resolved_config(self):
        payload = dict(self.resolved_config)
        meta: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "config_version": "v1.0",
        }
        meta["git"] = self.git_info
        payload["_meta"] = meta
        try:
            with open(self.paths["resolved_config"], "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
        except Exception as e:
            logger.warning("Failed to write resolved config: %s", e)

    # -------------------------
    # Public API
    # -------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.resolved_config.get(key, default)

    def get_runtime(self) -> Dict[str, Any]:
        return self.resolved_config.get("runtime", {})

    def experiment_info(self) -> Dict[str, Any]:
        return {
            "id": self.experiment_id,
            "dir": self.experiment_dir,
            "paths": self.paths,
            "runtime": self.get_runtime(),
            "env": self.env_info,
            "git": self.git_info,
        }

    def override(self, overrides: Dict[str, Any]):
        if not overrides:
            return
        self.overrides = _deep_update(self.overrides, overrides)
        self._load_and_resolve()

    # -------------------------
    # Convenience helpers
    # -------------------------
    def device(self) -> torch.device:
        """
        Return a torch.device for runtime.device.
        """
        dev = self.get_runtime().get("device", "cpu")
        if dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if dev == "mps":
            return torch.device("mps")
        return torch.device("cpu")

    def get_nested(self, *keys, default: Any = None) -> Any:
        """
        Safely get nested config values using dot notation.

        Example:
            cfg.get_nested("model", "name") -> returns cfg["model"]["name"]
            cfg.get_nested("train", "lr", default=1e-5)
        """
        current = self.resolved_config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)  # type: ignore[assignment]
                if current is None:
                    return default
            else:
                return default
        return current if current is not None else default

    def validate_required_paths(self):
        """
        Validate that required data paths exist.
        """
        train_path = self.get_nested("data", "train_path")
        val_path = self.get_nested("data", "val_path")

        missing = []
        if train_path and not os.path.exists(train_path):
            missing.append(f"Training data: {train_path}")
        if val_path and not os.path.exists(val_path):
            missing.append(f"Validation data: {val_path}")

        if missing:
            raise ConfigError(
                "Missing required data files:\n" + "\n".join(f"  - {m}" for m in missing)
            )

        logger.info("All required data paths validated successfully")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ConfigManager(\n"
            f"  experiment_id={self.experiment_id},\n"
            f"  device={self.get_runtime().get('device', 'unknown')},\n"
            f"  config_path={self.config_path},\n"
            f"  defaults_path={self.defaults_path}\n"
            f")"
        )
