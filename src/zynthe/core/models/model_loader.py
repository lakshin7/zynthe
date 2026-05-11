
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    CLIPModel,
    AutoTokenizer,
    PreTrainedModel,
)

if TYPE_CHECKING:
    from zynthe.core.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses & helpers
# ---------------------------------------------------------------------------


@dataclass
class ModelLoadSpec:
    """Declarative specification describing how to load teacher/student."""

    teacher_name: str
    student_name: str
    tokenizer_name: str
    model_type: str
    hf_token: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    revision: Optional[str] = None
    compile_graph: bool = False
    gradient_checkpointing: bool = False
    torch_dtype: Optional[str] = None
    quantization: Optional[str] = None  # e.g. "int8", "bnb-4bit"
    quantization_target: str = "both"  # teacher|student|both
    quantization_safe_fallback: bool = True
    adapter_path: Optional[str] = None
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    return_wrapped: bool = False
    compile_mode: str = "inference"  # inference|training
    compile_roles: Tuple[str, ...] = ("student",)


@dataclass
class ModelBundle:
    """Container returned by the enhanced loader."""

    teacher: PreTrainedModel
    student: PreTrainedModel
    tokenizer: Any
    spec: ModelLoadSpec
    device: torch.device
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Multi-platform adapter support (populated when auto_detect is enabled)
    teacher_adapter: Optional[Any] = None  # Optional[ModelAdapter]
    student_adapter: Optional[Any] = None  # Optional[ModelAdapter]

    def summary(self) -> Dict[str, Any]:
        summary = {
            "teacher": _summarize_model(self.teacher),
            "student": _summarize_model(self.student),
            "tokenizer": self.tokenizer.__class__.__name__,
            "device": str(self.device),
            "quantization": self.spec.quantization,
            "compile": self.spec.compile_graph,
        }
        if self.teacher_adapter is not None:
            summary["teacher_adapter"] = repr(self.teacher_adapter)
        if self.student_adapter is not None:
            summary["student_adapter"] = repr(self.student_adapter)
        return summary


def _cfg_get(cfg: Union["ConfigManager", Dict[str, Any]], key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _summarize_model(model: PreTrainedModel) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    name = getattr(config, "name_or_path", model.__class__.__name__)
    model_type = getattr(config, "model_type", "unknown")
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": name,
        "type": model_type,
        "parameters": int(param_count),
        "trainable_parameters": int(trainable),
    }


def _to_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if not dtype_name:
        return None
    name = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name, None)


def resolve_device(device_hint: Optional[Union[str, torch.device]] = None) -> torch.device:
    if isinstance(device_hint, torch.device):
        return device_hint
    if isinstance(device_hint, str) and device_hint:
        return torch.device(device_hint)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


class ModelLoader:
    """High-level coordinator for enterprise-grade model loading."""

    def __init__(
        self,
        cfg: Union["ConfigManager", Dict[str, Any]],
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.cfg = cfg
        self.device = resolve_device(device)
        self.model_cfg = _cfg_get(cfg, "model", {}) or {}
        self.runtime_cfg = _cfg_get(cfg, "runtime", {}) or {}
        self.hf_token = self._resolve_hf_token()
        self._login_if_needed()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        return_bundle: bool = False,
    ) -> Union[Tuple[PreTrainedModel, PreTrainedModel, Any], ModelBundle]:
        spec = self._build_spec()
        bundle = self._load_bundle(spec)
        return bundle if (return_bundle or spec.return_wrapped) else (
            bundle.teacher,
            bundle.student,
            bundle.tokenizer,
        )

    def resume_from_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        strict: bool = True,
        map_location: Optional[str] = None,
    ) -> Tuple[PreTrainedModel, Any]:
        """
        Load model/tokenizer from a checkpoint directory and optionally restore
        optimizer/scheduler/scaler state from a serialized training checkpoint.
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

        logger.info("Resuming model from checkpoint: %s", checkpoint_dir)

        # Determine model class based on config
        model_type = (self.model_cfg.get("type") or "").lower()
        if model_type in {"causal_lm", "causallm", "causal-lm", "lm"}:
            model_class = AutoModelForCausalLM
        elif model_type in {"vision", "image", "image_classification", "image-classification"}:
            model_class = AutoModelForImageClassification
        elif model_type in {"multimodal", "clip"}:
            model_class = CLIPModel
        elif model_type in {"sequence_classification", "sequenceclassification", "classification"}:
            model_class = AutoModelForSequenceClassification
        else:
            model_class = AutoModel

        resolved_location = map_location or str(self.device)
        model = model_class.from_pretrained(str(checkpoint_path), torch_dtype=self.model_cfg.get("torch_dtype"))

        try:
            tokenizer = self._load_artifact_processor(str(checkpoint_path), model_type)
        except Exception as e:
            logger.warning("Failed to load tokenizer from checkpoint, falling back to base tokenizer: %s", e)
            spec = self._build_spec()
            tokenizer = self._load_tokenizer(spec)

        resume_state: Dict[str, Any] = {
            "restored_optimizer": False,
            "restored_scheduler": False,
            "restored_scaler": False,
            "epoch": None,
            "global_step": None,
            "best_metric": None,
        }

        # Restore extended training state when present.
        state_candidates = [
            checkpoint_path / "checkpoint.pt",
            checkpoint_path / "latest.pt",
            checkpoint_path / "best.pt",
            checkpoint_path / "training_state.pt",
        ]
        state_path = next((candidate for candidate in state_candidates if candidate.exists()), None)
        if state_path is not None:
            try:
                from zynthe.core.models.model_saver import load_checkpoint

                payload, metadata = load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    path=str(state_path),
                    scheduler=scheduler,
                    scaler=scaler,
                    map_location=resolved_location,
                    strict=strict,
                )
                resume_state["restored_optimizer"] = bool(optimizer is not None and "optimizer_state_dict" in payload)
                resume_state["restored_scheduler"] = bool(scheduler is not None and "scheduler_state_dict" in payload)
                resume_state["restored_scaler"] = bool(scaler is not None and "scaler_state_dict" in payload)
                if metadata is not None:
                    resume_state["epoch"] = int(metadata.epoch)
                    resume_state["global_step"] = int(metadata.global_step)
                    resume_state["best_metric"] = metadata.best_metric
            except Exception as exc:
                logger.warning("Checkpoint state restoration failed for %s: %s", state_path, exc)

        self._last_resume_state = resume_state
        return model, tokenizer

    def get_last_resume_state(self) -> Dict[str, Any]:
        """Return details about the last checkpoint resume attempt."""
        return dict(getattr(self, "_last_resume_state", {}))

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def _build_spec(self) -> ModelLoadSpec:
        teacher_name = self.model_cfg.get("name") or self.model_cfg.get("teacher_name") or self.model_cfg.get("teacher")
        student_name = self.model_cfg.get("student_name", teacher_name)
        tokenizer_name = (
            self.model_cfg.get("tokenizer_name")
            or self.model_cfg.get("processor_name")
            or teacher_name
        )
        model_type = (self.model_cfg.get("type") or "").lower()

        if not teacher_name:
            raise ValueError(
                "Teacher model name must be specified in config "
                "(model.name, model.teacher_name, or model.teacher)"
            )

        quant_cfg = self.model_cfg.get("quantization", {}) or {}
        runtime = self.runtime_cfg.get("distillation", {}) or {}
        compile_roles_cfg = self.model_cfg.get(
            "compile_roles",
            runtime.get("compile_roles", ["student"]),
        )
        if isinstance(compile_roles_cfg, str):
            compile_roles = (compile_roles_cfg.lower(),)  # type: ignore[assignment]
        elif isinstance(compile_roles_cfg, Iterable):
            compile_roles = tuple(str(role).lower() for role in compile_roles_cfg)  # type: ignore[assignment]
        else:
            compile_roles = ("student",)  # type: ignore[assignment]

        spec = ModelLoadSpec(
            teacher_name=teacher_name,
            student_name=student_name or teacher_name,
            tokenizer_name=tokenizer_name or teacher_name,
            model_type=model_type or "transformer",
            hf_token=self.hf_token,
            trust_remote_code=self.model_cfg.get("trust_remote_code", False),
            local_files_only=self.model_cfg.get("local_files_only", False),
            revision=self.model_cfg.get("revision"),
            compile_graph=runtime.get("compile", False) or self.model_cfg.get("compile", False),
            gradient_checkpointing=self.model_cfg.get("gradient_checkpointing", False),
            torch_dtype=self.model_cfg.get("torch_dtype"),
            quantization=quant_cfg.get("type"),
            quantization_target=str(quant_cfg.get("target", quant_cfg.get("apply_to", "both"))).lower(),
            quantization_safe_fallback=bool(quant_cfg.get("safe_fallback", True)),
            adapter_path=quant_cfg.get("adapter"),
            extra_model_kwargs=self.model_cfg.get("model_kwargs", {}) or {},
            extra_tokenizer_kwargs=self.model_cfg.get("tokenizer_kwargs", {}) or {},
            return_wrapped=self.model_cfg.get("return_bundle", False),
            compile_mode=str(
                runtime.get("compile_mode", self.model_cfg.get("compile_mode", "inference"))
            ).lower(),
            compile_roles=compile_roles,
        )

        return spec

    # ------------------------------------------------------------------
    # Loading logic
    # ------------------------------------------------------------------

    def _load_bundle(
        self,
        spec: ModelLoadSpec,
    ) -> ModelBundle:
        model_class, model_kwargs = self._resolve_model_class(spec)
        tokenizer = self._load_tokenizer(spec)

        teacher = self._instantiate_model(
            model_class,
            spec.teacher_name,
            spec,
            role="teacher",
            **model_kwargs,
        )
        student = self._instantiate_model(
            model_class,
            spec.student_name,
            spec,
            role="student",
            **model_kwargs,
        )

        metadata = {
            "summary": {
                "teacher": _summarize_model(teacher),
                "student": _summarize_model(student),
            },
            "device": str(self.device),
            "used_agent": False,
            "quantization": spec.quantization,
            "quantization_target": spec.quantization_target,
        }

        if spec.gradient_checkpointing:
            for mdl in (teacher, student):
                try:
                    mdl.gradient_checkpointing_enable()  # type: ignore[attr-defined]
                    features = metadata.setdefault("features", [])  # type: ignore[assignment]
                    features.append("gradient_checkpointing")  # type: ignore[attr-defined]
                except AttributeError:
                    logger.warning("Gradient checkpointing requested but model does not support it")

        if spec.compile_graph:
            compiled_roles = []
            if "teacher" in spec.compile_roles:
                maybe_teacher = _maybe_compile(
                    teacher,
                    compile_mode=spec.compile_mode,
                    quantization=spec.quantization,
                    role="teacher",
                )
                if maybe_teacher is not teacher:
                    compiled_roles.append("teacher")
                teacher = maybe_teacher
            if "student" in spec.compile_roles:
                maybe_student = _maybe_compile(
                    student,
                    compile_mode=spec.compile_mode,
                    quantization=spec.quantization,
                    role="student",
                )
                if maybe_student is not student:
                    compiled_roles.append("student")
                student = maybe_student
            if compiled_roles:
                features = metadata.setdefault("features", [])  # type: ignore[assignment]
                features.append("torch.compile")  # type: ignore[attr-defined]
                metadata["compiled_roles"] = compiled_roles

        if spec.adapter_path:
            student = _maybe_attach_adapter(student, spec.adapter_path)

        bundle = ModelBundle(
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            spec=spec,
            device=self.device,
            metadata=metadata,
        )

        return bundle

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_class(self, spec: ModelLoadSpec) -> Tuple[type, Dict[str, Any]]:
        model_kwargs: Dict[str, Any] = dict(spec.extra_model_kwargs)

        dtype = _to_torch_dtype(spec.torch_dtype)
        if dtype is not None:
            model_kwargs.setdefault("torch_dtype", dtype)

        resolved_model_type = spec.model_type
        if resolved_model_type in {"", "auto", "transformer"}:
            try:
                cfg = AutoConfig.from_pretrained(
                    spec.teacher_name,
                    trust_remote_code=spec.trust_remote_code,
                    local_files_only=spec.local_files_only,
                    token=spec.hf_token,
                )
                architectures = [arch.lower() for arch in (getattr(cfg, 'architectures', None) or [])]
                if getattr(cfg, "is_decoder", False) or any("causal" in arch for arch in architectures):
                    resolved_model_type = "causallm"
                else:
                    resolved_model_type = "sequenceclassification"
            except Exception as exc:
                logger.warning("Auto model type resolution failed (%s); falling back to sequence classification", exc)
                resolved_model_type = "sequenceclassification"

        if resolved_model_type in {"vision", "image", "image_classification", "image-classification", "vit", "deit"}:
            model_class = AutoModelForImageClassification
            if "num_labels" in self.model_cfg:
                model_kwargs.setdefault("num_labels", self.model_cfg.get("num_labels"))
            if "label2id" in self.model_cfg:
                label_map = self.model_cfg.get("label2id", {})
                model_kwargs.setdefault("label2id", label_map)
                model_kwargs.setdefault("id2label", {idx: name for name, idx in label_map.items()})
        elif resolved_model_type in {"multimodal", "clip"}:
            model_class = CLIPModel
        elif resolved_model_type in {"causallm", "causal_lm", "causal-lm", "decoder", "gpt", "lm", "language_modeling"}:
            model_class = AutoModelForCausalLM
        elif resolved_model_type in {"sequenceclassification", "sequence_classification", "classification", "transformer"}:
            model_class = AutoModelForSequenceClassification
            if "num_labels" in self.model_cfg:
                model_kwargs.setdefault("num_labels", self.model_cfg.get("num_labels", 2))
            if "label2id" in self.model_cfg:
                label_map = self.model_cfg.get("label2id", {"negative": 0, "positive": 1})
                model_kwargs.setdefault("label2id", label_map)
                model_kwargs.setdefault(
                    "id2label",
                    {idx: name for name, idx in label_map.items()},
                )
        else:
            model_class = AutoModel

        # Attach revision/local flags
        if spec.revision:
            model_kwargs.setdefault("revision", spec.revision)
        model_kwargs.setdefault("trust_remote_code", spec.trust_remote_code)
        model_kwargs.setdefault("local_files_only", spec.local_files_only)

        if spec.hf_token:
            model_kwargs.setdefault("token", spec.hf_token)

        return model_class, model_kwargs

    def _load_tokenizer(self, spec: ModelLoadSpec) -> Any:
        tokenizer_kwargs = dict(spec.extra_tokenizer_kwargs)
        tokenizer_kwargs.setdefault("use_fast", True)
        tokenizer_kwargs.setdefault("trust_remote_code", spec.trust_remote_code)
        tokenizer_kwargs.setdefault("local_files_only", spec.local_files_only)
        if spec.hf_token:
            tokenizer_kwargs.setdefault("token", spec.hf_token)

        model_type = spec.model_type.lower()
        if model_type in {"vision", "image", "image_classification", "image-classification"}:
            processor_kwargs = dict(tokenizer_kwargs)
            processor_kwargs.pop("use_fast", None)
            return AutoImageProcessor.from_pretrained(spec.tokenizer_name, **processor_kwargs)

        if model_type in {"multimodal", "clip", "vlm"}:
            return AutoProcessor.from_pretrained(spec.tokenizer_name, **tokenizer_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_name, **tokenizer_kwargs)

        # Ensure padding token for decoder-only models
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = self.model_cfg.get("padding_side", tokenizer.padding_side)

        return tokenizer

    def _load_artifact_processor(self, path: str, model_type: str) -> Any:
        normalized_type = model_type.lower()
        if normalized_type in {"vision", "image", "image_classification", "image-classification"}:
            return AutoImageProcessor.from_pretrained(path)
        if normalized_type in {"multimodal", "clip", "vlm"}:
            return AutoProcessor.from_pretrained(path)
        return AutoTokenizer.from_pretrained(path)

    def _instantiate_model(
        self,
        model_class: type,
        name: str,
        spec: ModelLoadSpec,
        role: str,
        **model_kwargs: Any,
    ) -> PreTrainedModel:
        role_kwargs = dict(model_kwargs)
        quant_target = spec.quantization_target.lower()
        quant_enabled_for_role = quant_target in {"both", role}

        if spec.quantization and quant_enabled_for_role:
            q = spec.quantization.lower()
            if q in {"int8", "dynamic"}:
                role_kwargs.setdefault("torch_dtype", torch.float32)
                role_kwargs.setdefault("load_in_8bit", True)
            elif q in {"bnb-4bit", "4bit"}:
                role_kwargs.setdefault("load_in_4bit", True)

            # Safety: quantized loading is most stable on CUDA backends.
            if (
                (role_kwargs.get("load_in_8bit") or role_kwargs.get("load_in_4bit"))
                and self.device.type != "cuda"
            ):
                if spec.quantization_safe_fallback:
                    logger.warning(
                        "Quantization requested for %s on device '%s'; disabling quantization for safety.",
                        role,
                        self.device,
                    )
                    role_kwargs.pop("load_in_8bit", None)
                    role_kwargs.pop("load_in_4bit", None)
                else:
                    raise RuntimeError(
                        f"Quantization requested for role={role} requires CUDA. device={self.device}"
                    )

        logger.info(
            "Loading %s model '%s' (type=%s) on %s",
            role,
            name,
            spec.model_type,
            self.device,
        )
        model = model_class.from_pretrained(name, **role_kwargs)  # type: ignore[attr-defined]

        # With quantized loaders the dispatch is often handled by backend/device_map.
        is_quantized = bool(role_kwargs.get("load_in_8bit") or role_kwargs.get("load_in_4bit"))
        if not is_quantized:
            model.to(self.device)

        if role == "teacher":
            model.eval()
        else:
            default_train_mode = bool(self.model_cfg.get("student_train_mode", True))
            model.train(default_train_mode)
        return model

    def _resolve_hf_token(self) -> Optional[str]:
        token = self.model_cfg.get("hf_token") or os.getenv("HF_TOKEN")
        if token and token.strip():
            return token.strip()
        return None

    def _login_if_needed(self) -> None:
        if not self.hf_token:
            return
        try:
            from huggingface_hub import login

            login(token=self.hf_token, add_to_git_credential=False)
            logger.info("Logged into HuggingFace Hub with provided token")
        except Exception as exc:  # pragma: no cover - best effort login
            logger.warning("HuggingFace login failed: %s", exc)




def _maybe_compile(
    model: PreTrainedModel,
    *,
    compile_mode: str = "inference",
    quantization: Optional[str] = None,
    role: str = "student",
) -> PreTrainedModel:
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available in this PyTorch installation")
        return model
    if quantization:
        logger.warning("Skipping torch.compile for quantized %s model", role)
        return model

    compile_args: Dict[str, Any] = {"dynamic": True}
    if compile_mode == "training":
        compile_args["mode"] = "max-autotune"
    else:
        compile_args["mode"] = "reduce-overhead"

    try:
        compiled = torch.compile(model, **compile_args)  # type: ignore[arg-type]
        logger.info("Model compiled with torch.compile (role=%s, mode=%s)", role, compile_mode)
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - depends on version
        logger.warning("torch.compile failed: %s", exc)
        return model


def _maybe_attach_adapter(model: PreTrainedModel, adapter_path: str) -> PreTrainedModel:
    if not adapter_path:
        return model
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        logger.warning("Adapter path %s does not exist", adapter_path)
        return model

    try:
        from peft import PeftModel  # type: ignore[import]
    except ImportError:
        logger.warning("PEFT not installed; skipping adapter load")
        return model

    try:
        wrapped = PeftModel.from_pretrained(model, str(adapter_path_obj))
        logger.info("Loaded PEFT adapter from %s", adapter_path)
        return cast(PreTrainedModel, wrapped)
    except Exception as exc:
        logger.warning("Failed to load adapter %s: %s", adapter_path, exc)
        return model


# ---------------------------------------------------------------------------
# Backwards compatible helpers
# ---------------------------------------------------------------------------


def get_device(device_str: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Backward compatible alias for resolve_device."""
    return resolve_device(device_str)


def load_models(
    cfg: Union["ConfigManager", Dict[str, Any]],
    device: Optional[Union[str, torch.device]] = None,
    return_bundle: bool = False,
):
    """Convenience function that creates a ModelLoader and loads models."""

    loader = ModelLoader(cfg=cfg, device=device)
    bundle = cast(
        ModelBundle,
        loader.load(return_bundle=True),
    )

    # Auto-detect adapters if config requests it
    adapter_cfg = (
        _cfg_get(cfg, "adapters", {})
        if not isinstance(cfg, dict)
        else cfg.get("adapters", {})
    )
    if isinstance(adapter_cfg, dict) and adapter_cfg.get("auto_detect", False):
        try:
            from zynthe.core.adapters import AdapterRegistry
            registry = AdapterRegistry()
            bundle.teacher_adapter = registry.detect(bundle.teacher)
            bundle.student_adapter = registry.detect(bundle.student)
            logger.info(
                "Auto-detected adapters: teacher=%s, student=%s",
                bundle.teacher_adapter, bundle.student_adapter,
            )
        except ImportError:
            logger.warning("core.adapters not available; skipping adapter detection")

    if return_bundle:
        return bundle
    return bundle.teacher, bundle.student, bundle.tokenizer


def model_summary(model: PreTrainedModel) -> Dict[str, Any]:
    """Expose the enhanced summary helper for external callers."""
    return _summarize_model(model)


from zynthe.core.models.model_saver import ModelSaver
