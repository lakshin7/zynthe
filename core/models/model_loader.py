import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

if TYPE_CHECKING:
    from core.config.config_manager import ConfigManager

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
    adapter_path: Optional[str] = None
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    return_wrapped: bool = False


@dataclass
class ModelBundle:
    """Container returned by the enhanced loader."""

    teacher: PreTrainedModel
    student: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    spec: ModelLoadSpec
    device: torch.device
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "teacher": _summarize_model(self.teacher),
            "student": _summarize_model(self.student),
            "tokenizer": self.tokenizer.__class__.__name__,
            "device": str(self.device),
            "quantization": self.spec.quantization,
            "compile": self.spec.compile_graph,
        }


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
        use_agent: bool = True,
        data_samples: Optional[List[Dict[str, Any]]] = None,
        return_bundle: bool = False,
    ) -> Union[Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase], ModelBundle]:
        spec = self._build_spec(use_agent=use_agent, data_samples=data_samples)
        bundle = self._load_bundle(spec, data_samples=data_samples, use_agent=use_agent)
        return bundle if (return_bundle or spec.return_wrapped) else (
            bundle.teacher,
            bundle.student,
            bundle.tokenizer,
        )

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def _build_spec(
        self,
        use_agent: bool,
        data_samples: Optional[List[Dict[str, Any]]],
    ) -> ModelLoadSpec:
        teacher_name = self.model_cfg.get("name")
        student_name = self.model_cfg.get("student_name", teacher_name)
        tokenizer_name = self.model_cfg.get("tokenizer_name", teacher_name)
        model_type = (self.model_cfg.get("type") or "").lower()

        if not teacher_name and use_agent:
            teacher_name, tokenizer_name, data_samples = self._run_teacher_agent(data_samples)
            if not student_name:
                student_name = teacher_name
        elif not teacher_name:
            raise ValueError(
                "Teacher model name must be specified or enable use_agent=True for auto-selection"
            )

        quant_cfg = self.model_cfg.get("quantization", {}) or {}
        runtime = self.runtime_cfg.get("distillation", {}) or {}

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
            adapter_path=quant_cfg.get("adapter"),
            extra_model_kwargs=self.model_cfg.get("model_kwargs", {}) or {},
            extra_tokenizer_kwargs=self.model_cfg.get("tokenizer_kwargs", {}) or {},
            return_wrapped=self.model_cfg.get("return_bundle", False),
        )

        return spec

    # ------------------------------------------------------------------
    # Loading logic
    # ------------------------------------------------------------------

    def _load_bundle(
        self,
        spec: ModelLoadSpec,
        data_samples: Optional[List[Dict[str, Any]]],
        use_agent: bool,
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
            "used_agent": bool(use_agent and self.model_cfg.get("name") is None),
            "quantization": spec.quantization,
        }

        if spec.gradient_checkpointing:
            for mdl in (teacher, student):
                try:
                    mdl.gradient_checkpointing_enable()  # type: ignore[attr-defined]
                    metadata.setdefault("features", []).append("gradient_checkpointing")
                except AttributeError:
                    logger.warning("Gradient checkpointing requested but model does not support it")

        if spec.compile_graph:
            metadata.setdefault("features", []).append("torch.compile")
            teacher = _maybe_compile(teacher)
            student = _maybe_compile(student)

        if spec.adapter_path:
            _maybe_attach_adapter(student, spec.adapter_path)

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

        if spec.quantization:
            if spec.quantization.lower() in {"int8", "dynamic"}:
                model_kwargs.setdefault("torch_dtype", torch.float32)
                model_kwargs.setdefault("load_in_8bit", True)
            elif spec.quantization.lower() in {"bnb-4bit", "4bit"}:
                model_kwargs.setdefault("load_in_4bit", True)

        if spec.model_type in {"causallm", "decoder", "gpt", "lm"}:
            model_class = AutoModelForCausalLM
        elif spec.model_type in {"sequenceclassification", "classification", "transformer"}:
            model_class = AutoModelForSequenceClassification
            model_kwargs.setdefault("num_labels", self.model_cfg.get("num_labels", 2))
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

    def _load_tokenizer(self, spec: ModelLoadSpec) -> PreTrainedTokenizerBase:
        tokenizer_kwargs = dict(spec.extra_tokenizer_kwargs)
        tokenizer_kwargs.setdefault("use_fast", True)
        tokenizer_kwargs.setdefault("trust_remote_code", spec.trust_remote_code)
        tokenizer_kwargs.setdefault("local_files_only", spec.local_files_only)
        if spec.hf_token:
            tokenizer_kwargs.setdefault("token", spec.hf_token)

        tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_name, **tokenizer_kwargs)

        # Ensure padding token for decoder-only models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _instantiate_model(
        self,
        model_class: type,
        name: str,
        spec: ModelLoadSpec,
        role: str,
        **model_kwargs: Any,
    ) -> PreTrainedModel:
        logger.info(
            "Loading %s model '%s' (type=%s) on %s",
            role,
            name,
            spec.model_type,
            self.device,
        )
        model = model_class.from_pretrained(name, **model_kwargs)
        model.to(self.device)
        model.eval()
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

    def _run_teacher_agent(
        self,
        data_samples: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        logger.info("No teacher specified, invoking Teacher Agent")

        if data_samples is None:
            data_samples = self._load_agent_samples()

        from core.agents import quick_teacher_setup

        resource_constraint = "low" if str(self.device) == "mps" else "medium"

        result = quick_teacher_setup(
            data_samples=data_samples,
            device=str(self.device),
            resource_constraint=resource_constraint,
        )

        teacher_model = result["model"]
        tokenizer = result["tokenizer"]
        model_name = result["model_name"]

        self.model_cfg["name"] = model_name
        if tokenizer is not None:
            self.model_cfg.setdefault("tokenizer_name", model_name)

        # Keep loaded objects on hand for reuse
        self._agent_teacher = teacher_model
        self._agent_tokenizer = tokenizer
        return model_name, model_name, data_samples

    def _load_agent_samples(self) -> List[Dict[str, Any]]:
        try:
            from data.dataloaders import load_sample_data

            data_path = _cfg_get(self.cfg, "data", {}).get(
                "train_path",
                "data/imdb_train.jsonl",
            )
            samples = load_sample_data(data_path, max_samples=50)
            logger.info("Loaded %d samples for teacher agent", len(samples))
            return samples
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("Could not load data samples for agent: %s", exc)
            return []


def _maybe_compile(model: PreTrainedModel) -> PreTrainedModel:
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available in this PyTorch installation")
        return model
    try:
        compiled = torch.compile(model, dynamic=True)  # type: ignore[arg-type]
        logger.info("Model compiled with torch.compile")
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - depends on version
        logger.warning("torch.compile failed: %s", exc)
        return model


def _maybe_attach_adapter(model: PreTrainedModel, adapter_path: str) -> None:
    if not adapter_path:
        return
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        logger.warning("Adapter path %s does not exist", adapter_path)
        return

    try:
        from peft import PeftModel  # type: ignore[import]
    except ImportError:
        logger.warning("PEFT not installed; skipping adapter load")
        return

    try:
        PeftModel.from_pretrained(model, str(adapter_path_obj))
        logger.info("Loaded PEFT adapter from %s", adapter_path)
    except Exception as exc:
        logger.warning("Failed to load adapter %s: %s", adapter_path, exc)


# ---------------------------------------------------------------------------
# Backwards compatible helpers
# ---------------------------------------------------------------------------


def get_device(device_str: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Backward compatible alias for resolve_device."""
    return resolve_device(device_str)


def load_models(
    cfg: Union["ConfigManager", Dict[str, Any]],
    device: Optional[Union[str, torch.device]] = None,
    use_agent: bool = True,
    data_samples: Optional[List[Dict[str, Any]]] = None,
    return_bundle: bool = False,
):
    """Shim that mirrors the previous public API while using ModelLoader."""

    loader = ModelLoader(cfg=cfg, device=device)
    bundle = cast(
        ModelBundle,
        loader.load(use_agent=use_agent, data_samples=data_samples, return_bundle=True),
    )

    if return_bundle:
        return bundle
    return bundle.teacher, bundle.student, bundle.tokenizer


def model_summary(model: PreTrainedModel) -> Dict[str, Any]:
    """Expose the enhanced summary helper for external callers."""
    return _summarize_model(model)
