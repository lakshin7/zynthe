"""Preprocessing registry and base abstractions.

Provides a unified interface for dataset-specific adapters and model-family
specific preprocessors so new datasets/models can plug in without touching
trainer or dataloader logic.
"""

from __future__ import annotations
from typing import Dict, Callable, Any, Optional, List
import logging

LOG = logging.getLogger(__name__)


class SampleAdapter:
    """Dataset-specific normalization.

    Converts a raw dataset item (HF sample or JSONL dict) into a canonical
    intermediate representation: {"text": str, "label": int, ...extras }
    """

    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class ModelPreprocessor:
    """Model family specific feature creation.

    Takes a normalized sample and produces tensor features expected by the model.
    Returns a dict containing keys consumed by forward(): e.g. input_ids, attention_mask, labels.
    """

    def __init__(self, tokenizer, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config

    def prepare(self, sample: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class PreprocessRegistry:
    dataset_adapters: Dict[str, SampleAdapter] = {}
    dataset_metadata: Dict[str, Dict[str, Any]] = {}
    model_preprocessors: Dict[str, Callable[[Any, Dict[str, Any]], ModelPreprocessor]] = {}
    model_metadata: Dict[str, Dict[str, Any]] = {}

    # Keyword heuristics to auto-detect dataset adapters when exact key missing
    DATASET_KEYWORDS: List[tuple[str, List[str]]] = [
        ("imdb", ["imdb", "movie_reviews"]),
        ("sst2", ["sst-2", "sst2", "stanford_sentiment"]),
        ("sentiment140", ["sentiment140", "twitter_sentiment"]),
        ("mnli", ["mnli", "multi_nli", "multi_natural"]),
        ("qqp", ["qqp", "quora"]),
        ("qnli", ["qnli"]),
        ("cola", ["cola", "grammatical"]),
        ("ag_news", ["agnews", "ag_news"]),
        ("cifar10", ["cifar10", "cifar-10"]),
    ]

    MODEL_FAMILY_KEYWORDS: List[tuple[str, List[str]]] = [
        (
            "bert",
            [
                "bert",
                "roberta",
                "albert",
                "distilbert",
                "electra",
                "deberta",
                "minilm",
                "xlm-roberta",
                "ernie",
            ],
        ),
        ("t5", ["t5", "mt5", "flant5", "flan-t5"]),
        (
            "gpt2",
            [
                "gpt",
                "gpt2",
                "gpt-neo",
                "gpt-j",
                "gptj",
                "opt",
                "llama",
                "mistral",
                "falcon",
                "qwen",
                "mpt",
            ],
        ),
        ("vit", ["vit", "vision_transformer", "beit", "swin", "deit", "convnext"]),
    ]

    GENERIC_DATASET_KEY = "generic_text"

    @classmethod
    def register_dataset(
        cls, key: str, adapter: SampleAdapter, metadata: Optional[Dict[str, Any]] = None
    ):
        LOG.debug(f"Registering dataset adapter: {key}")
        cls.dataset_adapters[key] = adapter
        if metadata is not None:
            cls.dataset_metadata[key] = metadata
        elif key not in cls.dataset_metadata:
            cls.dataset_metadata[key] = {}

    @classmethod
    def register_model(
        cls,
        family: str,
        factory: Callable[[Any, Dict[str, Any]], ModelPreprocessor],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        LOG.debug(f"Registering model preprocessor: {family}")
        cls.model_preprocessors[family] = factory
        if metadata is not None:
            cls.model_metadata[family] = metadata
        elif family not in cls.model_metadata:
            cls.model_metadata[family] = {}

    @classmethod
    def get_dataset_adapter(cls, dataset_id: str) -> Optional[SampleAdapter]:
        resolved = cls.resolve_dataset_adapter(dataset_id)
        return resolved.get("adapter") if resolved else None

    @classmethod
    def resolve_dataset_adapter(cls, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return adapter with resolution metadata for diagnostics."""
        if not dataset_id:
            return None

        if dataset_id in cls.dataset_adapters:
            return {
                "adapter": cls.dataset_adapters[dataset_id],
                "resolved_key": dataset_id,
                "confidence": "high",
                "metadata": cls.dataset_metadata.get(dataset_id, {}),
                "reason": "Exact registry match",
            }

        parts = dataset_id.split("/")
        for part in parts:
            if part in cls.dataset_adapters:
                return {
                    "adapter": cls.dataset_adapters[part],
                    "resolved_key": part,
                    "confidence": "medium",
                    "metadata": cls.dataset_metadata.get(part, {}),
                    "reason": f"Matched dataset segment '{part}'",
                }

        for key in cls.dataset_adapters:
            if dataset_id.startswith(key):
                return {
                    "adapter": cls.dataset_adapters[key],
                    "resolved_key": key,
                    "confidence": "medium",
                    "metadata": cls.dataset_metadata.get(key, {}),
                    "reason": f"Dataset id starts with '{key}'",
                }

        lname = dataset_id.lower()
        for target, keywords in cls.DATASET_KEYWORDS:
            if any(keyword in lname for keyword in keywords) and target in cls.dataset_adapters:
                return {
                    "adapter": cls.dataset_adapters[target],
                    "resolved_key": target,
                    "confidence": "low",
                    "metadata": cls.dataset_metadata.get(target, {}),
                    "reason": f"Keyword heuristic matched '{target}'",
                }

        generic = cls.dataset_adapters.get(cls.GENERIC_DATASET_KEY)
        if generic:
            return {
                "adapter": generic,
                "resolved_key": cls.GENERIC_DATASET_KEY,
                "confidence": "low",
                "metadata": cls.dataset_metadata.get(cls.GENERIC_DATASET_KEY, {}),
                "reason": "Falling back to generic text adapter",
            }

        return None

    @classmethod
    def detect_model_family(cls, model_name: str) -> str:
        lname = model_name.lower()

        for family, keywords in cls.MODEL_FAMILY_KEYWORDS:
            if any(keyword in lname for keyword in keywords):
                return family

        return "generic"

    @classmethod
    def get_model_preprocessor(
        cls, model_name: str, tokenizer, config: Dict[str, Any]
    ) -> ModelPreprocessor:
        resolved = cls.resolve_model_preprocessor(model_name)
        if not resolved:
            raise RuntimeError(
                f"No preprocessor registered for model '{model_name}'. Register a generic handler first."
            )
        factory = resolved["factory"]
        return factory(tokenizer, config)

    @classmethod
    def resolve_model_preprocessor(cls, model_name: str) -> Optional[Dict[str, Any]]:
        family = cls.detect_model_family(model_name)
        factory = cls.model_preprocessors.get(family)
        if factory is not None:
            return {
                "family": family,
                "factory": factory,
                "confidence": "high",
                "metadata": cls.model_metadata.get(family, {}),
            }

        fallback = cls.model_preprocessors.get("generic")
        if fallback is not None:
            return {
                "family": "generic",
                "factory": fallback,
                "confidence": "low",
                "metadata": cls.model_metadata.get("generic", {}),
            }
        return None

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        """Return registry contents for inspection and UI display."""
        datasets = []
        for key, adapter in cls.dataset_adapters.items():
            data = {
                "key": key,
                "adapter": adapter.__class__.__name__,
                "metadata": cls.dataset_metadata.get(key, {}),
            }
            datasets.append(data)

        models = []
        for family, factory in cls.model_preprocessors.items():
            info = {
                "family": family,
                "factory": getattr(factory, "__name__", str(factory)),
                "metadata": cls.model_metadata.get(family, {}),
            }
            models.append(info)

        return {
            "datasets": datasets,
            "model_preprocessors": models,
        }


def ensure_registered(defaults_loader: Callable[[], None]):
    """Utility to call a defaults loader once (idempotent)."""
    if not PreprocessRegistry.dataset_adapters or not PreprocessRegistry.model_preprocessors:
        defaults_loader()
