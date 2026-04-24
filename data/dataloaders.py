"""Utilities for building PyTorch dataloaders backed by JSONL datasets."""

from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from core.preprocessing.advanced import run_advanced_pipeline
from core.preprocessing.built_ins import register_defaults
from core.preprocessing.registry import PreprocessRegistry, ensure_registered
from .augmentations import TextAugmenter, build_text_augmenter
from .preprocess import PreprocessConfig, apply_preprocess_pipeline, build_preprocess_config

LOG = logging.getLogger(__name__)
DEFAULT_CACHE_SIZE = 2048


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader workers deterministically from PyTorch worker seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _safe_label(value: Any) -> int:
    """Convert label-like values into integers safely."""

    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "+1", "true", "+", "pos", "positive"}:
                return 1
            if v in {"0", "-1", "false", "-", "neg", "negative"}:
                return 0
            return int(v)
    except Exception:  # pragma: no cover - defensive conversion
        pass
    return 0


def load_sample_data(file_path: str, max_samples: int = 100) -> List[Dict[str, Any]]:
    """Load up to ``max_samples`` JSONL records for quick inspection."""

    samples: List[Dict[str, Any]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx >= max_samples:
                    break
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        LOG.warning("Sample data path not found: %s", file_path)
    except Exception as exc:
        LOG.warning("Could not load sample data from %s: %s", file_path, exc)
    return samples


class JsonlDataset(Dataset):
    """Dataset with optional preprocessing, augmentation, and token caching."""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        *,
    max_length: int = 128,
    model_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
        dataset_id: Optional[str] = None,
        split: str = "train",
        augmenter: Optional[TextAugmenter] = None,
    ) -> None:
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name or "generic"
        self.config: Dict[str, Any] = dict(config) if config else {"model": {"max_length": max_length}}
        self.dataset_id = dataset_id or "unknown"
        self.split = split
        self.augmenter = augmenter if split == "train" else None

        self.samples: List[Dict[str, Any]] = self._read_jsonl(file_path)

        # Basic preprocessing (HTML stripping, whitespace normalization, etc.)
        self.preprocess_config: PreprocessConfig = build_preprocess_config(self.config, split=split)
        if self._advanced_enabled:
            # Advanced pipeline will apply token-level truncation.
            self.preprocess_config.max_characters = None
        self.samples, self.preprocess_stats = apply_preprocess_pipeline(self.samples, self.preprocess_config)
        LOG.debug(
            "[%s] basic preprocessing summary: %s",
            split,
            self.preprocess_stats.to_dict(),
        )

        # Advanced preprocessing (templating, dedup, tokenizer-aware truncation)
        self.samples = self._run_advanced_pipeline(self.samples)

        ensure_registered(register_defaults)
        try:
            self.adapter = PreprocessRegistry.get_dataset_adapter(self.dataset_id)
        except Exception:
            self.adapter = None
        self.preprocessor = PreprocessRegistry.get_model_preprocessor(self.model_name, self.tokenizer, self.config)

        cache_cfg = self._resolve_cache_config()
        self._cache_size = cache_cfg
        self._token_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def _advanced_enabled(self) -> bool:
        prep_cfg = self.config.get("preprocessing", {})
        if isinstance(prep_cfg, Mapping):
            adv_cfg = prep_cfg.get("advanced", {})
            if isinstance(adv_cfg, Mapping):
                return bool(adv_cfg.get("enable", False))
        return False

    def _resolve_cache_config(self) -> int:
        cache_cfg: dict = {}
        prep_cfg = self.config.get("preprocessing", {})
        if isinstance(prep_cfg, Mapping):
            cache_cfg = prep_cfg.get("cache") or {}
        size = cache_cfg.get("token_cache_size", cache_cfg.get("size", DEFAULT_CACHE_SIZE))
        try:
            size_int = int(size)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            size_int = DEFAULT_CACHE_SIZE
        return max(0, size_int)

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data.get("text"), str):
                    data["text"] = " ".join(data["text"].split())
                records.append(data)
        return records

    def _run_advanced_pipeline(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._advanced_enabled:
            return samples
        role = self.split
        before = len(samples)
        try:
            processed = run_advanced_pipeline(samples, self.tokenizer, self.config, role=role)
        except Exception as exc:
            LOG.warning("Advanced preprocessing skipped (%s): %s", self.file_path, exc)
            return samples

        if processed:
            LOG.debug("[%s] advanced preprocessing: %d -> %d", role, before, len(processed))
            return processed

        LOG.warning(
            "Advanced preprocessing removed all samples for %s; reverting to basic-preprocessed data",
            self.file_path,
        )
        # Re-read raw file and re-apply basic preprocessing as a fallback.
        fallback = self._read_jsonl(self.file_path)
        fallback, stats = apply_preprocess_pipeline(fallback, self.preprocess_config)
        LOG.debug("[%s] fallback preprocessing summary: %s", role, stats.to_dict())
        return fallback

    @staticmethod
    def _clone_item(item: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value.clone() if torch.is_tensor(value) else value for key, value in item.items()}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_sample = dict(self.samples[index])
        augmented_flag = False
        if self.augmenter:
            augmented_sample = self.augmenter(base_sample)
            augmented_flag = bool(augmented_sample.pop("__augmented__", False))
            augmented_sample.pop("augmentation_ops", None)
            base_sample = augmented_sample

        if self.adapter is not None:
            normalized = self.adapter.adapt(base_sample)
        else:
            normalized = {
                "text": base_sample.get("text", ""),
                "label": _safe_label(base_sample.get("label", 0)),
            }

        if normalized.get("text") is None:
            normalized["text"] = ""

        cache_key: Optional[str] = None
        if not augmented_flag and isinstance(normalized.get("text"), str):
            cache_key = normalized["text"]

        if cache_key and self._cache_size > 0 and cache_key in self._token_cache:
            cached = self._token_cache[cache_key]
            self._token_cache.move_to_end(cache_key)
            return self._clone_item(cached)

        item = self.preprocessor.prepare(normalized)

        if cache_key and self._cache_size > 0:
            self._token_cache[cache_key] = self._clone_item(item)
            if len(self._token_cache) > self._cache_size:
                self._token_cache.popitem(last=False)

        return item

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def label_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for sample in self.samples:
            label = _safe_label(sample.get("label", 0))
            counts[label] = counts.get(label, 0) + 1
        return counts

    def cache_info(self) -> Dict[str, int]:
        return {"size": len(self._token_cache), "capacity": self._cache_size}


def _default_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}
    output: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [record[key] for record in batch]
        first = values[0]
        if torch.is_tensor(first):
            try:
                output[key] = torch.stack(values)
                continue
            except Exception:
                pass
        output[key] = values
    return output


def _build_sampler(dataset: JsonlDataset, balance: bool) -> Optional[WeightedRandomSampler]:
    if not balance:
        return None
    counts = dataset.label_counts()
    if not counts or len(counts) == 1:
        return None
    total = sum(counts.values())
    class_count = len(counts)
    weights: List[float] = []
    for sample in dataset.samples:
        label = _safe_label(sample.get("label", 0))
        count = counts.get(label, 1)
        weight = float(total) / (class_count * count)
        weights.append(weight)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def create_dataloaders(cfg: Mapping[str, Any], tokenizer) -> Tuple[DataLoader, DataLoader]:
    """Instantiate train/validation dataloaders from a configuration mapping."""

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    # Route vision/image configurations through the universal image dataloader.
    modality = str(data_cfg.get("modality", model_cfg.get("type", ""))).lower()
    image_dataset = str(data_cfg.get("image_dataset", "")).strip()
    if image_dataset or modality == "vision" or str(data_cfg.get("type", "")).lower() == "image":
        from .image_dataloaders import create_image_dataloaders

        return create_image_dataloaders(cfg, tokenizer=tokenizer)

    train_path = data_cfg.get("train_path")
    val_path = data_cfg.get("val_path")
    if not train_path or not val_path:
        raise ValueError("Configuration must define data.train_path and data.val_path")

    max_length = int(model_cfg.get("max_length", 128))
    model_name = (
        model_cfg.get("student_name")
        or model_cfg.get("name")
        or getattr(tokenizer, "name_or_path", "generic")
    )
    dataset_id = data_cfg.get("dataset_id", "unknown")

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False))
    balance_train = bool(data_cfg.get("balance_train_classes", False))
    seed = int(
        train_cfg.get(
            "seed",
            cfg.get("seed", cfg.get("runtime", {}).get("seed", 42)),
        )
    )

    augmenter = build_text_augmenter(cfg, split="train")

    train_dataset = JsonlDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        model_name=model_name,
        config=cfg,
        dataset_id=dataset_id,
        split="train",
        augmenter=augmenter,
    )
    val_dataset = JsonlDataset(
        val_path,
        tokenizer,
        max_length=max_length,
        model_name=model_name,
        config=cfg,
        dataset_id=dataset_id,
        split="val",
        augmenter=None,
    )

    sampler = _build_sampler(train_dataset, balance_train)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_default_collate,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_default_collate,
        worker_init_fn=_seed_worker,
    )

    return train_loader, val_loader


def get_imdb_dataloaders(
    *,
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    dataset_id: str = "imdb",
    preprocessing: Optional[Mapping[str, Any]] = None,
    balance: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience wrapper for the bundled IMDB dataset."""

    base_cfg: Dict[str, Any] = {
        "data": {
            "train_path": train_path,
            "val_path": val_path,
            "dataset_id": dataset_id,
            "balance_train_classes": balance,
        },
        "model": {
            "max_length": max_length,
            "name": getattr(tokenizer, "name_or_path", "generic"),
        },
        "train": {
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    }
    if preprocessing:
        base_cfg["preprocessing"] = preprocessing
    return create_dataloaders(base_cfg, tokenizer)


__all__ = [
    "JsonlDataset",
    "create_dataloaders",
    "get_imdb_dataloaders",
    "load_sample_data",
]
