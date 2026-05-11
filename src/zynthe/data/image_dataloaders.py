"""Universal image dataloaders for vision and multimodal foundations.

Supported `data.image_dataset` values:
- `cifar10`
- `cifar100`
- `stl10`
- `imagenet`
- `image_folder`
- `hf:<dataset_id>`
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torchvision import datasets as tv_datasets
except Exception:  # pragma: no cover - optional dependency
    tv_datasets = None

try:
    from datasets import load_dataset as hf_load_dataset
except Exception:  # pragma: no cover - optional dependency
    hf_load_dataset = None

from zynthe.core.preprocessing.built_ins import register_defaults
from zynthe.core.preprocessing.registry import PreprocessRegistry, ensure_registered

LOG = logging.getLogger(__name__)


class _VisionDatasetWrapper(Dataset):
    """Normalize image datasets into model-ready feature dictionaries."""

    def __init__(self, base_dataset: Dataset, config: Mapping[str, Any], dataset_id: str):
        self.base_dataset = base_dataset
        self.config = dict(config)
        self.dataset_id = dataset_id

        ensure_registered(register_defaults)
        resolved = PreprocessRegistry.resolve_dataset_adapter(dataset_id)
        metadata = resolved.get("metadata", {}) if resolved else {}
        self.adapter = (
            resolved.get("adapter") if resolved and metadata.get("modality") == "vision" else None
        )

        model_name = self.config.get("model", {}).get("name", "vit")
        self.preprocessor = PreprocessRegistry.get_model_preprocessor(
            model_name, tokenizer=None, config=self.config
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _extract_image_label(self, raw_sample: Any) -> Tuple[Any, int]:
        if isinstance(raw_sample, dict):
            image = raw_sample.get("image", raw_sample.get("img", raw_sample.get("pixel_values")))
            label = raw_sample.get("label", raw_sample.get("labels", 0))
        elif isinstance(raw_sample, (tuple, list)) and len(raw_sample) >= 2:
            image, label = raw_sample[0], raw_sample[1]
        else:
            raise ValueError("Unsupported vision sample format; expected dict or (image, label)")

        if hasattr(label, "item"):
            label = int(label.item())  # type: ignore[union-attr]
        else:
            label = int(label)  # type: ignore[arg-type]
        return image, label

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw_sample = self.base_dataset[index]
        image, label = self._extract_image_label(raw_sample)
        normalized = {"image": image, "label": label}

        if self.adapter is not None:
            try:
                normalized = self.adapter.adapt(normalized)
            except Exception:
                LOG.debug(
                    "Vision adapter failed for dataset=%s; using default normalization",
                    self.dataset_id,
                    exc_info=True,
                )

        return self.preprocessor.prepare(normalized)


def _require_torchvision() -> Any:
    if tv_datasets is None:
        raise RuntimeError(
            "torchvision is required for image dataloaders. Install torchvision to use vision datasets."
        )
    return tv_datasets


def _split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    total = len(dataset)
    if total < 2:
        raise ValueError("Image dataset requires at least 2 samples to create train/val split")

    val_size = max(1, int(total * val_ratio))
    train_size = max(1, total - val_size)
    if train_size + val_size != total:
        val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    return train_ds, val_ds


def _resolve_image_dataset(data_cfg: Mapping[str, Any]) -> str:
    explicit = str(data_cfg.get("image_dataset", "")).strip().lower()
    if explicit:
        return explicit

    dataset_id = str(data_cfg.get("dataset_id", "")).strip().lower()
    if dataset_id.startswith("hf:"):
        return dataset_id
    if dataset_id in {"cifar10", "cifar100", "stl10", "imagenet", "image_folder"}:
        return dataset_id

    if str(data_cfg.get("type", "")).strip().lower() == "image":
        return dataset_id or "cifar10"

    return "cifar10"


def _build_torchvision_dataset(dataset_name: str, root: str, split: str) -> Dataset:
    ds_lib = _require_torchvision()
    is_train = split == "train"

    if dataset_name == "cifar10":
        return ds_lib.CIFAR10(root=root, train=is_train, download=True)
    if dataset_name == "cifar100":
        return ds_lib.CIFAR100(root=root, train=is_train, download=True)
    if dataset_name == "stl10":
        stl_split = "train" if is_train else "test"
        return ds_lib.STL10(root=root, split=stl_split, download=True)

    raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")


def _build_image_folder_datasets(root: str, seed: int, val_ratio: float) -> Tuple[Dataset, Dataset]:
    ds_lib = _require_torchvision()
    root_path = Path(root)
    train_dir = root_path / "train"
    val_dir = root_path / "val"

    if train_dir.exists() and val_dir.exists():
        return ds_lib.ImageFolder(str(train_dir)), ds_lib.ImageFolder(str(val_dir))

    combined = ds_lib.ImageFolder(str(root_path))
    return _split_dataset(combined, val_ratio=val_ratio, seed=seed)


def _build_hf_datasets(dataset_id: str, seed: int, val_ratio: float) -> Tuple[Dataset, Dataset]:
    if hf_load_dataset is None:
        raise RuntimeError("datasets package is required for hf:<dataset_id> image datasets")

    train_split_candidates = ["train", "training"]
    val_split_candidates = ["validation", "valid", "val", "test"]

    train_ds = None
    val_ds = None
    for split_name in train_split_candidates:
        try:
            train_ds = hf_load_dataset(dataset_id, split=split_name)
            break
        except Exception:
            continue
    for split_name in val_split_candidates:
        try:
            val_ds = hf_load_dataset(dataset_id, split=split_name)
            break
        except Exception:
            continue

    if train_ds is None:
        # Fallback for datasets that only expose one split.
        full_ds = hf_load_dataset(dataset_id, split="train")
        train_ds, val_ds = _split_dataset(full_ds, val_ratio=val_ratio, seed=seed)
        return train_ds, val_ds

    if val_ds is None:
        train_ds, val_ds = _split_dataset(train_ds, val_ratio=val_ratio, seed=seed)
        return train_ds, val_ds

    return train_ds, val_ds


def create_image_dataloaders(
    cfg: Mapping[str, Any], tokenizer: Optional[Any] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for image-centric datasets."""
    del tokenizer  # Kept for API compatibility with text dataloader signatures.

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    image_dataset = _resolve_image_dataset(data_cfg)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False))
    seed = int(train_cfg.get("seed", cfg.get("seed", cfg.get("runtime", {}).get("seed", 42))))
    val_ratio = float(data_cfg.get("val_split", data_cfg.get("validation_split", 0.2)))
    val_ratio = min(max(val_ratio, 0.05), 0.5)
    root = str(data_cfg.get("image_root", "./data"))

    if image_dataset.startswith("hf:"):
        hf_id = image_dataset.split(":", 1)[1]
        train_base, val_base = _build_hf_datasets(hf_id, seed=seed, val_ratio=val_ratio)
        dataset_hint = data_cfg.get("dataset_id") or "image_folder"
    elif image_dataset in {"cifar10", "cifar100", "stl10"}:
        train_base = _build_torchvision_dataset(image_dataset, root=root, split="train")
        val_base = _build_torchvision_dataset(image_dataset, root=root, split="val")
        dataset_hint = image_dataset
    elif image_dataset in {"imagenet", "image_folder"}:
        train_base, val_base = _build_image_folder_datasets(
            root=root, seed=seed, val_ratio=val_ratio
        )
        dataset_hint = "imagenet" if image_dataset == "imagenet" else "image_folder"
    else:
        raise ValueError(
            f"Unsupported image dataset '{image_dataset}'. "
            "Use one of: cifar10,cifar100,stl10,imagenet,image_folder,hf:<dataset_id>"
        )

    train_dataset = _VisionDatasetWrapper(train_base, config=cfg, dataset_id=str(dataset_hint))
    val_dataset = _VisionDatasetWrapper(val_base, config=cfg, dataset_id=str(dataset_hint))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


__all__ = ["create_image_dataloaders"]
