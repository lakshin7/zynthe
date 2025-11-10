"""Image dataloaders integrating advanced vision preprocessing.

Usage (config snippet):
model:
  name: vit-base-patch16
  image_size: 224
preprocessing:
  image_mean: [0.485, 0.456, 0.406]
  image_std:  [0.229, 0.224, 0.225]
  augment:
    random_flip: true
    random_crop: false
    color_jitter: false

data:
  train_cifar10: true
  val_cifar10: true

Call create_image_dataloaders(cfg) to obtain (train_loader, val_loader).
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
try:
    from torchvision import datasets
except Exception:
    datasets = None

from core.preprocessing.registry import PreprocessRegistry, ensure_registered
from core.preprocessing.built_ins import register_defaults


class CIFAR10Wrapper(Dataset):
    def __init__(self, root: str, train: bool, transform, config: Dict[str, Any]):
        if datasets is None:
            raise RuntimeError("torchvision not installed; cannot use CIFAR10 dataset")
        self.ds = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.config = config
        ensure_registered(register_defaults)
        # Use adapter for cifar10 (registered) then vit preprocessor via model name detection
        self.adapter = PreprocessRegistry.get_dataset_adapter("cifar10")
        model_name = config.get("model", {}).get("name", "vit")
        self.preprocessor = PreprocessRegistry.get_model_preprocessor(model_name, tokenizer=None, config=config)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        raw = {"image": img, "label": label}
        adapted = self.adapter.adapt(raw) if self.adapter else raw
        features = self.preprocessor.prepare(adapted)
        return features


def create_image_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    batch_size = cfg.get("train", {}).get("batch_size", 32)
    num_workers = cfg.get("train", {}).get("num_workers", 0)
    root = cfg.get("data", {}).get("image_root", "./data")

    # The transforms are already embedded in ViTPreprocessor; external transform is minimal.
    tx = None

    train_ds = CIFAR10Wrapper(root=root, train=True, transform=tx, config=cfg)
    val_ds = CIFAR10Wrapper(root=root, train=False, transform=tx, config=cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

__all__ = ["create_image_dataloaders"]
