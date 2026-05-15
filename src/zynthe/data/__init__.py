"""Zynthé data loading and preprocessing utilities.

Provides JSONL/image dataset loaders, text augmentation, and
preprocessing pipelines for distillation training.

Quick Start::

    from zynthe.data import create_dataloaders, JsonlDataset

    train_loader, val_loader = create_dataloaders(config, tokenizer)
"""

from __future__ import annotations

from .augmentations import TextAugmenter, build_text_augmenter
from .dataloaders import JsonlDataset, create_dataloaders, load_sample_data
from .preprocess import PreprocessConfig, apply_preprocess_pipeline, build_preprocess_config

__all__ = [
    "JsonlDataset",
    "create_dataloaders",
    "load_sample_data",
    "TextAugmenter",
    "build_text_augmenter",
    "PreprocessConfig",
    "apply_preprocess_pipeline",
    "build_preprocess_config",
]
