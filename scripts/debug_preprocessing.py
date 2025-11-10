#!/usr/bin/env python3
from pathlib import Path
import json
import sys
from typing import List, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from core.preprocessing.registry import PreprocessRegistry
from core.preprocessing.built_ins import register_defaults
from core.utils.data_validator import DataValidator


def peek_jsonl(path: Path, n: int = 3) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def main():
    data_dir = Path("data/sst2_test")
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"
    if not train_path.exists():
        print(f"No dataset found at {train_path}")
        sys.exit(1)

    # Show a few raw rows
    print("RAW SAMPLES:")
    for row in peek_jsonl(train_path, 2):
        print(row)

    # Build tokenizer and preprocessors
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    register_defaults()
    adapter = PreprocessRegistry.get_dataset_adapter("glue/sst2")
    preproc = PreprocessRegistry.get_model_preprocessor(model_name, tokenizer, {
        "model": {"max_length": 128},
        "preprocessing": {}
    })

    # Adapt + prepare a couple of samples
    print("\nADAPTED + TOKENIZED SAMPLES:")
    raw_rows = peek_jsonl(train_path, 2)
    for raw in raw_rows:
        norm = adapter.adapt(raw) if adapter else raw
        item = preproc.prepare(norm)
        keys = {k: (tuple(item[k].size()) if hasattr(item[k], 'size') else type(item[k])) for k in item}
        print(keys)

    # Quick batch validation
    from data.dataloaders import create_dataloaders
    cfg = {
        "model": {"max_length": 128, "student_name": model_name},
        "data": {"train_path": str(train_path), "val_path": str(val_path), "dataset_id": "glue/sst2"},
        "train": {"batch_size": 8}
    }
    train_loader, val_loader = create_dataloaders(cfg, tokenizer)
    DataValidator.assert_preprocessing_ok(train_loader.dataset, val_loader.dataset)
    batch = next(iter(train_loader))
    print("\nONE BATCH KEYS:", list(batch.keys()))
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["labels"].shape)


if __name__ == "__main__":
    main()
