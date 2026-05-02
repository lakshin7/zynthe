#!/usr/bin/env python3
"""CPU dry-run of the full training loop using tiny in-memory models.

Purpose:
- Validate loop wiring on low-resource laptops (e.g., Latitude 7490)
- No HuggingFace downloads
- No large model memory usage
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = type("Cfg", (), {"vocab_size": vocab_size, "model_type": "tiny_causal_lm"})()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        del attention_mask, labels, kwargs
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return {"logits": logits}


class TinyTextDataset(Dataset):
    def __init__(self, size: int = 64, seq_len: int = 64, vocab_size: int = 128, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab_size, (size, seq_len), generator=g, dtype=torch.long)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([row[k] for row in batch], dim=0) for k in keys}


def run_causal_lm_core(args) -> None:
    from core.distillers.causal_lm import SafeCausalLMTrainer

    device = torch.device("cpu")
    teacher = TinyCausalLM(vocab_size=args.vocab_size, hidden_size=args.hidden_size)
    student = TinyCausalLM(vocab_size=args.vocab_size, hidden_size=args.hidden_size)

    train_ds = TinyTextDataset(
        size=args.train_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=42,
    )
    val_ds = TinyTextDataset(
        size=args.val_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=777,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    config = {
        "seed": 42,
        "train": {
            "engine": "causal_lm_core",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": 3e-4,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "use_amp": False,
            "log_interval": 5,
            "fail_policy": "skip_step_with_backoff",
            "num_workers": 0,
            "pin_memory": False,
        },
        "distillation": {
            "task_type": "causal_lm",
            "temperature": 2.0,
            "alpha": 0.7,
            "use_ce": True,
            "ignore_index": -100,
            "shift_labels": True,
            "logit_clip": 80.0,
        },
        "checkpoint": {
            "save_every_epoch": True,
            "load_strict_first": True,
            "allow_shape_mismatch_fallback": True,
        },
        "data": {
            "train_path": "__synthetic__",
            "val_path": "__synthetic__",
        },
    }

    exp_dir = Path("experiments") / "dryrun_cpu"
    exp_dir.mkdir(parents=True, exist_ok=True)

    trainer = SafeCausalLMTrainer(
        teacher=teacher,
        student=student,
        tokenizer=None,
        config=config,
        device=device,
        experiment_dir=str(exp_dir),
    )

    result = trainer.fit(train_loader, val_loader)
    print("✅ Dry-run complete")
    print(result)
    print(f"Artifacts: {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run tiny CPU dry-run of full training loop")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=32)
    args = parser.parse_args()

    run_causal_lm_core(args)


if __name__ == "__main__":
    main()
