import json
import os
from types import SimpleNamespace
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DummyTokenizer:
    """Lightweight tokenizer stub for tests."""

    name_or_path = "zyn-dummy"

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return " ".join(str(int(token)) for token in ids)

    def batch_decode(self, batch, skip_special_tokens: bool = True):
        if isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().numpy()
        return [" ".join(str(int(token)) for token in seq) for seq in batch]

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as handle:
            json.dump({"dummy": True}, handle)


class TinyModel(nn.Module):
    """Compact encoder-style model that produces logits for classification."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 32, num_labels: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, num_labels)
        self.config = SimpleNamespace(_name_or_path="tiny-model", num_labels=num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embedded = embedded * mask
        pooled = embedded.mean(dim=1)
        logits = self.proj(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))


class RandomClassificationDataset(Dataset):
    """Deterministic toy dataset that yields token ids, masks, and labels."""

    def __init__(self, length: int, *, seq_len: int = 8, num_classes: int = 2, offset: int = 0):
        self.length = length
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.offset = offset

    def __len__(self) -> int:  # type: ignore[override]
        return self.length

    def __getitem__(self, index: int):  # type: ignore[override]
        generator = torch.Generator()
        generator.manual_seed(index + self.offset)
        input_ids = torch.randint(1, 50, (self.seq_len,), dtype=torch.long, generator=generator)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        label = torch.tensor((index + self.offset) % self.num_classes, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


@pytest.fixture()
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture()
def dummy_models() -> Tuple[TinyModel, TinyModel]:
    torch.manual_seed(0)
    teacher = TinyModel()
    student = TinyModel()
    return teacher, student


@pytest.fixture()
def sample_loaders() -> Tuple[DataLoader, DataLoader]:
    train_dataset = RandomClassificationDataset(length=12, seq_len=8, offset=0)
    val_dataset = RandomClassificationDataset(length=8, seq_len=8, offset=1)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    return train_loader, val_loader


@pytest.fixture()
def tiny_models() -> Tuple[nn.Module, nn.Module]:
    """Small teacher/student fixtures with optional HF-backed models."""

    try:
        from transformers import (
            BertConfig,
            BertForSequenceClassification,
            DistilBertConfig,
            DistilBertForSequenceClassification,
        )

        teacher = BertForSequenceClassification(
            BertConfig(
                vocab_size=97,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                num_labels=2,
            )
        )
        student = DistilBertForSequenceClassification(
            DistilBertConfig(
                vocab_size=97,
                dim=32,
                hidden_dim=64,
                n_layers=2,
                n_heads=4,
                num_labels=2,
            )
        )
        return teacher, student
    except Exception:
        # Fallback keeps tests runnable when transformers is unavailable.
        return TinyModel(vocab_size=97, hidden_size=32, num_labels=2), TinyModel(
            vocab_size=97,
            hidden_size=24,
            num_labels=2,
        )


@pytest.fixture()
def mock_dataloader() -> DataLoader:
    """Synthetic dataloader fixture for quick CPU-safe tests."""

    dataset = RandomClassificationDataset(length=10, seq_len=8, offset=42)
    return DataLoader(dataset, batch_size=2, shuffle=False)
