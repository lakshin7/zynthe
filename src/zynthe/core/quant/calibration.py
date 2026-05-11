"""Utilities to collect calibration statistics for static quantization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

import torch
from torch.utils.data import DataLoader

LOG = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration describing how calibration data should be sampled."""

    num_batches: int = 32
    max_samples: Optional[int] = 1024
    use_training_split: bool = False
    shuffle: bool = False


def _default_prepare(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            stacked = torch.stack([v.to(device) for v in value])
            out[key] = stacked
        else:
            out[key] = value
    return out


class CalibrationRunner:
    """Feeds representative batches through a prepared model to collect observers."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: Iterable[Dict[str, Any]] | DataLoader,
        device: torch.device | str,
        config: CalibrationConfig | None = None,
        prepare_batch: Optional[Callable[[Dict[str, Any], torch.device], Dict[str, Any]]] = None,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = torch.device(device)
        self.config = config or CalibrationConfig()
        self.prepare_batch = prepare_batch or _default_prepare

    def _iter_batches(self) -> Iterator[Dict[str, Any]]:
        consumed = 0
        for batch in self.loader:
            yield batch
            consumed += 1
            if self.config.num_batches and consumed >= self.config.num_batches:
                break

    def collect(self) -> int:
        """Run calibration and return number of samples used."""

        self.model.eval()
        processed = 0
        with torch.inference_mode():
            for raw_batch in self._iter_batches():
                batch = self.prepare_batch(raw_batch, self.device)
                try:
                    outputs = self.model(**batch)
                    if isinstance(outputs, dict) and "loss" in outputs:
                        _ = outputs["loss"]
                except TypeError:
                    # Some models expect positional inputs, fall back for common schema.
                    input_args = batch.get("input_ids")
                    attention = batch.get("attention_mask")
                    token_type = batch.get("token_type_ids")
                    if token_type is not None:
                        self.model(input_args, attention, token_type)
                    elif attention is not None:
                        self.model(input_args, attention)
                    else:
                        self.model(input_args)
                batch_size = 0
                for value in batch.values():
                    if torch.is_tensor(value) and value.ndim >= 1:
                        batch_size = int(value.shape[0])
                        break
                processed += batch_size or 1
                if self.config.max_samples and processed >= self.config.max_samples:
                    break
        return processed


def build_calibration_loader(
    cfg: Dict[str, Any],
    tokenizer,
    config: CalibrationConfig | None = None,
) -> DataLoader:
    """Create a small DataLoader suitable for calibration."""

    from zynthe.data.dataloaders import create_dataloaders

    config = config or CalibrationConfig()
    train_loader, val_loader = create_dataloaders(cfg, tokenizer)
    loader = train_loader if config.use_training_split else val_loader
    if (
        not config.shuffle
        and hasattr(loader, "sampler")
        and getattr(loader.sampler, "shuffle", False)
    ):
        LOG.info("Calibration loader uses validation split to avoid shuffling representative data.")
    return loader
