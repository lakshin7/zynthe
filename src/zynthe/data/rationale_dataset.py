"""Dataset adapter for rationale distillation.

Loads JSONL records with the schema::

    {"input": str, "label": str, "rationale": str}

Each record is the output of an offline LLM rationale-extraction
pipeline (e.g. Google's `distilling-step-by-step` repo).  The dataset
is *string-level* — the trainer / distiller is responsible for
tokenization and prefix-prepending.  This keeps the dataset
lightweight and decoupled from any specific tokenizer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("input", "label", "rationale")


class RationaleDataset(Dataset):
    """JSONL-backed dataset of ``(input, label, rationale)`` triples.

    Args:
        file_path: Path to a JSONL file.  Each line is a JSON object
            with keys ``input``, ``label``, ``rationale``.
        required: Skip records missing any of these keys instead of
            raising.  Useful for partial / heterogeneous dumps.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        required: bool = True,
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"RationaleDataset: {file_path} not found")
        self.records: List[Dict[str, Any]] = list(self._iter_records(required=required))
        if not self.records:
            raise ValueError(
                f"RationaleDataset: no valid records loaded from {file_path}"
            )

    def _iter_records(self, *, required: bool) -> Iterator[Dict[str, Any]]:
        with self.file_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    if required:
                        raise ValueError(
                            f"RationaleDataset: bad JSON at line {lineno}: {exc}"
                        ) from exc
                    logger.warning(
                        "skipping bad JSON at line %d of %s: %s",
                        lineno,
                        self.file_path,
                        exc,
                    )
                    continue
                if not all(k in rec for k in _REQUIRED_KEYS):
                    if required:
                        raise ValueError(
                            f"RationaleDataset: line {lineno} missing one of "
                            f"{_REQUIRED_KEYS} (got keys {sorted(rec.keys())})"
                        )
                    logger.warning(
                        "skipping line %d of %s (missing required keys)",
                        lineno,
                        self.file_path,
                    )
                    continue
                yield rec

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return dict(self.records[idx])

    def __repr__(self) -> str:
        return f"RationaleDataset(file={self.file_path!r}, n={len(self)})"
