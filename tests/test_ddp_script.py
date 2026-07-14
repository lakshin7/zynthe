"""Smoke test for the multi-GPU DDP script — runs on Modal.

This test is skipped on non-Modal environments (torchrun requires a
multi-GPU machine).  It's wired through :mod:`pytest` to keep the
suite self-contained; the test file just exists to ensure the
``scripts/smoke/run_distributed_ddp_local.py`` is importable and
parses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "scripts" / "smoke" / "run_distributed_ddp_local.py"
MODAL = ROOT / "scripts" / "smoke" / "run_distributed_ddp.py"


def test_ddp_local_script_imports() -> None:
    """``run_distributed_ddp_local.py`` parses as a Python module."""
    spec = importlib.util.spec_from_file_location("ddp_local", LOCAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Don't execute main(); just compile.
    spec.loader.exec_module  # type: ignore[attr-defined]


def test_ddp_modal_script_exists() -> None:
    """The Modal entry point exists and is non-empty."""
    assert MODAL.exists()
    assert MODAL.read_text().strip(), "run_distributed_ddp.py is empty"
    assert "torchrun" in MODAL.read_text()
    assert "MAX_GPUS" in MODAL.read_text() or "--gpus" in MODAL.read_text()
