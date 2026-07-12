"""Smoke-tests for the public Zynthe import surface.

Run as part of the default CI matrix under ``pip install zynthe`` (no extras).
Each entry in ``zynthe.__all__`` must either:

* import successfully in a fresh ``zynthe`` install (no extras), or
* raise an ``ImportError`` whose message points to a known optional extra.

Anything else (e.g., a missing optional dep that crashes another module,
a typo, a circular import) is treated as a failure.
"""

from __future__ import annotations

import importlib
from typing import Tuple

import pytest

import zynthe


PUBLIC_NAMES: Tuple[str, ...] = tuple(name for name in zynthe.__all__ if name != "__version__")


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_symbol_importable(name: str) -> None:
    """Each public name must resolve without unexpected errors."""
    try:
        value = getattr(zynthe, name)
    except ImportError as exc:  # missing optional dep is OK
        msg = str(exc).lower()
        assert any(
            token in msg
            for token in (
                "optional extra",
                "pip install zynthe[",
                "torch",
                "transformers",
                "accelerate",
            )
        ), f"Unexpected ImportError for {name}: {exc}"
        return
    except AttributeError as exc:
        pytest.fail(f"public name {name!r} missing: {exc}")

    assert value is not None, f"public name {name!r} resolved to None"


def test_version_aligned() -> None:
    """``zynthe.__version__`` must not be empty and be import-string."""
    assert isinstance(zynthe.__version__, str)
    assert zynthe.__version__, "version string is empty"
    parts = zynthe.__version__.split(".")
    assert len(parts) >= 2, f"version {zynthe.__version__!r} is not semver-ish"


def test_subpackage_imports_do_not_crash() -> None:
    """Probe direct imports of every public subpackage path."""
    candidates = [
        "zynthe.core.utils",
        "zynthe.core.utils.exceptions",
        "zynthe.core.distillers",
        "zynthe.core.pipelines",
        "zynthe.core.adapters",
        "zynthe.core.preflight",
        "zynthe.core.quant",
        "zynthe.core.config",
    ]
    for mod in candidates:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            msg = str(exc).lower()
            if "torch" in msg or "transformers" in msg:
                # Expected when torch is not installed locally; CI gates this.
                continue
            pytest.fail(f"unexpected import error for {mod}: {exc}")
