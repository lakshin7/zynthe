"""Phase 5 Iteration 3 — verify the MkDocs site structure.

MkDocs itself isn't available in CI; instead, this test parses
``mkdocs.yml`` and asserts every markdown file referenced in the
``nav`` exists on disk.  That catches the common failure mode where a
doc is renamed / moved but the site nav still points at the old path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS = REPO_ROOT / "mkdocs.yml"


@pytest.fixture(scope="module")
def mkdocs_nav() -> list:
    return yaml.safe_load(MKDOCS.read_text())["nav"]


def _flatten_nav(nav: list) -> list:
    """MkDocs nav is a nested list of dicts and strings; flatten to a
    list of file paths.
    """
    paths: list = []
    for item in nav:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(_flatten_nav(value))
        elif isinstance(item, list):
            paths.extend(_flatten_nav(item))
    return paths


def test_mkdocs_yml_parses() -> None:
    cfg = yaml.safe_load(MKDOCS.read_text())
    assert "site_name" in cfg
    assert cfg["site_name"] == "Zynthé"


def test_mkdocs_nav_files_exist(mkdocs_nav) -> None:
    """Every markdown file referenced in mkdocs nav must exist on disk."""
    missing: list = []
    for path in _flatten_nav(mkdocs_nav):
        if not path.endswith(".md"):
            continue
        # Nav paths are repo-root-relative; e.g. 'docs/index.md' or
        # 'index.md' (top-level).
        if not (REPO_ROOT / path).exists():
            missing.append(path)
    assert not missing, f"mkdocs nav references missing files: {missing}"


def test_mkdocs_includes_phase3_status_and_handoff(mkdocs_nav) -> None:
    """The phase-3 status doc and the plan-talk handoff should both
    be on the site nav (per Phase 5 iteration 3 plan).
    """
    paths = _flatten_nav(mkdocs_nav)
    assert any("phase3_status" in p for p in paths)
    assert any("HANDOFF" in p for p in paths)
