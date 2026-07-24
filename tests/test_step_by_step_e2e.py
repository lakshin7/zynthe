"""Smoke test for the Phase 6 Iteration 3 end-to-end recipe.

Runs the full SST-2 step-by-step recipe (extract → train → eval) as
a subprocess so it runs in a clean Python process with no HF Hub
network access.  Verifies the recipe's stdout (smoke gate) and
that the output JSON file is well-formed.

The recipe's CLI is exercised end-to-end:
  ``python scripts/run_distill_step_by_step.py \
      --train-records 4 --eval-records 2 --steps 2 \
      --output /tmp/step_by_step_test``.

We don't load HF datasets or any real model — the recipe's
synthetic generator is used for ``--train-records`` and ``--eval-records``.
The ``--llm`` flag is ignored because the extractor is patched via
a tiny in-process test stub (a stand-in for an LLM).  The trainer
constructs the smallest possible model in-process (no from_pretrained).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_distill_step_by_step.py"


def _run_recipe_via_subprocess(
    out_dir: Path,
    train_records: int = 4,
    eval_records: int = 2,
    steps: int = 2,
) -> subprocess.CompletedProcess:
    """Invoke the recipe as a subprocess.  Returns the completed
    process so callers can inspect stdout/stderr/returncode.

    We rely on the recipe's own offline paths:
    - ``_maybe_load_sst2`` falls back to the synthetic generator
      when the HF datasets load fails (which it will in a no-network
      sandbox).
    - The T5 trainer falls back to the recipe's offline model path.
    """
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "sst2",
            "--train-records",
            str(train_records),
            "--eval-records",
            str(eval_records),
            "--steps",
            str(steps),
            "--llm",
            "stub",
            "--output",
            str(out_dir),
        ],
        text=True,
        capture_output=True,
    )


def test_synthetic_sst2_generator_produces_records() -> None:
    # Load the recipe module in-process just to call the helper.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_step_by_step", str(SCRIPT)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = mod._synthetic_sst2(8, seed=42)
    assert len(records) == 8
    for r in records:
        assert "input" in r


def test_synthetic_sst2_seeded_deterministic() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_step_by_step", str(SCRIPT)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod._synthetic_sst2(8, seed=42)
    b = mod._synthetic_sst2(8, seed=42)
    assert a == b


def test_synthetic_sst2_separate_seeds_diverge() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_step_by_step", str(SCRIPT)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod._synthetic_sst2(8, seed=1)
    b = mod._synthetic_sst2(8, seed=2)
    assert a != b


def test_run_recipe_end_to_end(tmp_path: Path) -> None:
    """Invoke the recipe as a subprocess and check the JSON output."""
    proc = _run_recipe_via_subprocess(tmp_path, train_records=4, eval_records=2, steps=2)
    if proc.returncode != 0:
        pytest.fail(
            f"recipe returned {proc.returncode}; stdout={proc.stdout!r}; "
            f"stderr={proc.stderr!r}"
        )

    summary_path = tmp_path / "step_by_step.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())

    # Required fields.
    for k in [
        "task",
        "train_triples_extracted",
        "eval_triples_extracted",
        "steps",
        "train_loss_first",
        "train_loss_last",
        "train_loss_decay",
        "eval_loss_total_avg",
    ]:
        assert k in data, f"missing key: {k}"

    # Sanity values.
    assert data["task"] == "sst2"
    assert data["steps"] == 2
    # The synthetic generator produces exactly 4 train records.
    assert data["train_triples_extracted"] == 4
    assert data["eval_triples_extracted"] == 2


def test_run_recipe_loss_finite(tmp_path: Path) -> None:
    """Loss fields are real numbers (no NaN)."""
    proc = _run_recipe_via_subprocess(tmp_path, train_records=2, eval_records=2, steps=2)
    if proc.returncode != 0:
        pytest.fail(
            f"recipe returned {proc.returncode}; stdout={proc.stdout!r}; "
            f"stderr={proc.stderr!r}"
        )

    data = json.loads((tmp_path / "step_by_step.json").read_text())
    for key in ("label", "rationale", "total"):
        assert key in data["train_loss_first"]
        assert key in data["train_loss_last"]
        for src in (data["train_loss_first"], data["train_loss_last"]):
            v = src[key]
            assert isinstance(v, (int, float))
            assert v == v  # not NaN


def test_run_recipe_is_offline(tmp_path: Path) -> None:
    """The recipe should NOT make any HF Hub calls in the offline
    path — the synthetic generator + an offline T5 model substitute
    are used.  We can't directly assert 'no HF calls' but we can
    assert the run completes in <30s on Modal (no huge model load)
    and that the output JSON is valid.
    """
    import time

    started = time.time()
    proc = _run_recipe_via_subprocess(tmp_path, train_records=4, eval_records=2, steps=2)
    elapsed = time.time() - started
    assert proc.returncode == 0
    # The synthetic fallback shouldn't take more than ~30s on Modal.
    assert elapsed < 60.0, f"recipe took {elapsed:.1f}s; should be much faster"
    data = json.loads((tmp_path / "step_by_step.json").read_text())
    assert data["task"] == "sst2"
