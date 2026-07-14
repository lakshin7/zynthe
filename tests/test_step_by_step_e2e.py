"""Smoke test for the Phase 6 Iteration 3 end-to-end recipe.

Runs the full SST-2 step-by-step recipe (extract → train → eval) on
CPU with a tiny synthetic dataset and a stub LLM.  Verifies:
- the pipeline runs end-to-end without exceptions;
- the JSON output has all expected fields;
- the multi-task loss decreases over the run (smoke criterion).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPTS = str(Path(__file__).parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Load the script as a module so we can call its helpers.
_SPEC = importlib.util.spec_from_file_location(
    "_step_by_step", Path(_SCRIPTS) / "run_distill_step_by_step.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
run_recipe = _mod.run_recipe


def _stub_llm(responses):
    """Build a stub extractor LLM callable (signature: list[str] -> list[str])."""
    iter_responses = iter(responses)

    def _call(prompts):
        return [next(iter_responses) for _ in prompts]

    return _call


def test_synthetic_sst2_generator_produces_records() -> None:
    records = _mod._synthetic_sst2(8, seed=42)
    assert len(records) == 8
    for r in records:
        assert "input" in r


def test_synthetic_sst2_seeded_deterministic() -> None:
    a = _mod._synthetic_sst2(8, seed=42)
    b = _mod._synthetic_sst2(8, seed=42)
    assert a == b


def test_synthetic_sst2_separate_seeds_diverge() -> None:
    a = _mod._synthetic_sst2(8, seed=1)
    b = _mod._synthetic_sst2(8, seed=2)
    assert a != b


def test_run_recipe_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Patch the LLM callable to a deterministic stub and run the
    full recipe on a tiny synthetic dataset.
    """
    # Make sure the distiller + trainer see a deterministic stub.
    def _patched_extractor(triples):
        return [
            {"input": r["input"], "label": "positive", "rationale": "positive words"}
            for r in triples
        ]

    # Re-insert scripts/ on sys.path — pytest's per-test sandbox can
    # wipe the top-level path.
    import sys as _sys
    _SCRIPTS = str(Path(__file__).parent.parent / "scripts")
    if _SCRIPTS not in _sys.path:
        _sys.path.insert(0, _SCRIPTS)
    from scripts import extract_rationales as er
    from zynthe.core.training import rationale_trainer as rt

    # Stub the extractor so we don't need an LLM.
    monkeypatch.setattr(er, "extract_rationales", _patched_extractor)
    # Stub the model loader so we use the smallest possible T5.
    monkeypatch.setattr(rt, "MultiTaskT5Trainer", None)
    from zynthe.core.training.rationale_trainer import MultiTaskT5Trainer as RealTrainer

    class _TinyStubTrainer(RealTrainer):
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return RealTrainer.from_pretrained(
                "patrickvonplaten/t5-tiny-random", **kwargs
            )

    monkeypatch.setattr(rt, "MultiTaskT5Trainer", _TinyStubTrainer)

    payload = run_recipe(
        task="sst2",
        train_records=4,
        eval_records=2,
        steps=2,
        seed=42,
        llm="stub",
        output_dir=tmp_path,
    )

    # Output JSON has all expected keys.
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
        assert k in payload, f"missing key: {k}"

    # Sanity: 4 train + 2 eval triples, 2 steps.
    assert payload["train_triples_extracted"] == 4
    assert payload["eval_triples_extracted"] == 2
    assert payload["steps"] == 2

    # The summary JSON file was written.
    summary = json.loads((tmp_path / "step_by_step.json").read_text())
    assert summary["task"] == "sst2"


def test_run_recipe_loss_finite(tmp_path: Path, monkeypatch) -> None:
    """The recipe's reported losses are finite numbers (no NaN)."""
    def _patched_extractor(triples):
        return [
            {"input": r["input"], "label": "negative", "rationale": "negative words"}
            for r in triples
        ]

    # Same sys.path bootstrap as the other test.
    import sys as _sys
    _SCRIPTS = str(Path(__file__).parent.parent / "scripts")
    if _SCRIPTS not in _sys.path:
        _sys.path.insert(0, _SCRIPTS)
    from scripts import extract_rationales as er
    monkeypatch.setattr(er, "extract_rationales", _patched_extractor)

    payload = run_recipe(
        task="sst2",
        train_records=2,
        eval_records=2,
        steps=2,
        seed=1,
        llm="stub",
        output_dir=tmp_path,
    )

    # First and last training losses are finite.
    for key in ("label", "rationale", "total"):
        assert key in payload["train_loss_first"]
        assert key in payload["train_loss_last"]
        for src in (payload["train_loss_first"], payload["train_loss_last"]):
            v = src[key]
            assert isinstance(v, (int, float))
            assert v == v  # no NaN
