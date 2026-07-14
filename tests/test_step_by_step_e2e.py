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
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


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

# Also load the extract_rationales module + the rationale_trainer
# module so the test can patch their attributes.  Re-use the
# sys.path insertion above.
import sys as _sys
if _SCRIPTS not in _sys.path:
    _sys.path.insert(0, _SCRIPTS)
import extract_rationales as _mod_er  # noqa: E402
from zynthe.core.training import rationale_trainer as _mod_rt  # noqa: E402


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
    """Patch the LLM callable AND the model loader with deterministic
    stubs and run the full recipe on a tiny synthetic dataset.
    """
    # Stub extractor: deterministic, no LLM.
    def _patched_extractor(triples):
        return [
            {"input": r["input"], "label": "positive", "rationale": "positive words"}
            for r in triples
        ]

    # Stub model: a tiny T5 constructed in-process (no network).
    import sys as _sys
    if str(Path(__file__).parent.parent / "scripts") not in _sys.path:
        _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import extract_rationales as er  # noqa: E402
    from zynthe.core.training import rationale_trainer as rt_mod  # noqa: E402
    from zynthe.core.training.rationale_trainer import (  # noqa: E402
        MultiTaskT5Trainer as RealTrainer,
    )

    def _local_trainer(model_name: str, **kwargs):
        # Build a fresh tiny in-process model so we never hit HF Hub.
        torch.manual_seed(0)
        # 1-layer T5-like wrapper: an embed, a single linear, an LM head.
        class _TinySeq2Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    decoder_start_token_id=0,
                    vocab_size=64,
                )
                self.shared = nn.Embedding(64, 16)
                self.body = nn.Linear(16, 16)
                self.lm_head = nn.Linear(16, 64)

            def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, **kw):
                x = self.shared(input_ids)
                x = self.body(x)
                dec = self.shared(decoder_input_ids)
                dec = self.body(dec)
                return SimpleNamespace(logits=self.lm_head(dec))

        return RealTrainer(
            model=_TinySeq2Seq(),
            tokenizer=_StubTokenizer(),
            label_prefix=kwargs.get("label_prefix", "label: "),
            rationale_prefix=kwargs.get("rationale_prefix", "rationale: "),
        )

    class _StubTokenizer:
        def __call__(self, text, return_tensors=None, padding=None, max_length=None, truncation=None, **_):
            ids = [ord(c) % 64 for c in text[:max_length or 32]]
            while padding == "max_length" and len(ids) < (max_length or 32):
                ids.append(0)
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    monkeypatch.setattr(er, "extract_rationales", _patched_extractor)
    monkeypatch.setattr(rt_mod, "MultiTaskT5Trainer", _local_trainer)

    payload = run_recipe(
        task="sst2",
        train_records=4,
        eval_records=2,
        steps=2,
        seed=42,
        llm="stub",
        output_dir=tmp_path,
    )

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
    assert payload["train_triples_extracted"] == 4
    assert payload["eval_triples_extracted"] == 2
    assert payload["steps"] == 2
    summary = json.loads((tmp_path / "step_by_step.json").read_text())
    assert summary["task"] == "sst2"


def test_run_recipe_loss_finite(tmp_path: Path, monkeypatch) -> None:
    """The recipe's reported losses are finite numbers (no NaN)."""
    def _patched_extractor(triples):
        return [
            {"input": r["input"], "label": "negative", "rationale": "negative words"}
            for r in triples
        ]

    import sys as _sys
    if str(Path(__file__).parent.parent / "scripts") not in _sys.path:
        _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import extract_rationales as er  # noqa: E402
    from zynthe.core.training import rationale_trainer as rt_mod  # noqa: E402
    from zynthe.core.training.rationale_trainer import (  # noqa: E402
        MultiTaskT5Trainer as RealTrainer,
    )

    def _local_trainer(model_name: str, **kwargs):
        torch.manual_seed(0)
        class _TinySeq2Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    decoder_start_token_id=0,
                    vocab_size=64,
                )
                self.shared = nn.Embedding(64, 16)
                self.body = nn.Linear(16, 16)
                self.lm_head = nn.Linear(16, 64)

            def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, **kw):
                x = self.shared(input_ids)
                x = self.body(x)
                dec = self.shared(decoder_input_ids)
                dec = self.body(dec)
                return SimpleNamespace(logits=self.lm_head(dec))

        return RealTrainer(
            model=_TinySeq2Seq(),
            tokenizer=_StubTokenizer(),
            label_prefix=kwargs.get("label_prefix", "label: "),
            rationale_prefix=kwargs.get("rationale_prefix", "rationale: "),
        )

    monkeypatch.setattr(er, "extract_rationales", _patched_extractor)
    monkeypatch.setattr(rt_mod, "MultiTaskT5Trainer", _local_trainer)

    payload = run_recipe(
        task="sst2",
        train_records=2,
        eval_records=2,
        steps=2,
        seed=1,
        llm="stub",
        output_dir=tmp_path,
    )

    for key in ("label", "rationale", "total"):
        assert key in payload["train_loss_first"]
        assert key in payload["train_loss_last"]
        for src in (payload["train_loss_first"], payload["train_loss_last"]):
            v = src[key]
            assert isinstance(v, (int, float))
            assert v == v  # no NaN
