"""Tests for the Phase-6 LLM rationale extractor.

Pins:
* build_prompt constructs a 3-shot CoT prompt from a task preset.
* parse_label matches canonical class names case-insensitively and
  returns None on malformed output.
* parse_rationale extracts the reasoning between Reasoning: and
  Answer: markers.
* extract_rationales drops records with unparseable labels and
  iterates the inputs in order.
* The full LLM call is mocked — no model load required for the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_rationales import (
    ESNLI_PRESET,
    SST2_PRESET,
    PRESETS,
    build_prompt,
    default_llm_callable,
    extract_rationales,
    parse_label,
    parse_rationale,
)


# ----------------------------------------------------------------------------
# Task presets
# ----------------------------------------------------------------------------


def test_sst2_preset_classes() -> None:
    assert SST2_PRESET.classes == ("positive", "negative")
    assert len(SST2_PRESET.exemplars) >= 3


def test_esnli_preset_classes() -> None:
    assert set(ESNLI_PRESET.classes) == {"entailment", "neutral", "contradiction"}


def test_presets_registered() -> None:
    assert "sst2" in PRESETS
    assert "esnli" in PRESETS


# ----------------------------------------------------------------------------
# build_prompt
# ----------------------------------------------------------------------------


def test_build_prompt_contains_exemplars_and_input() -> None:
    prompt = build_prompt(SST2_PRESET, "It was a delightful comedy.")
    # Each exemplar is referenced.
    for ex in SST2_PRESET.exemplars:
        assert ex["input"][:30] in prompt
        assert ex["rationale"][:30] in prompt
        assert ex["label"] in prompt
    # The new input appears at the end with "Reasoning:" as the
    # generation prompt.
    assert "delightful comedy" in prompt
    assert prompt.rstrip().endswith("Reasoning:")


def test_build_prompt_includes_task_description() -> None:
    prompt = build_prompt(ESNLI_PRESET, "Premise: X.\nHypothesis: Y.")
    assert "Natural language inference" in prompt
    assert "entailment" in prompt


# ----------------------------------------------------------------------------
# parse_label
# ----------------------------------------------------------------------------


def test_parse_label_matches_canonical_class() -> None:
    txt = "Reasoning: blah\nAnswer: positive"
    assert parse_label(SST2_PRESET, txt) == "positive"


def test_parse_label_is_case_insensitive() -> None:
    assert parse_label(SST2_PRESET, "Answer: POSITIVE") == "positive"


def test_parse_label_returns_none_when_no_marker() -> None:
    assert parse_label(SST2_PRESET, "no marker here") is None


def test_parse_label_returns_none_for_unknown_class() -> None:
    # "maybe" is not in the SST-2 classes list.
    assert parse_label(SST2_PRESET, "Answer: maybe") is None


def test_parse_label_for_esnli_three_class() -> None:
    assert parse_label(ESNLI_PRESET, "Answer: entailment") == "entailment"
    assert parse_label(ESNLI_PRESET, "Answer: contradiction") == "contradiction"
    assert parse_label(ESNLI_PRESET, "Answer: neutral") == "neutral"


# ----------------------------------------------------------------------------
# parse_rationale
# ----------------------------------------------------------------------------


def test_parse_rationale_between_markers() -> None:
    txt = (
        "Reasoning: a quick thought\n"
        "Answer: positive"
    )
    assert parse_rationale(txt) == "a quick thought"


def test_parse_rationale_until_end_when_no_answer() -> None:
    txt = "Reasoning: continues to the end"
    assert parse_rationale(txt) == "continues to the end"


def test_parse_rationale_empty_when_no_marker() -> None:
    assert parse_rationale("no marker") == ""


# ----------------------------------------------------------------------------
# extract_rationales (mocked LLM)
# ----------------------------------------------------------------------------


def _stub_llm(responses: list):
    """Build an llm_callable that returns one response per call."""
    iter_responses = iter(responses)

    def _call(prompts):
        return [next(iter_responses) for _ in prompts]

    return _call


def test_extract_rationales_happy_path() -> None:
    inputs = [{"input": "It was great."}, {"input": "It was bad."}]
    llm = _stub_llm(
        [
            "Reasoning: positive words\nAnswer: positive",
            "Reasoning: negative words\nAnswer: negative",
        ]
    )
    triples = extract_rationales(
        inputs, task_name="sst2", llm_callable=llm
    )
    assert len(triples) == 2
    assert triples[0]["label"] == "positive"
    assert triples[0]["rationale"] == "positive words"
    assert triples[0]["input"] == "It was great."
    assert triples[1]["label"] == "negative"


def test_extract_rationales_drops_unparseable_label() -> None:
    inputs = [{"input": "x"}, {"input": "y"}, {"input": "z"}]
    llm = _stub_llm(
        [
            "Reasoning: x\nAnswer: positive",
            "Reasoning: y\nAnswer: maybe",  # unknown class
            "Reasoning: z",  # no Answer: marker
        ]
    )
    triples = extract_rationales(
        inputs, task_name="sst2", llm_callable=llm
    )
    assert len(triples) == 1
    assert triples[0]["input"] == "x"


def test_extract_rationales_respects_max_records() -> None:
    inputs = [{"input": f"q{i}"} for i in range(5)]
    llm = _stub_llm(
        [
            "Reasoning: a\nAnswer: positive",
            "Answer: positive",
        ]
    )
    triples = extract_rationales(
        inputs, task_name="sst2", llm_callable=llm, max_records=2
    )
    assert len(triples) == 2


def test_extract_rationales_unknown_task_raises() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        extract_rationales(
            [{"input": "x"}],
            task_name="bogus",
            llm_callable=_stub_llm(["Answer: positive"]),
        )


def test_extract_rationales_empty_input() -> None:
    triples = extract_rationales(
        [], task_name="sst2", llm_callable=_stub_llm([])
    )
    assert triples == []


# ----------------------------------------------------------------------------
# CLI roundtrip
# ----------------------------------------------------------------------------


def test_cli_roundtrip_writes_jsonl(tmp_path: Path, monkeypatch, capsys) -> None:
    """End-to-end CLI flow with a stub LLM: write inputs JSONL, run
    extract_rationales, verify outputs JSONL has the right shape.
    """
    import sys as _sys
    from scripts import extract_rationales as er

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(
        json.dumps({"input": "It was a wonderful film."}) + "\n"
        + json.dumps({"input": "It was a tedious film."}) + "\n"
    )
    monkeypatch.setattr(
        er,
        "default_llm_callable",
        lambda **kw: _stub_llm(
            [
                "Reasoning: wonderful signals positive\nAnswer: positive",
                "Reasoning: tedious signals negative\nAnswer: negative",
            ]
        ),
    )
    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "extract_rationales.py",
            "--input-jsonl",
            str(inp),
            "--output-jsonl",
            str(out),
            "--task",
            "sst2",
            "--llm",
            "stub",
        ],
    )
    rc = er.main()
    assert rc == 0
    out_lines = out.read_text().strip().splitlines()
    assert len(out_lines) == 2
    first = json.loads(out_lines[0])
    assert first["label"] == "positive"
    assert first["input"] == "It was a wonderful film."
