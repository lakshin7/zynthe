"""LLM-based rationale extraction (Distill step-by-step, paper §3.1).

Given an unlabelled dataset (JSONL with just ``{"input": str}``
records, or a HuggingFace ``datasets`` load), prompt a small
instruction-tuned LLM with few-shot chain-of-thought exemplars and
parse out ``(rationale, label)`` from the LLM's response.

The output JSONL is directly consumable by
:class:`zynthe.data.rationale_dataset.RationaleDataset`.

Usage::

    python scripts/extract_rationales.py \
        --input-jsonl data/inputs.jsonl \
        --output-jsonl data/rationales.jsonl \
        --task sst2 \
        --llm google/flan-t5-base \
        --max-records 200

The extractor is decoupled from the actual model call so tests can
pass a deterministic ``llm_callable`` that returns fixed strings
without spinning up a real model.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Task presets
# ----------------------------------------------------------------------------


@dataclass
class TaskPreset:
    """Per-task few-shot CoT configuration for the LLM.

    Attributes:
        name: Task identifier (e.g. ``"sst2"``).
        description: Human-readable note.
        classes: Class labels the LLM is asked to pick from.
        exemplars: ``(input, rationale, label)`` triples used as
            few-shot context.
        parse_label_regex: Regex used to extract the label from the
            LLM's response.  Must contain a capture group named
            ``label`` (e.g. ``Answer:\\s*(\\w+)``).
    """

    name: str
    description: str
    classes: Sequence[str]
    exemplars: Sequence[Dict[str, str]]  # each has "input", "rationale", "label"
    parse_label_regex: str = r"(?:Answer|answer):\s*(\w+)"


SST2_PRESET = TaskPreset(
    name="sst2",
    description="Binary sentiment classification (Stanford Sentiment Treebank)",
    classes=("positive", "negative"),
    exemplars=(
        {
            "input": "The movie was an absolute delight — a warm, funny, beautifully acted gem.",
            "rationale": (
                "The reviewer uses overwhelmingly positive words ('delight', "
                "'warm', 'funny', 'gem') that signal a clearly positive "
                "sentiment."
            ),
            "label": "positive",
        },
        {
            "input": "I found the plot predictable and the dialogue wooden; it was a bore.",
            "rationale": (
                "The reviewer uses negative descriptors ('predictable', "
                "'wooden', 'bore') which all point to a clearly negative "
                "sentiment."
            ),
            "label": "negative",
        },
        {
            "input": "It wasn't the worst film I've seen, but it certainly wasn't memorable either.",
            "rationale": (
                "The reviewer hedges ('wasn't the worst', 'wasn't memorable') "
                "without strong positive or negative emotion; this is a "
                "negative-leaning, low-energy review."
            ),
            "label": "negative",
        },
    ),
    parse_label_regex=r"(?:Answer|answer):\s*(\w+)",
)


ESNLI_PRESET = TaskPreset(
    name="esnli",
    description="Natural language inference (entailment / contradiction / neutral)",
    classes=("entailment", "neutral", "contradiction"),
    exemplars=(
        {
            "input": (
                "Premise: A man inspects the contents of a fridge.\n"
                "Hypothesis: A man is opening a refrigerator."
            ),
            "rationale": (
                "Opening a refrigerator is essentially the same as inspecting "
                "its contents, so the hypothesis is entailed by the premise."
            ),
            "label": "entailment",
        },
        {
            "input": (
                "Premise: A man is shopping for a new tie.\n"
                "Hypothesis: A man is attending a wedding."
            ),
            "rationale": (
                "Shopping for a tie does not imply attending a wedding, "
                "even if ties are sometimes worn at weddings. The "
                "relationship is neutral."
            ),
            "label": "neutral",
        },
        {
            "input": (
                "Premise: Two children are sitting on a wooden bench.\n"
                "Hypothesis: Two children are standing on a trampoline."
            ),
            "rationale": (
                "Sitting on a bench contradicts standing on a trampoline — "
                "the posture and the object are both different."
            ),
            "label": "contradiction",
        },
    ),
    parse_label_regex=r"(?:Answer|answer):\s*(\w+)",
)


PRESETS: Dict[str, TaskPreset] = {
    "sst2": SST2_PRESET,
    "esnli": ESNLI_PRESET,
}


# ----------------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------------


def build_prompt(task: TaskPreset, input_text: str) -> str:
    """Construct a few-shot CoT prompt for ``input_text``."""
    parts: List[str] = [task.description, ""]
    for ex in task.exemplars:
        parts.append(f"Question: {ex['input']}")
        parts.append(f"Reasoning: {ex['rationale']}")
        parts.append(f"Answer: {ex['label']}")
        parts.append("")
    parts.append(f"Question: {input_text}")
    parts.append("Reasoning:")
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------------


def parse_label(task: TaskPreset, llm_output: str) -> Optional[str]:
    """Extract the predicted label from ``llm_output``.

    Returns the matched class (case-insensitive match against
    ``task.classes``) or ``None`` if no label could be parsed.
    """
    pattern = re.compile(task.parse_label_regex)
    m = pattern.search(llm_output)
    if not m:
        return None
    raw = m.group(1).strip().lower()
    # Match against the canonical class names.
    for cls in task.classes:
        if raw == cls.lower():
            return cls
    return raw if raw in (c.lower() for c in task.classes) else None


def parse_rationale(llm_output: str) -> str:
    """Pull the rationale out of the LLM's response.

    The rationale is whatever sits between the ``Reasoning:`` marker
    and the ``Answer:`` marker.  If only ``Reasoning:`` is present,
    take the rest of the output.
    """
    reasoning_marker = "Reasoning:"
    answer_marker = "Answer:"
    if reasoning_marker not in llm_output:
        return ""
    after = llm_output.split(reasoning_marker, 1)[1].strip()
    if answer_marker in after:
        return after.split(answer_marker, 1)[0].strip()
    return after.strip()


# ----------------------------------------------------------------------------
# Extractor (the public entry point)
# ----------------------------------------------------------------------------


def extract_rationales(
    inputs: Iterable[Dict[str, str]],
    *,
    task_name: str,
    llm_callable: Callable[[List[str]], List[str]],
    max_records: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Run the LLM extractor on a stream of input records.

    Args:
        inputs: Iterable of dicts, each with at least an ``"input"``
            key.
        task_name: One of the keys in :data:`PRESETS`.
        llm_callable: A function that takes a list of prompts and
            returns a list of LLM outputs (one per prompt, same order).
            Tests pass a deterministic stub; production passes
            :func:`default_llm_callable`.
        max_records: If set, only the first ``max_records`` inputs
            are processed.

    Returns:
        A list of ``{"input", "label", "rationale"}`` dicts.  Items
        where the label could not be parsed are dropped (the LLM
        may have produced malformed output).
    """
    if task_name not in PRESETS:
        raise ValueError(
            f"Unknown task {task_name!r}; available: {list(PRESETS)}"
        )
    task = PRESETS[task_name]

    records = list(inputs)
    if max_records is not None:
        records = records[:max_records]

    prompts = [build_prompt(task, r["input"]) for r in records]
    logger.info(
        "extract_rationales: %d inputs on task %s", len(prompts), task_name
    )
    outputs = llm_callable(prompts)

    results: List[Dict[str, str]] = []
    for rec, llm_out in zip(records, outputs):
        label = parse_label(task, llm_out)
        if label is None:
            logger.warning(
                "could not parse label from LLM output; skipping: %r", llm_out[:200]
            )
            continue
        rationale = parse_rationale(llm_out)
        results.append(
            {
                "input": rec["input"],
                "label": label,
                "rationale": rationale,
            }
        )
    return results


def default_llm_callable(
    model_name: str = "google/flan-t5-base",
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Callable[[List[str]], List[str]]:
    """Return a closure that runs ``model.generate`` on a list of prompts.

    Use this for production.  Tests should pass their own stub.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    def _call(prompts: List[str]) -> List[str]:
        outputs: List[str] = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                for ids in gen:
                    outputs.append(tokenizer.decode(ids, skip_special_tokens=True))
        return outputs

    return _call


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------


def _read_inputs_jsonl(path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", type=Path, required=True,
                  help="JSONL of {'input': str} records.")
    p.add_argument("--output-jsonl", type=Path, required=True,
                  help="Where to write the (input, label, rationale) JSONL.")
    p.add_argument("--task", choices=list(PRESETS), default="sst2")
    p.add_argument("--llm", default="google/flan-t5-base",
                  help="HF model id for rationale generation.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    random.seed(args.seed)

    inputs = _read_inputs_jsonl(args.input_jsonl)
    if not inputs:
        print(f"[extract_rationales] no inputs in {args.input_jsonl}", file=sys.stderr)
        return 1

    print(
        f"[extract_rationales] {len(inputs)} inputs on task={args.task}, "
        f"llm={args.llm}, max_records={args.max_records}"
    )
    llm = default_llm_callable(
        model_name=args.llm,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    started = time.time()
    triples = extract_rationales(
        inputs, task_name=args.task, llm_callable=llm, max_records=args.max_records
    )
    elapsed = time.time() - started

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple) + "\n")

    print(
        f"[extract_rationales] {len(triples)}/{len(inputs)} triples in "
        f"{elapsed:.1f}s -> {args.output_jsonl}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
