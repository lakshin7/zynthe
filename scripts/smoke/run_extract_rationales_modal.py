"""Modal wrapper for the LLM rationale extractor smoke.

Clones the repo, installs zynthe, runs ``extract_rationales.py`` on
a small synthetic input JSONL using ``google/flan-t5-base``.

Output: ``tests/smoke/results/rationale_extract.jsonl`` (the triples
the extractor wrote).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import modal

REPO_URL = "https://github.com/lakshin7/zynthe.git"
BRANCH = "main"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.0",
    )
)

app = modal.App("zynthe-extract-rationales-smoke")


@app.function(
    gpu="L4",
    timeout=60 * 15,
    cpu=4.0,
    memory=16384,
    image=image,
)
def _run_extract(llm: str, max_records: int) -> int:
    subprocess.run(
        [
            "bash",
            "-lc",
            (
                "rm -rf /repo && "
                f"git clone --depth 1 --branch {BRANCH} {REPO_URL} /repo && "
                "cd /repo && pip install -e '.[dev]' --quiet"
            ),
        ],
        check=True,
        text=True,
    )

    # Write a synthetic SST-2 input JSONL.
    inputs = Path("/repo/tests/smoke/results/_sst2_synth_inputs.jsonl")
    inputs.parent.mkdir(parents=True, exist_ok=True)
    inputs.write_text(
        "\n".join(
            [
                '{"input": "This film was a delight — funny, warm, beautifully acted."}',
                '{"input": "I found the plot predictable and the dialogue wooden."}',
                '{"input": "It was not the worst movie I have seen, but not memorable either."}',
                '{"input": "An absolute masterpiece of cinema — every frame felt alive."}',
                '{"input": "A tedious, lifeless, and frankly boring experience."}',
                '{"input": "Somewhat enjoyable, though the second half dragged on."}',
                '{"input": "Brilliant, funny, touching — I will rewatch this."}',
                '{"input": "Disappointing, predictable, and derivative."}',
            ]
        )
        + "\n"
    )

    proc = subprocess.run(
        [
            "python",
            "/repo/scripts/extract_rationales.py",
            "--input-jsonl",
            str(inputs),
            "--output-jsonl",
            "/repo/tests/smoke/results/rationale_extract.jsonl",
            "--task",
            "sst2",
            "--llm",
            llm,
            "--max-records",
            str(max_records),
            "--batch-size",
            "4",
        ],
        text=True,
    )
    return int(proc.returncode)


@app.local_entrypoint()
def main(
    llm: str = "google/flan-t5-base",
    max_records: int = 8,
    timeout_min: int = 12,
) -> None:
    fn = _run_extract.with_options(gpu="L4", timeout=timeout_min * 60)
    rc = fn.remote(llm=llm, max_records=max_records)
    print(f"extract-rationales exit code: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default="google/flan-t5-base")
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--timeout-min", type=int, default=12)
    args = parser.parse_args()
    main(llm=args.llm, max_records=args.max_records, timeout_min=args.timeout_min)
