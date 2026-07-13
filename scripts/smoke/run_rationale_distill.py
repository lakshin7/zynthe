"""Rationale-distillation smoke experiment (Modal wrapper).

This wraps the local rationale-distill smoke (see
``run_rationale_distill_local.py``) on a Modal L4.  The local script
generates a small synthetic JSONL of (input, label, rationale)
triples and runs ~30 SGD steps with ``RationaleDistiller`` on a
tiny T5.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

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
        "pytest==8.3.3",
    )
)

app = modal.App("zynthe-rationale-smoke")


@app.function(
    gpu="L4",
    timeout=60 * 30,
    cpu=4.0,
    memory=8192,
    image=image,
)
def _run_rationale(steps: int, seed: int) -> int:
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
    cmd = [
        "python",
        "/repo/scripts/smoke/run_rationale_distill_local.py",
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--output",
        "/repo/tests/smoke/results/rationale.json",
    ]
    proc = subprocess.run(cmd, text=True, env={"GIT_COMMIT": "modal", **subprocess.os.environ})
    return int(proc.returncode)


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    steps: int = 30,
    seed: int = 42,
    timeout_min: int = 20,
) -> None:
    if gpu not in ("T4", "L4", "A10G", "A100"):
        print(f"GPU {gpu!r} not supported", file=sys.stderr)
        sys.exit(2)
    fn = _run_rationale.with_options(gpu=gpu, timeout=timeout_min * 60)
    rc = fn.remote(steps=steps, seed=seed)
    print(f"rationale smoke exit code: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-min", type=int, default=20)
    args = parser.parse_args()
    main(
        gpu=args.gpu,
        steps=args.steps,
        seed=args.seed,
        timeout_min=args.timeout_min,
    )
