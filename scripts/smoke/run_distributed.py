"""Run a tiny accelerate-ready distillation on Modal L4 (single GPU).

Confirms the ``prepare_distillation`` integration wires the distiller
into HuggingFace ``accelerate`` and runs end-to-end on a single GPU.
"""

from __future__ import annotations

import argparse
import os
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
        "accelerate==1.0.1",
        "pytest==8.3.3",
    )
)

app = modal.App("zynthe-distributed-smoke")


@app.function(
    gpu="L4",
    timeout=60 * 15,
    cpu=4.0,
    memory=8192,
    image=image,
)
def _run_distributed(steps: int) -> int:
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
        "/repo/scripts/smoke/run_distributed_local.py",
        "--steps",
        str(steps),
        "--output",
        "/repo/tests/smoke/results/distributed.json",
    ]
    proc = subprocess.run(cmd, text=True, env={"GIT_COMMIT": "modal", **subprocess.os.environ})
    return int(proc.returncode)


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    steps: int = 20,
    timeout_min: int = 12,
) -> None:
    if gpu not in ("T4", "L4", "A10G", "A100"):
        print(f"GPU {gpu!r} not supported", file=sys.stderr)
        sys.exit(2)
    fn = _run_distributed.with_options(gpu=gpu, timeout=timeout_min * 60)
    rc = fn.remote(steps=steps)
    print(f"distributed smoke exit code: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout-min", type=int, default=12)
    args = parser.parse_args()
    main(gpu=args.gpu, steps=args.steps, timeout_min=args.timeout_min)
