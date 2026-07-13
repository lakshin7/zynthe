"""Baseline-distillation smoke experiment.

Implements Zynthé report §227-249, Experiment #1 (Baseline
Distillation) at tiny scale so it fits on Modal L4.

This is *not* a GLUE benchmark — it's a smoke proof that the
distillation pipeline (KDHintonDistiller) reduces the loss
monotonically on a tiny teacher→student pair using synthetic
data.

Usage::

    modal run scripts/smoke/run_baseline_distill.py --gpu L4

Output is written to ``tests/smoke/results/baseline_sst2_<commit>.json``
plus a markdown summary at ``docs/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import json
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
        "pytest==8.3.3",
    )
)

app = modal.App("zynthe-baseline-smoke")


@app.function(
    gpu="L4",
    timeout=60 * 30,
    cpu=4.0,
    memory=8192,
    image=image,
)
def _run_baseline(steps: int, seed: int, batch_size: int) -> int:
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
        "/repo/scripts/smoke/run_baseline_distill_local.py",
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--output",
        "/repo/tests/smoke/results/baseline_sst2.json",
    ]
    env = {"GIT_COMMIT": "modal"}
    proc = subprocess.run(cmd, text=True, env={**subprocess.os.environ, **env})
    return int(proc.returncode)


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    steps: int = 50,
    seed: int = 42,
    batch_size: int = 4,
    timeout_min: int = 20,
) -> None:
    if gpu not in ("T4", "L4", "A10G", "A100"):
        print(f"GPU {gpu!r} not supported", file=sys.stderr)
        sys.exit(2)
    fn = _run_baseline.with_options(gpu=gpu, timeout=timeout_min * 60)
    rc = fn.remote(steps=steps, seed=seed, batch_size=batch_size)
    print(f"baseline smoke exit code: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-min", type=int, default=20)
    args = parser.parse_args()
    main(
        gpu=args.gpu,
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        timeout_min=args.timeout_min,
    )
