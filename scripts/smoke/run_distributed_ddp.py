"""Multi-GPU DDP smoke via torchrun on Modal L4 (2-5 GPUs).

Runs ``scripts/smoke/run_distributed_ddp_local.py`` under
``torchrun --nproc_per_node=N`` on Modal.  Validates that:
- accelerate.prepare() wraps the model in DistributedDataParallel.
- Each rank produces the same loss (data-parallel sync).
- Final loss is finite and decays over the run.

Usage::

    modal run scripts/smoke/run_distributed_ddp.py --gpus 2
    modal run scripts/smoke/run_distributed_ddp.py --gpus 4 --steps 30

The number of GPUs is capped at 5 (per the user-configured limit).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import modal

REPO_URL = "https://github.com/lakshin7/zynthe.git"
BRANCH = "main"
MAX_GPUS = 5

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

app = modal.App("zynthe-ddp-smoke")


@app.function(
    gpu="L4:2",  # default — overridden by --gpus flag
    timeout=60 * 20,
    cpu=4.0,
    memory=16384,
    image=image,
)
def _run_ddp(gpus: int, steps: int) -> int:
    print(f"[ddp-smoke] launching torchrun with nproc_per_node={gpus}")
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
        "torchrun",
        f"--nproc_per_node={gpus}",
        "/repo/scripts/smoke/run_distributed_ddp_local.py",
        "--steps",
        str(steps),
        "--output",
        "/repo/tests/smoke/results/distributed_ddp.json",
    ]
    proc = subprocess.run(cmd, text=True, env={"GIT_COMMIT": "modal", **subprocess.os.environ})
    return int(proc.returncode)


@app.local_entrypoint()
def main(gpus: int = 2, steps: int = 20, timeout_min: int = 15) -> None:
    if gpus < 1 or gpus > MAX_GPUS:
        print(f"--gpus must be in [1, {MAX_GPUS}]; got {gpus}", file=sys.stderr)
        sys.exit(2)
    # Modal supports gpu="L4:N" for 1 <= N <= 8.  Pass dynamically.
    gpu_spec = f"L4:{gpus}"
    fn = _run_ddp.with_options(gpu=gpu_spec, timeout=timeout_min * 60)
    rc = fn.remote(gpus=gpus, steps=steps)
    print(f"ddp smoke exit code: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=2, help="Number of L4 GPUs (1-5).")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout-min", type=int, default=15)
    args = parser.parse_args()
    main(gpus=args.gpus, steps=args.steps, timeout_min=args.timeout_min)
