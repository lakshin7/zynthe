"""Run the universal 5-family smoke gate on Modal L4.

This wraps ``scripts/smoke/universal_smoke.py`` so we can invoke it
via ``modal run``.

We default to L4 because that's a great price/performance tradeoff
for the smoke scale.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import modal

REPO_URL = "https://github.com/lakshin7/zynthe.git"
BRANCH = "main"
PAIRS = ["bert", "vit", "gpt2", "clip", "resnet"]
GPU_CHOICES = ["T4", "L4", "A10G", "A100"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.0",
        "datasets==3.1.0",
        "evaluate==0.4.3",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "Pillow==10.4.0",
    )
    .pip_install("torchvision==0.19.1")
)

app = modal.App("zynthe-smoke")


@app.function(
    gpu="L4",
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
    image=image,
)
def _run_smoke(
    branch: str,
    pairs: str,
    steps: int,
    quick: bool,
    max_budget: float | None,
    commit: str | None,
) -> int:
    """Clone the latest *branch*, install zynthe, and run the smoke gate.

    Returns the smoke-gate exit code (0 = all pairs succeeded, 1 = at
    least one failed, 2 = setup failure).
    """
    subprocess.run(
        [
            "bash",
            "-lc",
            (
                "rm -rf /repo && "
                f"git clone --depth 1 --branch {branch} {REPO_URL} /repo && "
                "cd /repo && git log -1 --oneline"
            ),
        ],
        check=True,
        text=True,
    )

    cmd = ["bash", "-lc", "cd /repo && pip install -e '.[dev]' --quiet"]
    subprocess.run(cmd, check=True, text=True)

    smoke_args = [
        "python",
        "/repo/scripts/smoke/universal_smoke.py",
        "--pairs",
        pairs,
        "--steps",
        str(steps),
        "--output",
        "/repo/tests/smoke/results/universal_smoke_modal.json",
    ]
    if quick:
        smoke_args.append("--quick")
    if max_budget is not None:
        smoke_args.extend(["--max-budget", str(max_budget)])

    env = {}
    if commit:
        env["GIT_COMMIT"] = commit

    proc = subprocess.run(smoke_args, text=True, env={**subprocess.os.environ, **env})
    return int(proc.returncode)


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    branch: str = BRANCH,
    pairs: str = "all",
    steps: int = 50,
    quick: bool = False,
    max_budget: float = 2.0,
    timeout_min: int = 60,
) -> None:
    if gpu not in GPU_CHOICES:
        print(f"GPU {gpu!r} not in {GPU_CHOICES}", file=sys.stderr)
        sys.exit(2)
    fn = _run_smoke.with_options(gpu=gpu, timeout=timeout_min * 60)
    rc = fn.remote(
        branch=branch,
        pairs=pairs,
        steps=steps,
        quick=quick,
        max_budget=max_budget,
        commit=None,
    )
    print(f"smoke gate exit code: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4", choices=GPU_CHOICES)
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-budget", type=float, default=2.0)
    parser.add_argument("--timeout-min", type=int, default=60)
    args = parser.parse_args()
    main(
        gpu=args.gpu,
        branch=args.branch,
        pairs=args.pairs,
        steps=args.steps,
        quick=args.quick,
        max_budget=args.max_budget,
        timeout_min=args.timeout_min,
    )
