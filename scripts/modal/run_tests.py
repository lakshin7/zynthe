"""Run the Zynthe pytest suite on a Modal L4 (or T4 / A10G / A100) GPU.

Why Modal: pytest-with-torch cannot run on the developer's CPU-only
laptop, and GitHub Actions runners are CPU-only by default. Modal gives
us a real CUDA runtime for ~$0.50/hr of testing.

Usage:

    modal run scripts/modal/run_tests.py --gpu L4
    modal run scripts/modal/run_tests.py --gpu T4 --select tests/test_distillers.py
    modal run scripts/modal/run_tests.py --gpu A10G --extra-args "-v --tb=short"

Default is L4 (24 GB, ~$0.80/hr) to match the Phase 2 universal-model
smoke gate plan.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import modal

REPO_URL = "https://github.com/lakshin7/zynthe.git"
DEFAULT_BRANCH = "main"
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
        "accelerate==1.0.1",
        "pytest==8.3.3",
        "pytest-cov==5.0.0",
        "flake8==7.1.0",
        "omegaconf==2.3.0",
        "pyyaml==6.0.2",
        "numpy==1.26.4",
        "rich==13.9.0",
        "psutil==6.1.0",
        "tqdm==4.66.5",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "pandas==2.2.3",
        "torchvision==0.19.1",
        "Pillow==10.4.0",
        "onnx==1.17.0",
        "onnxruntime==1.19.2",
        "optimum[onnxruntime]==1.23.3",
    )
)

app = modal.App("zynthe-tests")


@app.function(
    gpu="L4",
    timeout=60 * 45,
    cpu=4.0,
    memory=8192,
    image=image.run_commands(
        "git clone --depth 1 --branch main " + REPO_URL + " /repo",
    ),
)
def _run_pytest(branch: str, select: str, extra_args: str, gpu: str) -> int:
    """Install the repo and run pytest. Return the pytest exit code."""
    if branch != DEFAULT_BRANCH:
        # Re-clone at a non-default branch.
        subprocess.run(
            [
                "bash",
                "-lc",
                f"rm -rf /repo && git clone --depth 1 --branch {branch} {REPO_URL} /repo",
            ],
            check=True,
        )

    print(f"[modal] running pytest on GPU={gpu}")
    cmd = [
        "bash",
        "-lc",
        (
            "cd /repo && "
            "pip install -e '.[dev]' --quiet && "
            f"pytest {select} {extra_args}"
        ),
    ]
    proc = subprocess.run(cmd, text=True)
    return proc.returncode


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    branch: str = DEFAULT_BRANCH,
    select: str = "tests/",
    extra_args: str = "-v",
    timeout_min: int = 45,
) -> None:
    """CLI entry: forward the user args to the Modal GPU function."""
    if gpu not in GPU_CHOICES:
        print(f"GPU {gpu!r} not in {GPU_CHOICES}", file=sys.stderr)
        sys.exit(2)

    # The @app.function decorator pins GPU="L4"; rebuild an ad-hoc
    # call with the user's chosen GPU via .with_options().
    fn = _run_pytest.with_options(gpu=gpu, timeout=timeout_min * 60)
    rc = fn.remote(
        branch=branch,
        select=select,
        extra_args=extra_args,
        gpu=gpu,
    )
    print(f"pytest exit code: {rc}")
    sys.exit(int(rc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4", choices=GPU_CHOICES)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--select", default="tests/")
    parser.add_argument("--extra-args", default="-v")
    parser.add_argument("--timeout-min", type=int, default=45)
    args = parser.parse_args()
    main(
        gpu=args.gpu,
        branch=args.branch,
        select=args.select,
        extra_args=args.extra_args,
        timeout_min=args.timeout_min,
    )
