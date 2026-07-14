"""Smoke-level test for the PTQ numerics-parity benchmark.

The actual benchmark runs on Modal L4 (Iteration 4 smoke).  This
unit test verifies the local script runs end-to-end on CPU and
produces a JSON summary with the expected fields.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "smoke" / "run_ptq_parity.py"


def test_ptq_parity_local_runs_and_writes_summary(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(tmp_path),
        ],
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    # The script writes 'ptq_parity.json' under --output.
    summary_path = tmp_path / "ptq_parity.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert "abs_diff_max" in data
    assert "abs_diff_mean" in data
    assert "argmax_agree" in data
    assert "size_mb_fp32" in data
    assert "size_mb_int8_labelled_fp32" in data
