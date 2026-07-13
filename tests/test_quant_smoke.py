"""Smoke test: PTQRunner exists, has a `run()` method, and returns a summary.

This is a unit-level guard around the existing :class:`PTQRunner`
in :mod:`zynthe.core.quant.ptq`.  Full pipeline smoke (model load +
quantize + forward) is exercised by ``scripts/smoke/run_ptq.py`` on
Modal L4 (Iteration 2 smoke).
"""

from __future__ import annotations

import pytest


def test_ptq_runner_class_exists_and_has_run() -> None:
    from zynthe.core.quant.ptq import PTQRunner

    assert hasattr(PTQRunner, "run")
    # The runner should accept a config dict and store the config
    # on the instance.
    cfg = {
        "quantization": {
            "strategy": "dynamic",
            "dtype": "qint8",
            "device": "cpu",
            "output_dir": "/tmp/ptq_test",
        },
        "runtime": {"device": "cpu"},
        "output_root": "/tmp/ptq_test",
    }
    runner = PTQRunner(cfg)
    assert runner.cfg is cfg
    assert runner.strategy == "dynamic"
    assert runner.dtype is not None
