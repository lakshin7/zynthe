"""Behavior tests for :class:`PreflightAnalyzer` and
:class:`ResourceProbe`.

Pins the pure-Python memory-math helpers — they don't need a GPU so
they run cleanly under CPU-only pytest and on Modal L4.
"""

from __future__ import annotations

import pytest

from zynthe.core.preflight.resource_probe import ResourceProbe


# ----------------------------------------------------------------------------
# ResourceProbe public API (no torch-touching paths exercised).
# ----------------------------------------------------------------------------


def test_resource_probe_can_construct_and_probe() -> None:
    """Smoke test: probe() returns a dictionary with the expected keys."""
    probe = ResourceProbe()
    profile = probe.probe()
    assert "system" in profile
    assert "devices" in profile
    assert "memory" in profile
    assert "precision" in profile
    assert "compute" in profile
    assert "recommendations" in profile


def test_resource_probe_caches_results() -> None:
    """``probe()`` stores results on the instance (avoids re-probing)."""
    probe = ResourceProbe()
    profile = probe.probe()
    assert probe.probe_results == profile


# ----------------------------------------------------------------------------
# Memory estimation math
# ----------------------------------------------------------------------------


def test_estimate_memory_usage_fp32_matches_model_plus_optimizer_plus_grad() -> None:
    """For a 100M-param fp32 model the source uses ``bytes/(1024**3)``
    to convert to GB:

    * weights = 100M * 4 bytes = 400e6 bytes → ~0.3725 GB (binary GB)
    * optimizer = 2x weights
    * grads = 1x weights
    * activations ≈ batch_size * 1000 * 4 bytes ≈ 32 KB
    * total = sum * 1.2

    Verifies model / optimizer / gradient split and the *1.2 buffer.
    """
    probe = ResourceProbe()
    est = probe.estimate_memory_usage(model_params=100_000_000, batch_size=8)
    expected_model = 100_000_000 * 4 / (1024**3)
    assert est["model"] == pytest.approx(expected_model, rel=1e-9)
    assert est["optimizer"] == pytest.approx(2 * expected_model, rel=1e-9)
    assert est["gradients"] == pytest.approx(expected_model, rel=1e-9)
    raw = est["model"] + est["optimizer"] + est["gradients"] + est["activations"]
    assert est["total"] == pytest.approx(raw * 1.2, rel=1e-9)


def test_estimate_memory_usage_halves_at_fp16() -> None:
    """fp16 uses 2 bytes per param vs fp32's 4 → model weights half."""
    probe = ResourceProbe()
    fp32 = probe.estimate_memory_usage(model_params=80_000_000, batch_size=4, precision="fp32")
    fp16 = probe.estimate_memory_usage(model_params=80_000_000, batch_size=4, precision="fp16")
    # Half + tiny floating-point error tolerance.
    assert fp16["model"] == pytest.approx(fp32["model"] / 2.0, rel=1e-6)


def test_estimate_memory_usage_quarter_at_int8() -> None:
    """int8 = 1 byte per param → model weights a quarter of fp32's."""
    probe = ResourceProbe()
    fp32 = probe.estimate_memory_usage(model_params=80_000_000, batch_size=4, precision="fp32")
    int8 = probe.estimate_memory_usage(model_params=80_000_000, batch_size=4, precision="int8")
    assert int8["model"] == pytest.approx(fp32["model"] / 4.0, rel=1e-6)


def test_estimate_memory_usage_unknown_precision_falls_back_to_fp32() -> None:
    """Unknown precision tags default to 4 bytes per param (fp32)."""
    probe = ResourceProbe()
    fp32 = probe.estimate_memory_usage(model_params=10_000_000, batch_size=2, precision="fp32")
    weird = probe.estimate_memory_usage(
        model_params=10_000_000, batch_size=2, precision="fp99"
    )
    assert weird == fp32
