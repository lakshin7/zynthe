from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def build_eval_diagnostics(
    metrics: Mapping[str, Any],
    avg_loss: Optional[float],
    batch_losses: Sequence[float],
    confidence_samples: Sequence[float],
    probabilities: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Compute a compact set of evaluation diagnostics and warning flags."""
    diagnostics: Dict[str, Any] = {}
    warnings = []

    if metrics:
        core_metrics = [
            metrics.get("accuracy"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1"),
        ]
        numeric_vals = [
            float(val) for val in core_metrics if isinstance(val, (int, float, np.floating))
        ]
        if numeric_vals:
            spread = max(numeric_vals) - min(numeric_vals)
            diagnostics["core_metric_spread"] = float(spread)
            if spread < 0.01:
                warnings.append("metrics_overlap")

    if avg_loss is not None:
        diagnostics["eval_loss"] = float(avg_loss)

    if batch_losses:
        losses = np.asarray(batch_losses, dtype=np.float64)
        diagnostics["loss_std"] = float(losses.std())
        diagnostics["loss_min"] = float(losses.min())
        diagnostics["loss_max"] = float(losses.max())

    if confidence_samples:
        conf_arr = np.asarray(confidence_samples, dtype=np.float64)
        diagnostics["confidence_mean"] = float(conf_arr.mean())
        diagnostics["confidence_std"] = float(conf_arr.std())
        diagnostics["confidence_min"] = float(conf_arr.min())
        diagnostics["confidence_max"] = float(conf_arr.max())
        if conf_arr.mean() > 0.95 and conf_arr.std() < 0.02:
            warnings.append("confidence_collapse")

    if probabilities is not None and probabilities.ndim >= 1:
        probs = probabilities.reshape(-1, 1) if probabilities.ndim == 1 else probabilities
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
        diagnostics["prediction_entropy_mean"] = float(np.mean(entropy))
        diagnostics["prediction_entropy_std"] = float(np.std(entropy))
        if np.mean(entropy) < 0.2:
            warnings.append("low_entropy_predictions")

    diagnostics["warnings"] = warnings
    return diagnostics
