"""
Evaluation Report Module
Defines the standard EvaluationReport dataclass for modality-agnostic evaluation results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationReport:
    """
    Canonical modality-agnostic evaluation report.

    Canonical fields follow the Phase-2 plan:
    - loss, metrics, diagnostics, runtime, calibration, explainability
    - modality, timestamp

    Additional metadata fields are retained for backward compatibility with
    existing trainer and visualization paths.
    """

    # Canonical fields
    loss: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    runtime: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    explainability: Optional[Dict[str, Any]] = None
    modality: str = "text"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Compatibility fields
    model_name: str = "model"
    task_type: str = "classification"
    per_class_metrics: Optional[Dict[str, List[float]]] = None
    distillation_metrics: Optional[Dict[str, Any]] = None
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    predictions_path: Optional[str] = None
    labels_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def core_metrics(self) -> Dict[str, Any]:
        """Backward-compatible alias for the canonical `metrics` field."""
        return self.metrics

    @core_metrics.setter
    def core_metrics(self, value: Dict[str, Any]) -> None:
        self.metrics = dict(value or {})

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dict with canonical and compatibility keys."""
        return {
            "loss": self.loss,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "runtime": self.runtime,
            "calibration": self.calibration,
            "explainability": self.explainability,
            "modality": self.modality,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "per_class_metrics": self.per_class_metrics,
            "distillation_metrics": self.distillation_metrics,
            "artifact_paths": self.artifact_paths,
            "predictions_path": self.predictions_path,
            "labels_path": self.labels_path,
            "metadata": self.metadata,
            # Compatibility alias
            "core_metrics": self.metrics,
        }

    def save_json(self, filepath: str | Path) -> None:
        """Serialize the report to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        class _NumpySafeEncoder(json.JSONEncoder):
            """Handle numpy types that json.dump can't serialize."""

            def default(self, obj):
                try:
                    import numpy as np

                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                except ImportError:
                    pass
                return super().default(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, cls=_NumpySafeEncoder)

    def save_markdown(self, filepath: str | Path) -> None:
        """Persist a lightweight markdown summary for quick artifact inspection."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "# Evaluation Report",
            "",
            f"- Model: {self.model_name}",
            f"- Modality: {self.modality}",
            f"- Task Type: {self.task_type}",
            f"- Timestamp: {self.timestamp}",
            "",
            "## Core Metrics",
        ]
        if self.metrics:
            for key, value in self.metrics.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Diagnostics")
        if self.diagnostics:
            for key, value in self.diagnostics.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- None")

        if self.distillation_metrics:
            lines.append("")
            lines.append("## Distillation Metrics")
            for key, value in self.distillation_metrics.items():
                lines.append(f"- {key}: {value}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @classmethod
    def load_json(cls, filepath: str | Path) -> "EvaluationReport":
        """Deserialize from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Backward compatibility for older report schema.
        if "metrics" not in data and "core_metrics" in data:
            data["metrics"] = data.pop("core_metrics")
        if "loss" not in data and isinstance(data.get("metrics"), dict):
            maybe_loss = data["metrics"].get("loss")
            if isinstance(maybe_loss, (int, float)):
                data["loss"] = float(maybe_loss)
        data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        # Strip synthetic keys that to_dict() adds but aren't constructor args.
        data.pop("core_metrics", None)
        return cls(**data)
