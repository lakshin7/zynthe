from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from evaluation.evaluation_report import EvaluationReport
from evaluation.visualizer import plot_evaluation_dashboard, plot_extended_metrics


def test_evaluation_report_round_trip(tmp_path: Path) -> None:
    report = EvaluationReport(
        loss=0.42,
        metrics={"accuracy": 0.91, "f1": 0.89},
        diagnostics={"samples": 12},
        runtime={"seconds": 3.2},
        calibration={"prob_true": [0.1, 0.8], "prob_pred": [0.2, 0.7]},
        explainability={"tokens": ["good"]},
        modality="text",
    )
    report.distillation_metrics = {"kd_loss": 0.11, "agreement": [0.3, 0.5, 0.7]}
    report.metadata = {
        "train_losses": [1.0, 0.7, 0.4],
        "val_losses": [1.2, 0.8, 0.5],
        "metrics_history": {"accuracy": [0.7, 0.8, 0.9]},
    }

    json_path = tmp_path / "evaluation_report.json"
    md_path = tmp_path / "evaluation_report.md"
    report.save_json(json_path)
    report.save_markdown(md_path)

    loaded = EvaluationReport.load_json(json_path)

    assert loaded.loss == report.loss
    assert loaded.metrics["accuracy"] == 0.91
    assert loaded.diagnostics["samples"] == 12
    assert loaded.modality == "text"
    assert json_path.exists()
    assert md_path.exists()
    assert "accuracy" in md_path.read_text(encoding="utf-8")


def test_dashboard_handles_empty_and_partial_reports(tmp_path: Path) -> None:
    empty_report = EvaluationReport()
    empty_path = tmp_path / "dashboard_empty.png"
    plot_evaluation_dashboard(empty_report, str(empty_path))
    assert empty_path.exists()

    partial_report = EvaluationReport(
        metrics={"accuracy": 0.75},
        distillation_metrics={"kd_loss": 0.2, "agreement": [0.1, 0.2, 0.3]},
        modality="vision",
    )
    partial_report.metadata = {
        "train_losses": [0.9, 0.6],
        "val_losses": [1.1, 0.7],
        "metrics_history": {"accuracy": [0.65, 0.75]},
    }

    dashboard_path = tmp_path / "dashboard_partial.png"
    extended_path = tmp_path / "extended_partial.png"
    plot_evaluation_dashboard(partial_report, str(dashboard_path))
    plot_extended_metrics(partial_report, str(extended_path))

    assert dashboard_path.exists()
    assert extended_path.exists()
