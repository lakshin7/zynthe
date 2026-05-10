from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from zynthe.evaluation.evaluation_report import EvaluationReport
import logging

logger = logging.getLogger(__name__)

def plot_teacher_student_comparison(
    student_train_losses: List[float],
    student_val_losses: List[float],
    teacher_train_losses: List[float],
    teacher_val_losses: List[float],
    student_metrics: Dict[str, List[float]],
    teacher_metrics: Dict[str, List[float]],
    save_path: str,
):
    """Plot side-by-side comparison of teacher vs student losses & accuracy.

    Produces a 2-row figure:
      Row 1: Train/Val loss curves for teacher and student
      Row 2: Validation accuracy (and optionally F1) comparison

    Args:
        student_train_losses: Student epoch train losses
        student_val_losses: Student epoch validation losses
        teacher_train_losses: Teacher epoch train losses
        teacher_val_losses: Teacher epoch validation losses
        student_metrics: Student metrics history dict (accuracy, f1, ...)
        teacher_metrics: Teacher metrics history dict (accuracy, f1, ...)
        save_path: Output file path
    """
    if not (student_train_losses or teacher_train_losses):
        logger.info("[PLOT] No teacher/student loss data to compare.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax_loss, ax_acc = axes

    # --- Loss comparison ---
    ax_loss.set_title("Teacher vs Student Loss Curves", fontsize=14, fontweight="bold")
    if teacher_train_losses:
        ax_loss.plot(range(1, len(teacher_train_losses)+1), teacher_train_losses, label="Teacher Train", color="#9467bd", marker="o")
    if teacher_val_losses:
        ax_loss.plot(range(1, len(teacher_val_losses)+1), teacher_val_losses, label="Teacher Val", color="#c5b0d5", marker="s")
    if student_train_losses:
        ax_loss.plot(range(1, len(student_train_losses)+1), student_train_losses, label="Student Train", color="#1f77b4", marker="o")
    if student_val_losses:
        ax_loss.plot(range(1, len(student_val_losses)+1), student_val_losses, label="Student Val", color="#ff7f0e", marker="s")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="best", fontsize=10, ncol=2)

    # --- Accuracy / F1 comparison ---
    ax_acc.set_title("Teacher vs Student Validation Metrics", fontsize=14, fontweight="bold")
    teacher_acc = teacher_metrics.get('accuracy', []) if isinstance(teacher_metrics, dict) else []
    student_acc = student_metrics.get('accuracy', []) if isinstance(student_metrics, dict) else []
    teacher_f1 = teacher_metrics.get('f1', []) if isinstance(teacher_metrics, dict) else []
    student_f1 = student_metrics.get('f1', []) if isinstance(student_metrics, dict) else []

    plotted_any = False
    if teacher_acc:
        ax_acc.plot(range(1, len(teacher_acc)+1), teacher_acc, label="Teacher Acc", color="#2ca02c", marker="o")
        plotted_any = True
    if student_acc:
        ax_acc.plot(range(1, len(student_acc)+1), student_acc, label="Student Acc", color="#17becf", marker="o")
        plotted_any = True
    if teacher_f1:
        ax_acc.plot(range(1, len(teacher_f1)+1), teacher_f1, label="Teacher F1", color="#8c564b", linestyle="--", marker="s")
        plotted_any = True
    if student_f1:
        ax_acc.plot(range(1, len(student_f1)+1), student_f1, label="Student F1", color="#e377c2", linestyle="--", marker="s")
        plotted_any = True

    if plotted_any:
        ax_acc.set_xlabel("Epoch", fontsize=12)
        ax_acc.set_ylabel("Metric Value", fontsize=12)
        ax_acc.set_ylim(0, 1.05)
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(loc="best", fontsize=10, ncol=2)
    else:
        ax_acc.text(0.5, 0.5, "No metric data available", ha='center', va='center', fontsize=12)
        ax_acc.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[PLOT] Teacher vs Student comparison saved to: {save_path}")
def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Dict[str, List[float]],
    save_path: str,
    lr_history: Optional[List[float]] = None,
    batch_train_losses: Optional[List[List[float]]] = None,
    batch_val_losses: Optional[List[List[float]]] = None,
    batch_val_running_acc: Optional[List[List[float]]] = None,
    highlight_extrema: bool = True,
    show_bands: bool = True,
    annotate: bool = True,
):
    """
    Enhanced training visualization with optional batch-level overlays and LR curve.

    Args:
        train_losses: List of training losses (per epoch)
        val_losses: List of validation losses (per epoch)
        metrics: Dict of metric name -> list of values (per epoch)
        save_path: Path to save the plot
        lr_history: Optional list of learning rates per epoch
        batch_train_losses: Optional per-epoch lists of per-batch training losses
        batch_val_losses: Optional per-epoch lists of per-batch validation losses
        batch_val_running_acc: Optional per-epoch lists of running validation accuracy per batch
        highlight_extrema: Mark best/worst points on plots
        show_bands: Show cumulative-best shaded envelopes for losses
        annotate: Add text annotations near extrema
    """
    if not train_losses and not val_losses and not metrics:
        logger.info("No data to plot.")
        return

    # Determine number of subplots needed
    num_plots = 2  # Loss + Metrics
    if lr_history:
        num_plots = 3  # Loss + Metrics + LR

    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    if num_plots == 2:
        # Ensure axs is indexable consistently
        axs = [axs[0], axs[1]]

    plot_idx = 0

    # ========== PLOT 1: LOSS CURVES WITH OPTIONAL BANDS & ANNOTATIONS ==========
    ax_loss = axs[plot_idx]
    title = "Training & Validation Loss"
    if batch_train_losses or batch_val_losses or batch_val_running_acc:
        title += " (Epoch + Batch Detail)"
    ax_loss.set_title(title, fontsize=14, fontweight="bold")

    epochs_train = list(range(1, len(train_losses) + 1)) if train_losses else []
    epochs_val = list(range(1, len(val_losses) + 1)) if val_losses else []

    # Epoch-level curves
    if train_losses:
        ax_loss.plot(
            epochs_train,
            train_losses,
            label="Train Loss",
            marker="o",
            linewidth=2,
            markersize=6,
            color="#1f77b4",
        )
    if val_losses:
        ax_loss.plot(
            epochs_val,
            val_losses,
            label="Validation Loss",
            marker="s",
            linewidth=2,
            markersize=6,
            color="#ff7f0e",
        )

    # Shaded cumulative-best envelopes
    if show_bands and len(train_losses) > 1:
        t_arr = np.array(train_losses)
        ax_loss.fill_between(
            epochs_train,
            t_arr,
            np.minimum.accumulate(t_arr),
            color="#1f77b4",
            alpha=0.08,
            label="Train (cumulative best)",
        )
    if show_bands and len(val_losses) > 1:
        v_arr = np.array(val_losses)
        ax_loss.fill_between(
            epochs_val,
            v_arr,
            np.minimum.accumulate(v_arr),
            color="#ff7f0e",
            alpha=0.08,
            label="Val (cumulative best)",
        )

    # Optional: batch-level overlays as thin, semi-transparent lines
    def _plot_batch_series(series_by_epoch, base_color, label_prefix):
        if not series_by_epoch:
            return
        for e_idx, series in enumerate(series_by_epoch, start=1):
            if not series:
                continue
            x = np.linspace(e_idx - 0.45, e_idx + 0.45, num=len(series))
            ax_loss.plot(
                x,
                series,
                color=base_color,
                alpha=0.25,
                linewidth=1.0,
                label=(f"{label_prefix} (batch)" if e_idx == 1 else None),
            )

    _plot_batch_series(batch_train_losses, "#1f77b4", "Train")
    _plot_batch_series(batch_val_losses, "#ff7f0e", "Val")

    # Optional: running validation accuracy (batch) overlaid on secondary axis
    if batch_val_running_acc:
        ax2 = ax_loss.twinx()
        for e_idx, series in enumerate(batch_val_running_acc, start=1):
            if not series:
                continue
            x = np.linspace(e_idx - 0.45, e_idx + 0.45, num=len(series))
            ax2.plot(
                x,
                series,
                color="#2ca02c",
                alpha=0.3,
                linewidth=1.0,
                label=("Val Running Acc (batch)" if e_idx == 1 else None),
            )
        ax2.set_ylabel("Running Acc (val)", fontsize=11, color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # Extrema markers and annotations
    if highlight_extrema:
        if train_losses:
            t_min_epoch = int(np.argmin(train_losses)) + 1
            t_min_val = float(np.min(train_losses))
            ax_loss.scatter(
                [t_min_epoch],
                [t_min_val],
                color="#1f77b4",
                s=80,
                edgecolors="black",
                zorder=5,
            )
            if annotate:
                ax_loss.annotate(
                    f"best train\n{t_min_val:.4f}",
                    (t_min_epoch, t_min_val),
                    textcoords="offset points",
                    xytext=(0, -30),
                    ha="center",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="#1f77b4",
                        alpha=0.7,
                    ),
                )
        if val_losses:
            v_min_epoch = int(np.argmin(val_losses)) + 1
            v_min_val = float(np.min(val_losses))
            ax_loss.scatter(
                [v_min_epoch],
                [v_min_val],
                color="#ff7f0e",
                s=80,
                edgecolors="black",
                zorder=5,
            )
            if annotate:
                ax_loss.annotate(
                    f"best val\n{v_min_val:.4f}",
                    (v_min_epoch, v_min_val),
                    textcoords="offset points",
                    xytext=(0, -30),
                    ha="center",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="#ff7f0e",
                        alpha=0.7,
                    ),
                )

        if len(train_losses) > 1:
            t_max_epoch = int(np.argmax(train_losses)) + 1
            t_max_val = float(np.max(train_losses))
            ax_loss.scatter(
                [t_max_epoch], [t_max_val], color="#1f77b4", s=60, marker="X", zorder=5
            )
            if annotate:
                ax_loss.annotate(
                    f"worst train\n{t_max_val:.4f}",
                    (t_max_epoch, t_max_val),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )
        if len(val_losses) > 1:
            v_max_epoch = int(np.argmax(val_losses)) + 1
            v_max_val = float(np.max(val_losses))
            ax_loss.scatter(
                [v_max_epoch], [v_max_val], color="#ff7f0e", s=60, marker="X", zorder=5
            )
            if annotate:
                ax_loss.annotate(
                    f"worst val\n{v_max_val:.4f}",
                    (v_max_epoch, v_max_val),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )

    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.legend(loc="best", fontsize=10)
    ax_loss.grid(True, alpha=0.3)
    plot_idx += 1

    # ========== PLOT 2: METRICS CURVES WITH ENHANCED SEPARATION ==========
    axs[plot_idx].set_title("Validation Metrics over Epochs", fontsize=14, fontweight="bold")

    # Use different line styles and markers to differentiate overlapping metrics
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    # Use a qualitative colormap; fall back to 'viridis' if tab10 unavailable
    try:
        cmap = plt.get_cmap("tab10")
    except Exception:
        cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, max(len(metrics), 1))]

    for idx, (metric_name, values) in enumerate(metrics.items()):
        if not values:
            continue
        style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        x_epochs = range(1, len(values) + 1)
        axs[plot_idx].plot(
            x_epochs,
            values,
            label=metric_name.capitalize(),
            linestyle=style,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.85,
        )
        if highlight_extrema and len(values) >= 1:
            best_epoch = int(np.argmax(values)) + 1
            best_val = float(np.max(values))
            axs[plot_idx].scatter(
                [best_epoch], [best_val], color=color, s=70, edgecolors="black", zorder=5
            )
            if annotate:
                axs[plot_idx].annotate(
                    f"best {metric_name}\n{best_val:.4f}",
                    (best_epoch, best_val),
                    textcoords="offset points",
                    xytext=(0, -28),
                    ha="center",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.65
                    ),
                )
        if highlight_extrema and len(values) > 1:
            worst_epoch = int(np.argmin(values)) + 1
            worst_val = float(np.min(values))
            axs[plot_idx].scatter(
                [worst_epoch], [worst_val], color=color, s=50, marker="X", zorder=5
            )
            if annotate:
                axs[plot_idx].annotate(
                    f"worst {metric_name}\n{worst_val:.4f}",
                    (worst_epoch, worst_val),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                )

    axs[plot_idx].set_xlabel("Epoch", fontsize=12)
    axs[plot_idx].set_ylabel("Metric Value", fontsize=12)
    axs[plot_idx].set_ylim([0, 1.05])  # Metrics are typically 0-1
    axs[plot_idx].legend(loc="best", fontsize=10, ncol=2)
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # ========== PLOT 3: LEARNING RATE CURVE (OPTIONAL) ==========
    if lr_history:
        axs[plot_idx].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        axs[plot_idx].plot(
            range(1, len(lr_history) + 1),
            lr_history,
            label="Learning Rate",
            marker="o",
            linewidth=2,
            markersize=6,
            color="red",
        )
        axs[plot_idx].set_xlabel("Epoch", fontsize=12)
        axs[plot_idx].set_ylabel("Learning Rate", fontsize=12)
        axs[plot_idx].set_yscale("log")  # Log scale for better LR visualization
        axs[plot_idx].legend(loc="best", fontsize=10)
        axs[plot_idx].grid(True, alpha=0.3, which="both")

    plt.tight_layout(pad=2.0)  # Add padding between subplots
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[PLOT] Enhanced training curves saved to: {save_path}")
def plot_epoch_micro_series(
    title_prefix: str,
    epoch_idx: int,
    train_batch_losses: Optional[List[float]],
    val_batch_losses: Optional[List[float]],
    val_running_acc: Optional[List[float]],
    save_path: str,
):
    """
    Plot micro-series for a single epoch: per-batch train/val losses and running validation accuracy.

    Args:
        title_prefix: Label prefix like "Student" or "Teacher"
        epoch_idx: 1-based epoch index
        train_batch_losses: List of per-batch training losses for this epoch
        val_batch_losses: List of per-batch validation losses for this epoch
        val_running_acc: List of running accuracy values computed over validation batches
        save_path: Path to save the figure
    """
    # Normalize inputs
    train_series = train_batch_losses or []
    val_series = val_batch_losses or []
    acc_series = val_running_acc or []

    if not train_series and not val_series and not acc_series:
        logger.info("[PLOT] No micro-series data to plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_title(f"{title_prefix} Epoch {epoch_idx} Micro-series", fontsize=14, fontweight="bold")

    # Plot per-batch losses
    if train_series:
        ax.plot(
            np.arange(1, len(train_series) + 1),
            train_series,
            color="#1f77b4",
            alpha=0.8,
            linewidth=1.5,
            label="Train Loss (batch)",
        )
    if val_series:
        ax.plot(
            np.arange(1, len(val_series) + 1),
            val_series,
            color="#ff7f0e",
            alpha=0.8,
            linewidth=1.5,
            label="Val Loss (batch)",
        )

    ax.set_xlabel("Batch Index", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Secondary axis for running accuracy
    if acc_series:
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(1, len(acc_series) + 1),
            acc_series,
            color="#2ca02c",
            alpha=0.6,
            linewidth=1.2,
            label="Val Running Acc",
        )
        ax2.set_ylabel("Running Acc (val)", fontsize=12, color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        # Build combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=10)
    else:
        ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[PLOT] Micro-series saved to: {save_path}")
def plot_metric_grid(metrics: Dict[str, List[float]], save_path: str, columns: int = 2) -> None:
    """Plot each metric on its own axis to highlight subtle differences."""
    metric_series = [(name, values) for name, values in metrics.items() if isinstance(values, (list, tuple)) and values]
    if not metric_series:
        logger.info("[PLOT] No metric history provided for grid plot.")
        return

    rows = math.ceil(len(metric_series) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, values) in zip(axes, metric_series):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker='o', linewidth=2, color='#1f77b4')
        ax.set_title(name.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(metric_series):]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[PLOT] Metric grid saved to: {save_path}")
def plot_calibration_curve(calibration: Dict[str, Any], save_path: str) -> None:
    """Render a reliability diagram from calibration statistics."""
    if not calibration:
        logger.info("[PLOT] Calibration data missing; skipping reliability plot.")
        return
    prob_true = calibration.get('prob_true')
    prob_pred = calibration.get('prob_pred')
    if prob_true is None or prob_pred is None:
        logger.info("[PLOT] Calibration arrays not found; skipping reliability plot.")
        return

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color='#1f77b4')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Reliability Diagram', fontsize=13, fontweight='bold')

    brier = calibration.get('brier_score')
    if isinstance(brier, (int, float, np.floating)):
        plt.text(0.05, 0.9, f'Brier Score: {brier:.4f}', transform=plt.gca().transAxes)

    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[PLOT] Calibration curve saved to: {save_path}")
def plot_runtime_profile(runtime: Dict[str, Any], save_path: str) -> None:
    """Visualize latency distribution and throughput statistics."""
    if not runtime:
        logger.info("[PLOT] Runtime stats unavailable; skipping profile plot.")
        return

    numeric_items = {k: v for k, v in runtime.items() if isinstance(v, (int, float, np.floating)) and k not in {'batches', 'batches_completed'}}
    if not numeric_items:
        logger.info("[PLOT] Runtime stats lack numeric values; skipping profile plot.")
        return

    metrics_names = list(numeric_items.keys())
    values_array = np.asarray([numeric_items[name] for name in metrics_names], dtype=float)

    plt.figure(figsize=(max(6, len(metrics_names) * 1.2), 4))
    bars = plt.bar(metrics_names, values_array, color='#1f77b4', alpha=0.8)
    plt.ylabel('Milliseconds / Throughput')
    plt.xticks(rotation=30, ha='right')
    plt.title('Runtime Profile', fontsize=13, fontweight='bold')
    for bar, value in zip(bars, values_array):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{float(value):.2f}', ha='center', va='bottom', fontsize=9)

    batches = runtime.get('batches') or runtime.get('batches_completed')
    if batches:
        plt.text(0.02, 0.92, f'Batches: {batches}', transform=plt.gca().transAxes)

    throughput = runtime.get('throughput_samples_per_s')
    if throughput:
        plt.text(0.02, 0.84, f'Throughput: {throughput:.2f} samples/s', transform=plt.gca().transAxes)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[PLOT] Runtime profile saved to: {save_path}")
def plot_evaluation_dashboard(report: "EvaluationReport", save_path: str):
    """Plot a comprehensive dashboard from an EvaluationReport."""
    metrics = getattr(report, 'metrics', {}) or getattr(report, 'core_metrics', {}) or {}
    dist_metrics = getattr(report, 'distillation_metrics', {}) or {}
    runtime = getattr(report, 'runtime', None) or {}
    calibration = getattr(report, 'calibration', None) or {}
    metadata = getattr(report, 'metadata', {}) or {}
    train_losses = metadata.get('train_losses', []) if isinstance(metadata, dict) else []
    val_losses = metadata.get('val_losses', []) if isinstance(metadata, dict) else []
    metric_history = metadata.get('metrics_history', {}) if isinstance(metadata, dict) else {}

    panel_specs = ['core_metrics']
    if isinstance(dist_metrics, dict) and dist_metrics:
        panel_specs.append('distillation')
    if isinstance(runtime, dict) and runtime:
        panel_specs.append('runtime')
    if isinstance(calibration, dict) and calibration.get('prob_true') is not None and calibration.get('prob_pred') is not None:
        panel_specs.append('calibration')
    if train_losses or val_losses:
        panel_specs.append('training_curves')
    if isinstance(metric_history, dict) and any(isinstance(v, (list, tuple)) and v for v in metric_history.values()):
        panel_specs.append('metric_grid')

    cols = 2
    rows = max(1, math.ceil(len(panel_specs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(
        f"Evaluation Dashboard: {getattr(report, 'model_name', 'model')} ({getattr(report, 'modality', 'text').capitalize()})",
        fontsize=16,
        fontweight='bold',
    )

    for ax, panel in zip(axes, panel_specs):
        if panel == 'core_metrics':
            numeric = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating))
            }
            if not numeric:
                ax.text(0.5, 0.5, 'No core metrics', ha='center', va='center')
                ax.axis('off')
                continue
            names = list(numeric.keys())
            values = [float(numeric[name]) for name in names]
            ax.bar(names, values, color='#1f77b4')
            ax.set_title('Core Metrics')
            ax.tick_params(axis='x', rotation=30)
            ax.grid(axis='y', alpha=0.3)
        elif panel == 'distillation':
            numeric = {
                k: v for k, v in dist_metrics.items()
                if isinstance(v, (int, float, np.floating))
            }
            if not numeric:
                ax.text(0.5, 0.5, 'No distillation metrics', ha='center', va='center')
                ax.axis('off')
                continue
            ax.barh(list(numeric.keys()), [float(v) for v in numeric.values()], color='#ff7f0e')
            ax.set_title('Distillation Metrics')
            ax.grid(axis='x', alpha=0.3)
        elif panel == 'runtime':
            numeric = {
                k: v for k, v in runtime.items()
                if isinstance(v, (int, float, np.floating)) and k not in {'batches', 'batches_completed'}
            }
            if not numeric:
                ax.text(0.5, 0.5, 'No runtime metrics', ha='center', va='center')
                ax.axis('off')
                continue
            ax.bar(list(numeric.keys()), [float(v) for v in numeric.values()], color='#17becf')
            ax.set_title('Runtime Profile')
            ax.tick_params(axis='x', rotation=30)
            ax.grid(axis='y', alpha=0.3)
        elif panel == 'calibration':
            prob_true = np.asarray(calibration.get('prob_true', []), dtype=float)
            prob_pred = np.asarray(calibration.get('prob_pred', []), dtype=float)
            if prob_true.size == 0 or prob_pred.size == 0:
                ax.text(0.5, 0.5, 'No calibration data', ha='center', va='center')
                ax.axis('off')
                continue
            ax.plot(prob_pred, prob_true, marker='o', label='Model')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Observed Frequency')
            ax.set_title('Calibration')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
        elif panel == 'training_curves':
            if train_losses:
                ax.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
            if val_losses:
                ax.plot(range(1, len(val_losses) + 1), val_losses, marker='s', label='Val Loss')
            ax.set_title('Training Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
        elif panel == 'metric_grid':
            plotted = False
            for key, values in metric_history.items():
                if isinstance(values, (list, tuple)) and values:
                    ax.plot(range(1, len(values) + 1), values, label=str(key))
                    plotted = True
            if not plotted:
                ax.text(0.5, 0.5, 'No metric history', ha='center', va='center')
                ax.axis('off')
                continue
            ax.set_title('Metric History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)

    for ax in axes[len(panel_specs):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[PLOT] Evaluation dashboard saved to: {save_path}")
def plot_distillation_gap(teacher_metrics: Dict, student_metrics: Dict, save_path: str):
    """Plot the gap between teacher and student performance."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    common_keys = set(teacher_metrics.keys()).intersection(set(student_metrics.keys()))
    plot_keys = [k for k in common_keys if isinstance(teacher_metrics[k], (int, float))]
    
    if not plot_keys:
        return
        
    teacher_vals = [teacher_metrics[k] for k in plot_keys]
    student_vals = [student_metrics[k] for k in plot_keys]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_keys))
    width = 0.35
    
    ax.bar(x - width/2, teacher_vals, width, label='Teacher', color='#2E86AB')
    ax.bar(x + width/2, student_vals, width, label='Student', color='#A23B72')
    
    ax.set_ylabel('Score')
    ax.set_title('Teacher vs Student Performance Gap')
    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in plot_keys])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add gap labels
    for i in range(len(plot_keys)):
        gap = teacher_vals[i] - student_vals[i]
        ax.annotate(f"-{gap:.3f}", 
                    xy=(i, max(teacher_vals[i], student_vals[i])),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, color='red')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[PLOT] Distillation gap saved to: {save_path}")
def plot_extended_metrics(report: "EvaluationReport", save_path: str):
    """Plot extended/custom distillation metrics."""
    metrics = getattr(report, 'distillation_metrics', None)
    if not isinstance(metrics, dict) or not metrics:
        logger.info("[PLOT] No extended metrics available; skipping extended metrics plot.")
        return

    scalar_metrics = {
        key: value for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating))
    }
    series_metrics = {
        key: value for key, value in metrics.items()
        if isinstance(value, (list, tuple)) and value
    }

    if not scalar_metrics and not series_metrics:
        logger.info("[PLOT] Extended metrics contain no plottable values; skipping.")
        return

    fig, axes = plt.subplots(1, 2 if scalar_metrics and series_metrics else 1, figsize=(12, 4.5))
    axes = np.atleast_1d(axes)
    idx = 0

    if scalar_metrics:
        ax = axes[idx]
        idx += 1
        names = list(scalar_metrics.keys())
        vals = [float(scalar_metrics[name]) for name in names]
        ax.bar(names, vals, color='#9467bd')
        ax.set_title('Extended Scalars')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)

    if series_metrics:
        ax = axes[idx]
        for key, values in series_metrics.items():
            ax.plot(range(1, len(values) + 1), values, marker='o', label=str(key))
        ax.set_title('Extended Trends')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[PLOT] Extended metrics saved to: {save_path}")