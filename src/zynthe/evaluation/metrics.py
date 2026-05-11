from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:  # Added guard for older scikit-learn versions
    from sklearn.metrics import top_k_accuracy_score  # type: ignore
except ImportError:  # pragma: no cover - fallback for older sklearn
    top_k_accuracy_score = None


def _to_numpy(arr):
    if arr is None:
        return np.array([])
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_accuracy(preds, labels):
    return accuracy_score(_to_numpy(labels), _to_numpy(preds))


def compute_f1(preds, labels):
    return f1_score(_to_numpy(labels), _to_numpy(preds), average="macro")


def compute_precision(preds, labels):
    return precision_score(_to_numpy(labels), _to_numpy(preds), average="macro", zero_division=0)


def compute_recall(preds, labels):
    return recall_score(_to_numpy(labels), _to_numpy(preds), average="macro", zero_division=0)


def compute_confusion_matrix(preds, labels):
    return confusion_matrix(_to_numpy(labels), _to_numpy(preds))


def compute_balanced_accuracy(preds, labels):
    return balanced_accuracy_score(_to_numpy(labels), _to_numpy(preds))


def compute_mcc(preds, labels):
    try:
        return matthews_corrcoef(_to_numpy(labels), _to_numpy(preds))
    except ValueError:
        return 0.0


def compute_cohen_kappa(preds, labels):
    return cohen_kappa_score(_to_numpy(labels), _to_numpy(preds))


def compute_classwise_metrics(preds, labels):
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    if len(labels_np) == 0:
        return {"precision_per_class": {}, "recall_per_class": {}, "f1_per_class": {}}
    precision = precision_score(labels_np, preds_np, average=None, zero_division=0)
    recall = recall_score(labels_np, preds_np, average=None, zero_division=0)
    f1 = f1_score(labels_np, preds_np, average=None)
    classes = np.unique(np.concatenate((labels_np, preds_np)))

    # Handle scalar results (single class case)
    precision = np.atleast_1d(precision)
    recall = np.atleast_1d(recall)
    f1 = np.atleast_1d(f1)

    precision_dict = {int(cls): float(p) for cls, p in zip(classes, precision)}
    recall_dict = {int(cls): float(r) for cls, r in zip(classes, recall)}
    f1_dict = {int(cls): float(f) for cls, f in zip(classes, f1)}
    return {
        "precision_per_class": precision_dict,
        "recall_per_class": recall_dict,
        "f1_per_class": f1_dict,
    }


def compute_roc_auc(pred_probs, labels):
    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(pred_probs)
    if len(labels_np) == 0 or len(preds_np) == 0:
        return None
    unique_labels = np.unique(labels_np)
    if len(unique_labels) == 2:
        if preds_np.ndim == 2 and preds_np.shape[1] == 2:
            preds_pos = preds_np[:, 1]
        else:
            preds_pos = preds_np
        try:
            return roc_auc_score(labels_np, preds_pos)
        except ValueError:
            return None
    elif len(unique_labels) > 2:
        warnings.warn("ROC AUC is not defined for multi-class classification with this function.")
    return None


def compute_top_k_accuracy(pred_probs, labels, k: int = 3):
    if top_k_accuracy_score is None or pred_probs is None:
        return None
    preds_np = _to_numpy(pred_probs)
    if preds_np.ndim != 2 or preds_np.shape[1] < k:
        return None
    labels_np = _to_numpy(labels)
    try:
        return top_k_accuracy_score(labels_np, preds_np, k=k, labels=np.unique(labels_np))
    except Exception:
        return None


def compute_average_precision(pred_probs, labels):
    if pred_probs is None:
        return None
    preds_np = _to_numpy(pred_probs)
    labels_np = _to_numpy(labels)
    try:
        if preds_np.ndim == 1 or preds_np.shape[1] == 1:
            return average_precision_score(
                labels_np, preds_np if preds_np.ndim == 1 else preds_np[:, 0]
            )
        return average_precision_score(labels_np, preds_np, average="macro")
    except ValueError:
        return None


def compute_brier(pred_probs, labels):
    if pred_probs is None:
        return None
    preds_np = _to_numpy(pred_probs)
    labels_np = _to_numpy(labels)
    if preds_np.ndim == 2 and preds_np.shape[1] > 1:
        # Use probability assigned to true class
        probs_true = preds_np[np.arange(len(labels_np)), labels_np.astype(int)]
    else:
        probs_true = preds_np if preds_np.ndim == 1 else preds_np.reshape(-1)
    try:
        return brier_score_loss(labels_np, probs_true)
    except ValueError:
        return None


CalibrationPayload = Dict[str, Union[np.ndarray, float]]


def compute_calibration(pred_probs, labels, n_bins: int = 10) -> Optional[CalibrationPayload]:
    if pred_probs is None:
        return None
    preds_np = _to_numpy(pred_probs)
    labels_np = _to_numpy(labels)
    if preds_np.ndim == 2 and preds_np.shape[1] > 1:
        # Use probability for predicted class
        probs = preds_np.max(axis=1)
    else:
        probs = preds_np if preds_np.ndim == 1 else preds_np.reshape(-1)
    if probs.size == 0 or labels_np.size == 0:
        return None
    try:
        prob_true, prob_pred = calibration_curve(
            labels_np, probs, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        return None
    brier = compute_brier(probs, labels_np)
    payload: CalibrationPayload = {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }
    if brier is not None:
        payload["brier_score"] = float(brier)
    return payload


def compute_precision_recall_curve_data(pred_probs, labels):
    if pred_probs is None:
        return None
    preds_np = _to_numpy(pred_probs)
    labels_np = _to_numpy(labels)
    if preds_np.ndim == 2 and preds_np.shape[1] > 1:
        # Assume positive class is 1 (binary); for multiclass fallback to macro average by selecting predicted class
        if preds_np.shape[1] == 2:
            scores = preds_np[:, 1]
        else:
            return None
    else:
        scores = preds_np if preds_np.ndim == 1 else preds_np.reshape(-1)
    try:
        precision, recall, thresholds = precision_recall_curve(labels_np, scores)
    except ValueError:
        return None
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


def compute_all_metrics(preds, labels, pred_probs=None):
    if preds is None or labels is None:
        return {}
    metrics = {
        "accuracy": compute_accuracy(preds, labels),
        "f1": compute_f1(preds, labels),
        "precision": compute_precision(preds, labels),
        "recall": compute_recall(preds, labels),
        "confusion_matrix": compute_confusion_matrix(preds, labels),
        "balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "matthews_corrcoef": compute_mcc(preds, labels),
        "cohen_kappa": compute_cohen_kappa(preds, labels),
    }
    metrics.update(compute_classwise_metrics(preds, labels))
    if pred_probs is not None:
        roc_auc = compute_roc_auc(pred_probs, labels)
        if roc_auc is not None:
            metrics["roc_auc"] = roc_auc
            metrics["y_true"] = _to_numpy(labels)
            if pred_probs.ndim == 1:
                metrics["y_score"] = _to_numpy(pred_probs)
            else:
                metrics["y_score"] = pred_probs[:, 1]
        top3 = compute_top_k_accuracy(
            pred_probs, labels, k=min(3, pred_probs.shape[1] if pred_probs.ndim == 2 else 1)
        )
        if top3 is not None:
            metrics["top3_accuracy"] = top3
        avg_precision = compute_average_precision(pred_probs, labels)
        if avg_precision is not None:
            metrics["average_precision"] = avg_precision
        brier = compute_brier(pred_probs, labels)
        if brier is not None:
            metrics["brier_score"] = brier
        calibration = compute_calibration(pred_probs, labels)
        if calibration is not None:
            metrics["calibration"] = calibration
        pr_curve = compute_precision_recall_curve_data(pred_probs, labels)
        if pr_curve is not None:
            metrics["precision_recall_curve"] = pr_curve
    return metrics


def plot_metrics(metrics, save_dir, labels=None):
    # Only plot the last epoch metrics if metrics is a list or tuple
    if metrics is None or len(metrics) == 0:
        return
    if isinstance(metrics, (list, tuple)):
        metrics = metrics[-1]
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dict or list of dicts")

    if save_dir is None:
        return
    if isinstance(save_dir, (list, tuple)):
        save_dir = str(save_dir[0])
    else:
        save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Save numeric metrics to JSON file
    metrics_to_save = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            metrics_to_save[key] = float(value)
        elif isinstance(value, dict):
            # Save numeric values inside dicts (like classwise metrics)
            sub_dict = {}
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float, np.floating)):
                    sub_dict[sub_key] = float(sub_val)
            if sub_dict:
                metrics_to_save[key] = sub_dict
    if metrics_to_save:
        json_path = os.path.join(save_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_to_save, f, indent=4)

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        plt.figure(figsize=(8, 6))

        # Add class labels if not provided
        if labels is None:
            num_classes = cm.shape[0]
            labels = [f"Class {i}" for i in range(num_classes)]

        # Create heatmap with better formatting
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
        )

        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.title(
            "Confusion Matrix\n(Rows=Actual, Columns=Predicted)", fontsize=14, fontweight="bold"
        )

        # Add accuracy text
        accuracy = metrics.get("accuracy", 0)
        if accuracy > 0:
            plt.text(
                0.5,
                -0.15,
                f"Overall Accuracy: {accuracy:.2%}",
                transform=plt.gca().transAxes,
                ha="center",
                fontsize=11,
                style="italic",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()

    roc_auc = metrics.get("roc_auc")
    y_true = metrics.get("y_true")
    y_score = metrics.get("y_score")
    if roc_auc is not None and y_true is not None and y_score is not None:
        if len(y_true) > 0 and len(y_score) > 0:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "roc_curve.png"))
            plt.close()
