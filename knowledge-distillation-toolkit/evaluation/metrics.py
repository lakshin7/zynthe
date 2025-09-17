import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import json
import warnings


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


def compute_classwise_metrics(preds, labels):
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    if len(labels_np) == 0:
        return {"precision_per_class": {}, "recall_per_class": {}, "f1_per_class": {}}
    precision = precision_score(labels_np, preds_np, average=None, zero_division=0)
    recall = recall_score(labels_np, preds_np, average=None, zero_division=0)
    f1 = f1_score(labels_np, preds_np, average=None)
    classes = np.unique(np.concatenate((labels_np, preds_np)))
    precision_dict = {int(cls): p for cls, p in zip(classes, precision)}
    recall_dict = {int(cls): r for cls, r in zip(classes, recall)}
    f1_dict = {int(cls): f for cls, f in zip(classes, f1)}
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


def compute_all_metrics(preds, labels, pred_probs=None):
    if preds is None or labels is None:
        return {}
    metrics = {
        "accuracy": compute_accuracy(preds, labels),
        "f1": compute_f1(preds, labels),
        "precision": compute_precision(preds, labels),
        "recall": compute_recall(preds, labels),
        "confusion_matrix": compute_confusion_matrix(preds, labels),
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
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels if labels is not None else "auto",
                    yticklabels=labels if labels is not None else "auto")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()

    roc_auc = metrics.get("roc_auc")
    y_true = metrics.get("y_true")
    y_score = metrics.get("y_score")
    if roc_auc is not None and y_true is not None and y_score is not None:
        if len(y_true) > 0 and len(y_score) > 0:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "roc_curve.png"))
            plt.close()