import math

import matplotlib
import pytest
import torch
import torch.nn as nn

from core.distillers.kd_hinton import KDHintonDistiller
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_all_metrics
from training.trainer import Trainer

matplotlib.use("Agg", force=True)


@pytest.fixture()
def cpu_device():
    return torch.device("cpu")


def test_compute_all_metrics_basic():
    preds = [0, 1, 1, 0, 1]
    labels = [0, 1, 0, 0, 1]
    metrics = compute_all_metrics(preds, labels)

    assert pytest.approx(0.8, rel=1e-3) == metrics["accuracy"]
    assert "balanced_accuracy" in metrics
    assert "matthews_corrcoef" in metrics
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_distiller_compute_loss(dummy_models, cpu_device):
    teacher, student = dummy_models
    config = {
        "kd_hinton": {
            "temperature": 2.0,
            "alpha": 0.5,
            "hint_enabled": False,
        }
    }
    distiller = KDHintonDistiller(teacher=teacher, student=student, config=config, device=cpu_device)

    batch_size, seq_len = 6, 8
    input_ids = torch.randint(1, 50, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.long)

    teacher_outputs = distiller.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    student_outputs = distiller.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss, loss_dict = distiller.compute_loss(student_outputs, teacher_outputs, labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert "kd_loss" in loss_dict
    assert "total" in loss_dict


def test_evaluator_smoke(dummy_tokenizer, dummy_models, sample_loaders, cpu_device):
    _, student = dummy_models
    student.to(cpu_device)

    _, val_loader = sample_loaders
    loss_fn = nn.CrossEntropyLoss()

    evaluator = Evaluator(
        model=student,
        dataloader=val_loader,
        tokenizer=dummy_tokenizer,
        device=cpu_device,
        loss_fn=loss_fn,
        enable_runtime_profiling=False,
    )

    results = evaluator.evaluate()

    assert results["num_samples"] > 0
    assert "metrics" in results
    assert "accuracy" in results["metrics"]
    assert isinstance(results.get("diagnostics"), dict)


def test_trainer_train_and_evaluate(tmp_path, dummy_tokenizer, dummy_models, sample_loaders, cpu_device):
    teacher, student = dummy_models
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()

    train_loader, val_loader = sample_loaders

    config = {
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 1e-3,
            "learning_rate": 1e-3,
            "optimizer": "adamw",
            "use_amp": False,
            "csv_logging": False,
            "log_detail": False,
            "enable_comparison_plot": False,
            "show_eta": False,
            "batch_log_interval": 1000,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "dynamic_lr": False,
        },
        "distillation": {
            "type": "kd",
            "config": {
                "config": {
                    "kd_hinton": {
                        "temperature": 2.0,
                        "alpha": 0.5,
                        "hint_enabled": False,
                    }
                }
            },
        },
    }

    trainer = Trainer(
        teacher=teacher,
        student=student,
        tokenizer=dummy_tokenizer,
        config=config,
        device=cpu_device,
        experiment_dir=str(experiment_dir),
    )

    train_loss = trainer.train_epoch(train_loader)
    assert math.isfinite(train_loss)

    val_loss, metrics, extended, details = trainer.evaluate(val_loader, compute_extended=False)

    assert math.isfinite(val_loss)
    assert isinstance(metrics, list)
    assert details["metrics"]
    assert "accuracy" in details["metrics"]
    assert len(trainer.metrics_history["accuracy"]) >= 1
    assert isinstance(extended, dict)

    runtime_snapshots = trainer.eval_runtime_history
    if runtime_snapshots:
        snapshot = runtime_snapshots[-1]
        assert "mean_ms" in snapshot
        assert snapshot["batches"] > 0
