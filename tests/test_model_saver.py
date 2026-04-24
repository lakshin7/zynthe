from __future__ import annotations

import json
from pathlib import Path

import torch

from core.models.model_saver import ModelSaver
from evaluation.evaluation_report import EvaluationReport


class TinyExportModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.config = type("Cfg", (), {"_name_or_path": "tiny-export-model"})()

    def forward(self, input_ids=None, attention_mask=None):
        if input_ids is None:
            input_ids = torch.zeros(1, 4)
        return self.linear(input_ids.float())

    def save_pretrained(self, path: str, safe_serialization: bool = False) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_dir / ("model.safetensors" if safe_serialization else "pytorch_model.bin"))


class TinyTokenizer:
    def save_pretrained(self, path: str) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer.json").write_text(json.dumps({"tiny": True}), encoding="utf-8")

    def __call__(self, text: str, return_tensors: str = "pt"):
        return {"input_ids": torch.ones(1, 4, dtype=torch.float32), "attention_mask": torch.ones(1, 4)}


def test_save_training_run_creates_expected_bundle(tmp_path: Path) -> None:
    model = TinyExportModel()
    tokenizer = TinyTokenizer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    report = EvaluationReport(loss=0.1, metrics={"accuracy": 0.95}, modality="text")

    save_dir = ModelSaver.save_training_run(
        model=model,
        tokenizer=tokenizer,
        save_dir=str(tmp_path / "bundle"),
        config={"train": {"epochs": 2}},
        metrics_history={"accuracy": [0.9, 0.95]},
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=8,
        best_metric=0.95,
        evaluation_report=report,
        use_safetensors=True,
    )

    save_path = Path(save_dir)
    assert (save_path / "training_run_manifest.json").exists()
    assert (save_path / "training_config.json").exists()
    assert (save_path / "metrics_history.json").exists()
    assert (save_path / "evaluation_report.json").exists()
    assert (save_path / "checkpoint.pt").exists()
    assert (save_path / "tokenizer.json").exists()

    manifest = json.loads((save_path / "training_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["has_config"] is True
    assert manifest["has_metrics_history"] is True
    assert manifest["has_checkpoint"] is True
    assert manifest["has_evaluation_report"] is True


def test_export_for_deployment_supports_multi_format(tmp_path: Path) -> None:
    model = TinyExportModel()
    tokenizer = TinyTokenizer()

    save_dir = ModelSaver.export_for_deployment(
        model=model,
        tokenizer=tokenizer,
        save_dir=str(tmp_path / "exports"),
        export_format="safetensors,gguf,bitnet",
    )

    export_root = Path(save_dir)
    assert (export_root / "safetensors" / "export_metadata.json").exists()
    assert (export_root / "gguf" / "README_GGUF.txt").exists()
    assert (export_root / "bitnet" / "README_BITNET.txt").exists()


def test_export_for_deployment_writes_torchscript(tmp_path: Path) -> None:
    model = TinyExportModel()
    tokenizer = TinyTokenizer()
    output = ModelSaver.export_for_deployment(
        model=model,
        tokenizer=tokenizer,
        save_dir=str(tmp_path / "torchscript"),
        export_format="torchscript",
        example_inputs=(torch.ones(1, 4),),
    )

    export_path = Path(output)
    assert (export_path / "model.torchscript.pt").exists()
    assert (export_path / "export_metadata.json").exists()
