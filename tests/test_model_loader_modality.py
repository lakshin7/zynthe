from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import core.models.model_loader as model_loader
from core.models.model_loader import ModelLoader


class FakeVisionModel(nn.Module):
    loaded_names: list[str] = []

    def __init__(self, name: str = "fake-vision") -> None:
        super().__init__()
        self.config = SimpleNamespace(_name_or_path=name, model_type="vit")
        self.classifier = nn.Linear(3, 2)

    @classmethod
    def from_pretrained(cls, name: str, **kwargs):
        cls.loaded_names.append(name)
        return cls(name=name)

    def forward(self, pixel_values=None, labels=None):
        if pixel_values is None:
            pixel_values = torch.zeros(1, 3, 1, 1)
        pooled = pixel_values.float().mean(dim=(-1, -2))
        logits = self.classifier(pooled)
        return SimpleNamespace(logits=logits, loss=None)


class FakeProcessor:
    loaded_names: list[str] = []

    @classmethod
    def from_pretrained(cls, name: str, **kwargs):
        cls.loaded_names.append(name)
        return cls()

    def save_pretrained(self, path: str) -> None:
        return None


class UnexpectedAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("vision loading should not use AutoModel")


class UnexpectedTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("vision loading should not use AutoTokenizer")


def test_model_loader_accepts_teacher_name_alias() -> None:
    loader = ModelLoader(
        {
            "model": {
                "teacher_name": "teacher-a",
                "student_name": "student-b",
                "type": "causal_lm",
            }
        },
        device="cpu",
    )

    spec = loader._build_spec(use_agent=False, data_samples=None)

    assert spec.teacher_name == "teacher-a"
    assert spec.student_name == "student-b"


def test_model_loader_uses_image_classification_model_and_processor(monkeypatch) -> None:
    FakeVisionModel.loaded_names.clear()
    FakeProcessor.loaded_names.clear()
    monkeypatch.setattr(model_loader, "AutoModel", UnexpectedAutoModel)
    monkeypatch.setattr(model_loader, "AutoTokenizer", UnexpectedTokenizer)
    monkeypatch.setattr(model_loader, "AutoModelForImageClassification", FakeVisionModel, raising=False)
    monkeypatch.setattr(model_loader, "AutoImageProcessor", FakeProcessor, raising=False)

    loader = ModelLoader(
        {
            "model": {
                "name": "teacher-vit",
                "student_name": "student-vit",
                "type": "vision",
            }
        },
        device="cpu",
    )

    teacher, student, processor = loader.load()

    assert isinstance(teacher, FakeVisionModel)
    assert isinstance(student, FakeVisionModel)
    assert isinstance(processor, FakeProcessor)
    assert FakeVisionModel.loaded_names == ["teacher-vit", "student-vit"]
    assert FakeProcessor.loaded_names == ["teacher-vit"]
