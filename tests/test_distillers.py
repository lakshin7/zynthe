import pytest
import torch
import torch.nn as nn
from zynthe.core.distillers.kd_hinton import KDHintonDistiller
from zynthe.core.distillers.feature_distiller import FeatureDistiller

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x, labels=None):
        out = self.fc(x)
        return type('Output', (), {'logits': out, 'hidden_states': [out]})()

def test_kd_hinton():
    teacher = DummyModel()
    student = DummyModel()
    distiller = KDHintonDistiller(teacher, student, config={"alpha": 0.5, "temperature": 2.0}, device="cpu")

    x = torch.randn(2, 10)
    targets = torch.randint(0, 10, (2,))
    s_out = student(x)
    t_out = teacher(x)
    loss, details = distiller.compute_loss(s_out, t_out, targets)
    assert isinstance(loss, torch.Tensor)
    assert "kd_loss" in details

def test_feature_distiller():
    teacher = DummyModel()
    student = DummyModel()
    distiller = FeatureDistiller(teacher, student, config={"alpha": 0.5}, device="cpu")

    x = torch.randn(2, 10)
    s_out = student(x)
    t_out = teacher(x)
    loss, details = distiller.compute_loss(s_out, t_out, student_features={"layer1": x}, teacher_features={"layer1": x})
    assert isinstance(loss, torch.Tensor)
