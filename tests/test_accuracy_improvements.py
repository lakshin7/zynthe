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

def test_adaptive_kd():
    teacher = DummyModel()
    student = DummyModel()
    distiller = KDHintonDistiller(teacher, student, config={"alpha": 0.5, "temperature": 2.0}, device="cpu")

    # Run multiple times to trigger adaptive weighting
    distiller.train()
    x = torch.randn(2, 10)
    targets = torch.randint(0, 10, (2,))
    for i in range(5):
        s_out = student(x)
        t_out = teacher(x)
        loss, details = distiller.compute_loss(s_out, t_out, targets)

    assert "ce_loss" in details

def test_dynamic_feature_weighting():
    teacher = DummyModel()
    student = DummyModel()
    distiller = FeatureDistiller(teacher, student, config={"alpha": 0.5}, device="cpu")

    x = torch.randn(2, 10)
    s_out = student(x)
    t_out = teacher(x)
    loss, details = distiller.compute_loss(s_out, t_out, student_features={"layer1": x}, teacher_features={"layer1": x})

test_adaptive_kd()
test_dynamic_feature_weighting()
print("Accuracy improvements initialized without errors")
