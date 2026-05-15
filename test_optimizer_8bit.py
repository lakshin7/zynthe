import torch
import torch.nn as nn
from zynthe.training.optimizer import OptimizerFactory

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

model = DummyModel()
config = {"optimizer": "adamw8bit", "learning_rate": 1e-4}

optim = OptimizerFactory.get_optimizer(model, config)
print(f"Optimizer created: {type(optim)}")
