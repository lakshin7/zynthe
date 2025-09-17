import torch
import torch.nn as nn
from typing import Optional

class ProjectionHead(nn.Module):
	"""
	Simple MLP projection head for feature distillation.
	Can be extended for more complex heads (e.g., with normalization, dropout).
	"""
	def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
		super().__init__()
		if hidden_dim is not None:
			self.net = nn.Sequential(
				nn.Linear(in_dim, hidden_dim),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim, out_dim)
			)
		else:
			self.net = nn.Linear(in_dim, out_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
