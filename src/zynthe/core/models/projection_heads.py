"""Projection head utilities for distillation and representation tasks."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Configurable MLP projection head."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        layers = []

        if hidden_dim is not None:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation is not None:
                layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        if normalize:
            layers.append(nn.LayerNorm(out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualProjectionHead(nn.Module):
    """Residual MLP projection head with skip connections."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden = hidden_dim or out_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x) + self.shortcut(x))


class AttentionProjectionHead(nn.Module):
    """Projection head leveraging a lightweight self-attention block."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class ProjectionHeadFactory:
    """Factory for creating projection heads by name."""

    _registry: Dict[str, Type[nn.Module]] = {
        "mlp": ProjectionHead,
        "residual": ResidualProjectionHead,
        "attention": AttentionProjectionHead,
    }

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> nn.Module:
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown projection head '{name}'. Available: {list(cls._registry)}")
        return cls._registry[key](*args, **kwargs)

    @classmethod
    def register(cls, name: str, module_cls: Type[nn.Module]) -> None:
        cls._registry[name.lower()] = module_cls


def register_projection_head(name: str, module_cls: Type[nn.Module]) -> None:
    """Public helper to extend the projection head factory."""

    ProjectionHeadFactory.register(name, module_cls)
