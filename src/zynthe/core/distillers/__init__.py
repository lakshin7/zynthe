"""Public distillation API surface."""

from __future__ import annotations

from .attention_transfer import AttentionTransferDistiller
from .aux_head_distiller import AuxHeadDistiller
from .causal_lm import SafeCausalLMTrainer
from .contrastive_distiller import ContrastiveDistiller
from .feature_distiller import FeatureDistiller
from .kd_hinton import KDHintonDistiller
from .projection_distiller import ProjectionDistiller
from .relational_distiller import RelationalDistiller
from .similarity_transfer import SimilarityTransfer

try:
    from .toolkit import DistillationToolkit
except Exception:  # pragma: no cover - optional dependency chain (viz libs)
    DistillationToolkit = None  # type: ignore[assignment,misc]

from .presets import describe_preset, get_preset, list_presets

try:
    from .multi_stage_distiller import MultiStageDistiller
except Exception:  # pragma: no cover - optional dependency chain (viz libs)
    MultiStageDistiller = None  # type: ignore[assignment,misc]

__all__ = [
    "AttentionTransferDistiller",
    "AuxHeadDistiller",
    "ContrastiveDistiller",
    "FeatureDistiller",
    "KDHintonDistiller",
    "MultiStageDistiller",
    "ProjectionDistiller",
    "RelationalDistiller",
    "SimilarityTransfer",
    "DistillationToolkit",
    "SafeCausalLMTrainer",
    "list_presets",
    "describe_preset",
    "get_preset",
]
