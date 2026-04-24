"""Public distillation API surface."""

from .attention_transfer import AttentionTransferDistiller
from .feature_distiller import FeatureDistiller
from .kd_hinton import KDHintonDistiller
from .similarity_transfer import SimilarityTransfer
from .causal_lm import SafeCausalLMTrainer

try:
	from .toolkit import DistillationToolkit
except Exception:  # pragma: no cover - optional dependency chain (viz libs)
	DistillationToolkit = None  # type: ignore[assignment,misc]

from .presets import list_presets, describe_preset, get_preset

try:
	from .multi_stage_distiller import MultiStageDistiller
except Exception:  # pragma: no cover - optional dependency chain (viz libs)
	MultiStageDistiller = None  # type: ignore[assignment,misc]

__all__ = [
	"AttentionTransferDistiller",
	"FeatureDistiller",
	"KDHintonDistiller",
	"MultiStageDistiller",
	"SimilarityTransfer",
	"DistillationToolkit",
	"SafeCausalLMTrainer",
	"list_presets",
	"describe_preset",
	"get_preset",
]
