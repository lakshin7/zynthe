"""Distillation preset library.

Provides ready-to-use configuration templates so non-experts can launch
enterprise-grade distillation workflows with a single preset name. Presets
cover common product scenarios (quick start, balanced, transformer heavy,
maximum compression) and are consumed by :mod:`core.distillers.toolkit`
(see :class:`DistillationToolkit`).
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from typing import OrderedDict as OrderedDictType

PresetConfig = Dict[str, Any]


PRESET_LIBRARY: OrderedDictType[str, PresetConfig] = OrderedDict(
    {
        "quick_start": {
            "description": "Single-stage logit distillation for rapid baseline transfer.",
            "distillation": {
                "multi_stage": False,
                "method": "kd",
                "stages": [
                    {
                        "name": "Logit Alignment",
                        "type": "kd",
                        "epochs": 2,
                        "config": {
                            "kd_hinton": {
                                "temperature": 4.0,
                                "alpha": 0.9,
                                "hint_enabled": False,
                                "confidence_scaling": True,
                            }
                        },
                    }
                ],
                "loss_schedule": {"alpha": 0.9, "beta": 0.1, "gamma": 0.0},
            },
            "training": {"epochs": 2},
        },
        "balanced": {
            "description": "Three-stage plan balancing accuracy, compression, and structure.",
            "distillation": {
                "multi_stage": True,
                "loss_schedule": {"alpha": 0.8, "beta": 0.5, "gamma": 0.3},
                "stages": [
                    {
                        "name": "Stage 1 - Logit Alignment",
                        "type": "kd",
                        "epochs": 3,
                        "config": {
                            "kd_hinton": {
                                "temperature": 4.0,
                                "alpha": 0.85,
                                "hint_enabled": False,
                                "confidence_scaling": True,
                            }
                        },
                    },
                    {
                        "name": "Stage 2 - Feature Refinement",
                        "type": "feature",
                        "epochs": 4,
                        "config": {
                            "feature_distillation": {
                                "metrics": ["l2", "cka", "cosine"],
                                "metric_weights": {"l2": 1.0, "cka": 0.7, "cosine": 0.3},
                                "auto_align": True,
                                "auto_layers": "last",
                                "auto_layer_count": 3,
                                "contrastive_temperature": 0.07,
                            }
                        },
                    },
                    {
                        "name": "Stage 3 - Structural Similarity",
                        "type": "similarity",
                        "epochs": 3,
                        "config": {
                            "similarity_transfer": {
                                "similarity_metric": "cosine",
                                "weight": 0.2,
                                "kd_weight": 0.1,
                                "progressive": False,
                                "auto_layers": "last",
                                "auto_layer_count": 1,
                                "normalize": True,
                                "weight_schedule": {
                                    "type": "linear",
                                    "start": 0.05,
                                    "end": 0.2,
                                },
                            },
                        },
                    },
                ],
            },
            "training": {"epochs": 10},
        },
        "vision_transformer": {
            "description": "Transformer-focused pipeline with attention alignment.",
            "distillation": {
                "multi_stage": True,
                "loss_schedule": {"alpha": 0.85, "beta": 0.45, "gamma": 0.35},
                "stages": [
                    {
                        "name": "Stage 1 - KD Warmup",
                        "type": "kd",
                        "epochs": 2,
                        "config": {
                            "kd_hinton": {
                                "temperature": 5.0,
                                "alpha": 0.9,
                                "hint_enabled": False,
                                "confidence_scaling": True,
                            }
                        },
                    },
                    {
                        "name": "Stage 2 - Attention Transfer",
                        "type": "attention",
                        "epochs": 3,
                        "config": {
                            "attention_transfer": {
                                "type": ["self", "relational"],
                                "weight": 0.4,
                                "temperature": 1.5,
                                "use_attention_rollout": True,
                                "use_dual_matching": True,
                                "layer_mapping": {},
                                "teacher_layers": [],
                                "student_layers": [],
                            }
                        },
                    },
                    {
                        "name": "Stage 3 - Feature Polishing",
                        "type": "feature",
                        "epochs": 3,
                        "config": {
                            "feature_distillation": {
                                "metrics": ["l2", "gram"],
                                "auto_align": True,
                                "auto_layers": "attn",
                                "auto_layer_count": 2,
                            }
                        },
                    },
                ],
            },
            "training": {"epochs": 8},
        },
        "all_distillers_t4": {
            "description": "T4-safe full pipeline using KD + Feature + Similarity + Attention.",
            "distillation": {
                "multi_stage": True,
                "loss_schedule": {"alpha": 0.85, "beta": 0.5, "gamma": 0.35},
                "stages": [
                    {
                        "name": "Stage 1 - KD Foundation",
                        "type": "kd",
                        "epochs": 2,
                        "config": {
                            "learning_rate": 2e-5,
                            "weight_decay": 0.01,
                            "scheduler_type": "cosine",
                            "kd_hinton": {
                                "temperature": 4.0,
                                "alpha": 0.85,
                                "hint_enabled": True,
                                "auto_hints": {
                                    "strategy": "last",
                                    "count": 2,
                                    "regressor": "linear",
                                },
                                "confidence_scaling": True,
                            },
                        },
                    },
                    {
                        "name": "Stage 2 - Feature Alignment",
                        "type": "feature",
                        "epochs": 2,
                        "depends_on": [1],
                        "config": {
                            "learning_rate": 1.5e-5,
                            "feature_distillation": {
                                "metrics": ["l2", "cka"],
                                "metric_weights": {"l2": 1.0, "cka": 0.6},
                                "auto_align": True,
                                "auto_layers": "last",
                                "auto_layer_count": 2,
                            },
                        },
                    },
                    {
                        "name": "Stage 3 - Structural Similarity",
                        "type": "similarity",
                        "epochs": 1,
                        "depends_on": [2],
                        "config": {
                            "learning_rate": 1.5e-5,
                            "similarity_transfer": {
                                "similarity_metric": "cosine",
                                "weight": 0.2,
                                "kd_weight": 0.1,
                                "progressive": False,
                                "auto_layers": "last",
                                "auto_layer_count": 1,
                                "normalize": True,
                                "weight_schedule": {
                                    "type": "linear",
                                    "start": 0.05,
                                    "end": 0.2,
                                },
                            },
                        },
                    },
                    {
                        "name": "Stage 4 - Attention Refinement",
                        "type": "attention",
                        "epochs": 1,
                        "depends_on": [3],
                        "config": {
                            "learning_rate": 1e-5,
                            "attention_transfer": {
                                "type": ["self", "relational"],
                                "weight": 0.3,
                                "temperature": 1.2,
                                "auto_detect_layers": True,
                                "use_attention_rollout": True,
                                "use_dual_matching": False,
                                "use_cross_layer_flow": False,
                                "entropy_regularizer": 0.02,
                            },
                        },
                    },
                ],
            },
            "quality_gate": {
                "stop_on_regression": True,
                "max_accuracy_drop": 2.0,
                "min_stage_accuracy": 50.0,
            },
            "train": {"epochs": 6, "batch_size": 16, "mixed_precision": True},
        },
        "all_distillers_classification_smoke": {
            "description": "Fast classification smoke test running all four distillers.",
            "distillation": {
                "multi_stage": True,
                "task_type": "classification",
                "ignore_index": -100,
                "loss_schedule": {"alpha": 0.85, "beta": 0.5, "gamma": 0.35},
                "stages": [
                    {
                        "name": "Stage 1 - KD Warmup",
                        "type": "kd_hinton",
                        "epochs": 1,
                        "config": {
                            "learning_rate": 2e-5,
                            "kd_hinton": {
                                "temperature": 4.0,
                                "alpha": 0.85,
                                "hint_enabled": True,
                                "auto_hints": {
                                    "strategy": "last",
                                    "count": 1,
                                    "regressor": "linear",
                                },
                            },
                        },
                    },
                    {
                        "name": "Stage 2 - Feature Alignment",
                        "type": "feature",
                        "epochs": 1,
                        "depends_on": [1],
                        "config": {
                            "learning_rate": 1.5e-5,
                            "feature_distillation": {
                                "metrics": ["l2", "cka"],
                                "metric_weights": {"l2": 1.0, "cka": 0.6},
                                "auto_align": True,
                                "auto_layers": "last",
                                "auto_layer_count": 2,
                            },
                        },
                    },
                    {
                        "name": "Stage 3 - Similarity Transfer",
                        "type": "similarity",
                        "epochs": 1,
                        "depends_on": [2],
                        "config": {
                            "learning_rate": 1.5e-5,
                            "similarity_transfer": {
                                "similarity_metric": "cosine",
                                "weight": 0.2,
                                "kd_weight": 0.1,
                                "progressive": False,
                                "auto_layers": "last",
                                "auto_layer_count": 1,
                                "normalize": True,
                                "weight_schedule": {
                                    "type": "linear",
                                    "start": 0.05,
                                    "end": 0.2,
                                },
                            },
                        },
                    },
                    {
                        "name": "Stage 4 - Attention Refinement",
                        "type": "attention",
                        "epochs": 1,
                        "depends_on": [3],
                        "config": {
                            "learning_rate": 1e-5,
                            "attention_transfer": {
                                "type": ["self", "relational"],
                                "weight": 0.3,
                                "temperature": 1.2,
                                "auto_detect_layers": True,
                                "use_attention_rollout": True,
                                "use_dual_matching": False,
                            },
                        },
                    },
                ],
            },
            "train": {
                "epochs": 4,
                "batch_size": 16,
                "mixed_precision": True,
                "gradient_accumulation_steps": 1,
            },
        },
        "all_distillers_causal_lm_smoke": {
            "description": "Fast GPT/causal-LM smoke test running all four distillers.",
            "distillation": {
                "multi_stage": True,
                "task_type": "causal_lm",
                "loss_mode": "auto",
                "ignore_index": -100,
                "shift_labels": True,
                "loss_schedule": {"alpha": 0.8, "beta": 0.45, "gamma": 0.35},
                "stages": [
                    {
                        "name": "Stage 1 - Token KD",
                        "type": "kd_hinton",
                        "epochs": 1,
                        "config": {
                            "learning_rate": 2e-5,
                            "kd_hinton": {
                                "temperature": 2.0,
                                "alpha": 0.8,
                                "hint_enabled": False,
                            },
                        },
                    },
                    {
                        "name": "Stage 2 - Similarity Structure",
                        "type": "similarity",
                        "epochs": 1,
                        "depends_on": [1],
                        "config": {
                            "learning_rate": 1.5e-5,
                            "similarity_transfer": {
                                "similarity_metric": "cosine",
                                "weight": 0.2,
                                "kd_weight": 0.1,
                                "progressive": False,
                                "auto_layers": "last",
                                "auto_layer_count": 1,
                                "normalize": True,
                                "weight_schedule": {
                                    "type": "linear",
                                    "start": 0.05,
                                    "end": 0.2,
                                },
                            },
                        },
                    },
                    {
                        "name": "Stage 3 - Attention Transfer",
                        "type": "attention",
                        "epochs": 1,
                        "depends_on": [2],
                        "config": {
                            "learning_rate": 1e-5,
                            "attention_transfer": {
                                "type": ["self", "relational"],
                                "weight": 0.25,
                                "temperature": 1.0,
                                "auto_detect_layers": True,
                                "use_attention_rollout": True,
                                "use_dual_matching": False,
                            },
                        },
                    },
                    {
                        "name": "Stage 4 - Feature Polishing",
                        "type": "feature",
                        "epochs": 1,
                        "depends_on": [3],
                        "config": {
                            "learning_rate": 1e-5,
                            "feature_distillation": {
                                "metrics": ["l2"],
                                "auto_align": True,
                                "auto_layers": "last",
                                "auto_layer_count": 1,
                            },
                        },
                    },
                ],
            },
            "train": {
                "epochs": 4,
                "batch_size": 2,
                "mixed_precision": True,
                "gradient_accumulation_steps": 8,
            },
            "model": {
                "type": "causallm",
                "max_length": 512,
            },
        },
        "multimodal": {
            "description": "Contrastive dual-encoder distillation for CLIP-style vision-language models.",
            "distillation": {
                "multi_stage": True,
                "loss_schedule": {"alpha": 0.7, "beta": 0.6, "gamma": 0.4},
                "stages": [
                    {
                        "name": "Stage 1 - Embedding Alignment",
                        "type": "feature",
                        "epochs": 3,
                        "config": {
                            "feature_distillation": {
                                "metrics": ["cosine", "l2"],
                                "metric_weights": {"cosine": 1.0, "l2": 0.5},
                                "auto_align": True,
                                "auto_layers": "last",
                                "auto_layer_count": 2,
                                "contrastive_temperature": 0.07,
                            }
                        },
                    },
                    {
                        "name": "Stage 2 - Relational Similarity",
                        "type": "similarity",
                        "epochs": 3,
                        "depends_on": [1],
                        "config": {
                            "similarity_metric": "cosine",
                            "weight": 0.6,
                            "kd_weight": 0.2,
                            "progressive": True,
                            "progressive_epochs": 2,
                            "layers": ["hidden:-1", "hidden:-2"],
                        },
                    },
                    {
                        "name": "Stage 3 - Attention Alignment",
                        "type": "attention",
                        "epochs": 2,
                        "depends_on": [2],
                        "config": {
                            "attention_transfer": {
                                "type": ["self", "relational"],
                                "weight": 0.4,
                                "temperature": 1.2,
                                "auto_detect_layers": True,
                                "use_attention_rollout": True,
                                "use_dual_matching": True,
                                "entropy_regularizer": 0.02,
                            }
                        },
                    },
                ],
            },
            "training": {"epochs": 8},
        },
        "compression_max": {
            "description": "Aggressive compression with similarity, attention, and optional QAT.",
            "distillation": {
                "multi_stage": True,
                "loss_schedule": {"alpha": 0.9, "beta": 0.6, "gamma": 0.4},
                "stages": [
                    {
                        "name": "Stage 1 - Logit KD",
                        "type": "kd",
                        "epochs": 3,
                        "config": {
                            "kd_hinton": {
                                "temperature": 6.0,
                                "alpha": 0.92,
                                "hint_enabled": True,
                                "auto_hints": {
                                    "strategy": "uniform",
                                    "count": 3,
                                    "regressor": "mlp",
                                },
                                "confidence_scaling": True,
                            }
                        },
                    },
                    {
                        "name": "Stage 2 - Feature Compression",
                        "type": "feature",
                        "epochs": 4,
                        "config": {
                            "feature_distillation": {
                                "metrics": ["l2", "cka", "contrastive"],
                                "metric_weights": {"l2": 1.0, "cka": 0.5, "contrastive": 0.7},
                                "auto_align": True,
                                "auto_layers": "mixed",
                                "auto_layer_count": 4,
                                "contrastive_temperature": 0.05,
                            }
                        },
                    },
                    {
                        "name": "Stage 3 - Relational Transfer",
                        "type": "similarity",
                        "epochs": 3,
                        "config": {
                            "similarity_metric": "graph",
                            "graph_mode": True,
                            "graph_threshold": 0.6,
                            "weight": 0.5,
                            "kd_weight": 0.4,
                            "layers": ["hidden:-1", "hidden:-3"],
                        },
                    },
                    {
                        "name": "Stage 4 - Attention Alignment",
                        "type": "attention",
                        "epochs": 2,
                        "config": {
                            "attention_transfer": {
                                "type": ["spatial", "self"],
                                "weight": 0.35,
                                "temperature": 1.2,
                                "use_cross_layer_flow": True,
                                "use_dual_matching": True,
                                "entropy_regularizer": 0.05,
                            }
                        },
                    },
                ],
            },
            "training": {"epochs": 12},
        },
    }
)


def list_presets() -> List[str]:
    """Return available preset identifiers."""
    return list(PRESET_LIBRARY.keys())


def describe_preset(name: str) -> str:
    """Return human-readable description for *name*."""
    preset = PRESET_LIBRARY.get(name)
    if not preset:
        raise KeyError(f"Unknown distillation preset: {name}")
    return preset.get("description", "")


def get_preset(name: str, *, deep: bool = True) -> PresetConfig:
    """Retrieve preset configuration.

    Args:
        name: Preset name registered in :data:`PRESET_LIBRARY`.
        deep: When True (default) return a deep copy so callers can freely mutate.
    """
    if name not in PRESET_LIBRARY:
        raise KeyError(f"Unknown distillation preset: {name}")
    return deepcopy(PRESET_LIBRARY[name]) if deep else PRESET_LIBRARY[name]


# ----------------------------------------------------------------------------
# Phase 5 Iteration 2 — typed Plan / Stage DSL
# ----------------------------------------------------------------------------
#
# The :func:`get_preset(name)` API is a stringly-typed dict lookup.  The
# :class:`Plan` and :class:`Stage` types below are a thin typed wrapper
# that makes the intent more explicit at the call site and gives us
# validation we couldn't easily add to the dict form.  Both shapes
# interoperate — :meth:`Plan.to_dict` produces the same dict that
# :func:`get_preset` returns, so the typed DSL doesn't fork the
# downstream pipeline-builder code path.
#
# Example::
#
#     plan = Plan(name="vision_default", epochs=10, stages=[
#         Stage(loss="kd_hinton", weight=0.6, temperature=4.0),
#         Stage(loss="feature", weight=0.4, layers=["layers.1", "layers.3"]),
#     ])
#     cfg = plan.to_dict()
#


@dataclass
class Stage:
    """A single distillation stage in a :class:`Plan`.

    Attributes:
        loss: Distiller registry key (e.g. ``"kd_hinton"``, ``"feature"``).
        weight: Loss weight for the multi-stage aggregator.
        epochs: Optional number of epochs to spend on this stage.  ``None``
            inherits from the parent :class:`Plan`.
        config: Distiller-specific configuration.  Empty dict by default.
    """

    loss: str
    weight: float = 1.0
    epochs: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.loss:
            raise ValueError("Stage.loss must be a non-empty distiller name")
        if self.weight < 0:
            raise ValueError(f"Stage.weight must be >= 0; got {self.weight}")
        if self.epochs is not None and self.epochs < 1:
            raise ValueError(f"Stage.epochs must be >= 1; got {self.epochs}")

    def to_dict(self) -> Dict[str, Any]:
        """Render the stage as a pipeline-builder-friendly dict.

        Mirrors the shape of stages in the existing ``PRESET_LIBRARY``:
            ``{"name": ..., "type": ..., "epochs": ..., "config": {...}}``
        """
        out: Dict[str, Any] = {
            "name": f"Stage - {self.loss}",
            "type": self.loss,
            "config": dict(self.config),
        }
        if self.epochs is not None:
            out["epochs"] = self.epochs
        return out


@dataclass
class Plan:
    """A typed multi-stage distillation plan.

    Attributes:
        name: Plan identifier — used in logs / output dirs.
        stages: Ordered list of :class:`Stage`.  Each stage's loss
            weight is normalised by ``MultiStagePipeline.setup()`` so
            the absolute magnitudes are not important.
        epochs: Default number of epochs for the whole plan.
        description: Human-readable note (used by ``describe_preset``).
    """

    name: str
    stages: List[Stage] = field(default_factory=list)
    epochs: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Plan.name must be a non-empty string")
        if not self.stages:
            raise ValueError(f"Plan {self.name!r} has no stages")
        if self.epochs < 1:
            raise ValueError(f"Plan.epochs must be >= 1; got {self.epochs}")

    def to_dict(self) -> PresetConfig:
        """Render the plan as the dict the rest of zynthe consumes.

        Output shape mirrors :func:`get_preset(name)`:

            {
                "description": "...",
                "training": {"epochs": ...},
                "distillation": {
                    "type": "multi_stage",
                    "stages": [<stage_dict>, ...],
                },
            }
        """
        return {
            "description": self.description,
            "training": {"epochs": self.epochs},
            "distillation": {
                "type": "multi_stage",
                "stages": [s.to_dict() for s in self.stages],
            },
        }

    @classmethod
    def from_preset(cls, name: str) -> "Plan":
        """Build a typed :class:`Plan` from a preset already in
        :data:`PRESET_LIBRARY`.

        Falls back to a best-effort conversion (a single stage with
        the full preset as its config) for presets that don't map
        cleanly onto the ``Plan`` shape.
        """
        raw = get_preset(name)
        stages_cfg = (
            raw.get("distillation", {}).get("stages")
            or raw.get("stages")
            or []
        )
        if stages_cfg:
            stages: List[Stage] = []
            for s in stages_cfg:
                if not isinstance(s, dict):
                    continue
                loss = s.get("type") or s.get("loss") or s.get("name", "kd_hinton")
                weight = float(s.get("weight", 1.0))
                cfg = s.get("config", {})
                stages.append(Stage(loss=loss, weight=weight, config=cfg))
            return cls(
                name=name,
                stages=stages,
                epochs=int(raw.get("training", {}).get("epochs", 1)),
                description=raw.get("description", ""),
            )
        # Fallback: best-effort — one stage with the full preset as
        # its config.
        return cls(
            name=name,
            stages=[Stage(loss="kd_hinton", weight=1.0, config=raw)],
            epochs=int(raw.get("training", {}).get("epochs", 1)),
            description=raw.get("description", ""),
        )


def register_plan(plan: Plan) -> None:
    """Register a :class:`Plan` so it's available via
    :func:`get_preset(plan.name)`."""
    PRESET_LIBRARY[plan.name] = plan.to_dict()


# ----------------------------------------------------------------------------
# Phase 5 Iteration 2 — 5 new presets
# ----------------------------------------------------------------------------
#
# Each preset is defined as a Plan and then registered with the legacy
# dict API so the toolkit / smoke gates can find it via
# get_preset(name).

register_plan(
    Plan(
        name="compression_max",
        description=(
            "Maximum compression ratio: heavy Hinton KD + feature L2 + small "
            "rationale supervision.  Suitable for mobile / edge targets where "
            "latency is the only priority."
        ),
        epochs=15,
        stages=[
            Stage(loss="kd_hinton", weight=0.7, config={"temperature": 4.0, "alpha": 0.7}),
            Stage(
                loss="feature",
                weight=0.3,
                config={
                    "feature_distillation": {
                        "enabled": True,
                        "layers": [{"teacher": "layers.11", "student": "layers.3"}],
                        "metrics": ["l2"],
                    }
                },
            ),
        ],
    )
)

register_plan(
    Plan(
        name="fidelity_first",
        description=(
            "Conservative distillation: CRD + relational + aux heads. "
            "Slowest but most accurate on small labeled datasets."
        ),
        epochs=20,
        stages=[
            Stage(loss="kd_hinton", weight=0.4, config={"temperature": 2.0}),
            Stage(
                loss="contrastive",
                weight=0.3,
                config={"temperature": 0.07, "memory_bank_size": 256},
            ),
            Stage(
                loss="relational",
                weight=0.2,
                config={"student_layers": ["layers.6"], "teacher_layers": ["encoder.layer.6"]},
            ),
            Stage(
                loss="aux_head",
                weight=0.1,
                config={"student_layers": ["layers.4", "layers.8"], "num_classes": 2},
            ),
        ],
    )
)

register_plan(
    Plan(
        name="vision_default",
        description=(
            "Default vision pair (ViT / DeiT / ResNet family): feature + "
            "attention alignment on a small set of layers."
        ),
        epochs=10,
        stages=[
            Stage(loss="kd_hinton", weight=0.4, config={"temperature": 3.0}),
            Stage(
                loss="feature",
                weight=0.4,
                config={
                    "feature_distillation": {
                        "enabled": True,
                        "layers": [
                            {"teacher": "encoder.layer.6", "student": "layers.2"},
                            {"teacher": "encoder.layer.10", "student": "layers.4"},
                        ],
                        "metrics": ["l2", "cosine"],
                    }
                },
            ),
            Stage(
                loss="attention",
                weight=0.2,
                config={
                    "attention_transfer": {
                        "type": ["spatial", "self"],
                        "temperature": 1.0,
                    }
                },
            ),
        ],
    )
)

register_plan(
    Plan(
        name="causal_lm_default",
        description=(
            "Decoder-only LLM distillation: Hinton KD on the LM head + small "
            "rationale multi-task for data efficiency."
        ),
        epochs=8,
        stages=[
            Stage(
                loss="kd_hinton",
                weight=0.8,
                config={
                    "temperature": 2.0,
                    "alpha": 0.6,
                    "dynamic_temperature": "learnable",
                },
            ),
            Stage(
                loss="rationale",
                weight=0.2,
                config={
                    "label_prefix": "label: ",
                    "rationale_prefix": "rationale: ",
                    "label_weight": 1.0,
                    "rationale_weight": 0.5,
                },
            ),
        ],
    )
)

register_plan(
    Plan(
        name="multimodal_default",
        description=(
            "CLIP-style vision-language: feature alignment between vision "
            "and text towers + global contrastive term."
        ),
        epochs=12,
        stages=[
            Stage(loss="kd_hinton", weight=0.4, config={"temperature": 2.0}),
            Stage(
                loss="feature",
                weight=0.3,
                config={
                    "feature_distillation": {
                        "enabled": True,
                        "layers": [
                            {"teacher": "vision_model.encoder.layers.5", "student": "vision_model.encoder.layers.2"},
                        ],
                        "metrics": ["l2", "gram"],
                    }
                },
            ),
            Stage(
                loss="contrastive",
                weight=0.3,
                config={"temperature": 0.1, "projection_dim": 128},
            ),
        ],
    )
)


__all__ = [
    "list_presets",
    "describe_preset",
    "get_preset",
    "Plan",
    "Stage",
    "PRESET_LIBRARY",
    "register_plan",
]
