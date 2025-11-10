"""Distillation preset library.

Provides ready-to-use configuration templates so non-experts can launch
enterprise-grade distillation workflows with a single preset name. Presets
cover common product scenarios (quick start, balanced, transformer heavy,
maximum compression) and are consumed by :mod:`core.distillers.toolkit`
(see :class:`DistillationToolkit`).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, OrderedDict as OrderedDictType
from collections import OrderedDict

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
                                "hint_enabled": True,
                                "auto_hints": {
                                    "strategy": "last",
                                    "count": 2,
                                    "regressor": "1x1conv",
                                },
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
                            "similarity_metric": "cosine",
                            "weight": 0.6,
                            "kd_weight": 0.3,
                            "progressive": True,
                            "progressive_epochs": 2,
                            "layers": ["hidden:-1", "hidden:-2"],
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


__all__ = ["list_presets", "describe_preset", "get_preset", "PRESET_LIBRARY"]
