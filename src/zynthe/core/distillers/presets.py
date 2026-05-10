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
                                    "regressor": "1x1conv",
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
                                "weight": 0.4,
                                "kd_weight": 0.35,
                                "progressive": False,
                                "layers": ["hidden:-1", "hidden:-2"],
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
                                    "regressor": "1x1conv",
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
                                "weight": 0.45,
                                "kd_weight": 0.3,
                                "progressive": False,
                                "layers": ["hidden:-1", "hidden:-2"],
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
                                "weight": 0.4,
                                "kd_weight": 0.35,
                                "progressive": False,
                                "layers": ["hidden:-1", "hidden:-2"],
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


__all__ = ["list_presets", "describe_preset", "get_preset", "PRESET_LIBRARY"]
