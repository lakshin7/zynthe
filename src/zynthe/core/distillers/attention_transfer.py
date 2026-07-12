from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from zynthe.core.distillers.base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


# ============================================================================
# 1. ATTENTION EXTRACTOR MODULE
# ============================================================================


class AttentionExtractor:
    """
    Universal Attention Extractor with hooks for CNNs, Transformers, and Multimodal models.

    Capabilities:
    - CNN: Extract feature maps and compute spatial attention
    - Transformer: Extract self-attention scores from each block
    - Multimodal: Extract both self-attention and cross-attention
    - Video: Extract temporal attention weights
    """

    def __init__(self, model: nn.Module, layer_names: List[str], model_type: str = "auto"):
        """
        Args:
            model: PyTorch model to extract attention from
            layer_names: List of layer names to hook (e.g., ["layer3", "layer4"])
            model_type: "cnn", "transformer", "multimodal", "video", or "auto"
        """
        self.model = model
        self.layer_names = layer_names
        self.model_type = self._detect_model_type(model) if model_type == "auto" else model_type

        # Storage for extracted features and attentions
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.attention_scores: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []

        self._register_hooks()

    def _detect_model_type(self, model: nn.Module) -> str:
        """Auto-detect model architecture type."""
        model_class = model.__class__.__name__.lower()
        model_conf = getattr(model, "config", None)
        model_type = (
            str(getattr(model_conf, "model_type", "")).lower() if model_conf is not None else ""
        )

        if model_type in {"bert", "roberta", "distilbert", "gpt2", "llama", "vit", "deit"}:
            return "transformer"
        if (
            "vit" in model_class
            or "bert" in model_class
            or "gpt" in model_class
            or "transformer" in model_class
        ):
            return "transformer"
        elif "resnet" in model_class or "efficientnet" in model_class or "mobilenet" in model_class:
            return "cnn"
        elif "clip" in model_class or "blip" in model_class:
            return "multimodal"
        elif hasattr(model, "encoder") and hasattr(model, "get_input_embeddings"):
            return "transformer"
        else:
            warnings.warn(f"Unknown model type: {model_class}, defaulting to 'transformer'")
            return "transformer"

    def _register_hooks(self):
        """Register forward hooks to capture intermediate outputs."""
        named_modules = dict(self.model.named_modules())

        for layer_name in self.layer_names:
            if layer_name not in named_modules:
                warnings.warn(f"Layer '{layer_name}' not found in model, skipping")
                continue

            layer = named_modules[layer_name]

            if self.model_type == "cnn":
                hook = layer.register_forward_hook(self._get_feature_hook(layer_name))
            elif self.model_type == "transformer":
                hook = layer.register_forward_hook(self._get_attention_hook(layer_name))
            elif self.model_type == "multimodal":
                hook = layer.register_forward_hook(self._get_multimodal_hook(layer_name))
            else:
                hook = layer.register_forward_hook(self._get_feature_hook(layer_name))

            self.hooks.append(hook)

    def _get_feature_hook(self, name: str) -> Callable:
        """Hook for CNN feature maps."""

        def hook(module, input, output):
            self.feature_maps[name] = output

        return hook

    def _get_attention_hook(self, name: str) -> Callable:
        """Hook for Transformer attention scores."""

        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple) and len(output) > 0:
                # Some transformers return (output, attention_weights)
                if len(output) > 1 and output[1] is not None:
                    self.attention_scores[name] = output[1]
                self.feature_maps[name] = output[0]
            elif isinstance(output, torch.Tensor):
                self.feature_maps[name] = output

        return hook

    def _get_multimodal_hook(self, name: str) -> Callable:
        """Hook for multimodal models (self + cross attention)."""

        def hook(module, input, output):
            if hasattr(output, "attentions"):
                # Extract both self and cross attention if available
                self.attention_scores[f"{name}_self"] = output.attentions
                if hasattr(output, "cross_attentions"):
                    self.attention_scores[f"{name}_cross"] = output.cross_attentions
            self.feature_maps[name] = output if not isinstance(output, tuple) else output[0]

        return hook

    def extract_attention_maps(self, model_output: Any = None) -> Dict[str, torch.Tensor]:
        """
        Extract and compute attention maps from stored features.

        Returns:
            Dictionary mapping layer names to attention maps
        """
        attention_maps = {}

        # For CNNs: compute spatial attention from feature maps
        if self.model_type == "cnn":
            for name, feat_map in self.feature_maps.items():
                if feat_map.dim() == 4:  # [B, C, H, W]
                    attention_maps[name] = feat_map.pow(2).mean(dim=1)  # Spatial attention

        # For Transformers: use extracted attention scores
        elif self.model_type in ["transformer", "multimodal"]:
            attention_maps = self.attention_scores.copy()

        return attention_maps

    def clear(self):
        """Clear stored features and attentions."""
        self.feature_maps.clear()
        self.attention_scores.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# ============================================================================
# 2. ATTENTION MATCHER
# ============================================================================


class AttentionMatcher:
    """
    Aligns and normalizes attention maps between teacher and student.

    Features:
    - Resize maps for mismatched resolutions
    - Multiple normalization strategies (L2, softmax, sigmoid)
    - Layer correlation and matching
    - Temporal alignment for video models
    """

    def __init__(
        self,
        normalization: str = "softmax",
        interpolation_mode: str = "bilinear",
        layer_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            normalization: "l2", "softmax", "sigmoid", or "none"
            interpolation_mode: "nearest", "bilinear", "bicubic"
            layer_mapping: Maps teacher layers to student layers (e.g., {"layer4": "layer2"})
        """
        self.normalization = normalization
        self.interpolation_mode = interpolation_mode
        self.layer_mapping = layer_mapping or {}

    def normalize(self, attention_map: torch.Tensor) -> torch.Tensor:
        """Apply normalization to attention map."""
        if self.normalization == "l2":
            return F.normalize(attention_map.view(attention_map.size(0), -1), p=2, dim=1)
        elif self.normalization == "softmax":
            flat = attention_map.view(attention_map.size(0), -1)
            return F.softmax(flat, dim=-1)
        elif self.normalization == "sigmoid":
            return torch.sigmoid(attention_map)
        else:  # "none"
            return attention_map

    def resize(self, student_map: torch.Tensor, teacher_map: torch.Tensor) -> torch.Tensor:
        """Resize student attention map to match teacher dimensions."""
        if student_map.shape == teacher_map.shape:
            return student_map

        # Handle different tensor dimensions
        if student_map.dim() == 2:  # [B, L]
            # Sequence length mismatch - interpolate
            student_map = student_map.unsqueeze(1)  # [B, 1, L]
            resized = F.interpolate(student_map, size=teacher_map.size(1), mode="linear")
            return resized.squeeze(1)

        elif student_map.dim() == 3:  # [B, H, W] or [B, N_heads, L]
            if student_map.size(1) == student_map.size(2):  # Square attention [B, L, L]
                # Multi-head attention matrix
                resized = F.interpolate(
                    student_map.unsqueeze(1),
                    size=(teacher_map.size(1), teacher_map.size(2)),
                    mode="bilinear",
                )
                return resized.squeeze(1)
            else:
                # Spatial attention [B, H, W]
                resized = F.interpolate(
                    student_map.unsqueeze(1),
                    size=(teacher_map.size(1), teacher_map.size(2)),
                    mode=self.interpolation_mode,
                )
                return resized.squeeze(1)

        elif student_map.dim() == 4:  # [B, N_heads, L, L] or [B, C, H, W]
            if student_map.size(-1) == student_map.size(-2):  # Attention matrix
                resized = F.interpolate(
                    student_map.view(-1, 1, student_map.size(-2), student_map.size(-1)),
                    size=(teacher_map.size(-2), teacher_map.size(-1)),
                    mode="bilinear",
                )
                return resized.view(
                    student_map.size(0),
                    student_map.size(1),
                    teacher_map.size(-2),
                    teacher_map.size(-1),
                )
            else:
                # Feature map
                resized = F.interpolate(
                    student_map,
                    size=(teacher_map.size(2), teacher_map.size(3)),
                    mode=self.interpolation_mode,
                )
                return resized

        return student_map

    def match_layers(
        self,
        teacher_attentions: Dict[str, torch.Tensor],
        student_attentions: Dict[str, torch.Tensor],
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """
        Match corresponding layers between teacher and student.

        Returns:
            List of (layer_name, teacher_attn, student_attn) tuples
        """
        matched_pairs = []

        for teacher_layer, teacher_attn in teacher_attentions.items():
            # Check if we have an explicit mapping
            student_layer = self.layer_mapping.get(teacher_layer, teacher_layer)

            if student_layer in student_attentions:
                student_attn = student_attentions[student_layer]

                # Resize student to match teacher
                student_attn_resized = self.resize(student_attn, teacher_attn)

                # Normalize both
                teacher_attn_norm = self.normalize(teacher_attn)
                student_attn_norm = self.normalize(student_attn_resized)

                matched_pairs.append((teacher_layer, teacher_attn_norm, student_attn_norm))

        return matched_pairs


# ============================================================================
# 3. ATTENTION LOSS COMPOSER
# ============================================================================


class AttentionLossComposer:
    """
    Flexible attention loss computation with multiple formulations.

    Supports:
    - L2 distance (base AT)
    - KL divergence (probabilistic)
    - Contrastive loss (for embeddings/cross-modal)
    - Relational loss (Gram matrix matching)
    """

    def __init__(
        self,
        loss_types: List[str] = ["l2"],
        weights: Optional[List[float]] = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            loss_types: List from ["l2", "kl", "contrastive", "relational"]
            weights: Weight for each loss type (default: equal weights)
            temperature: Temperature for KL divergence
        """
        self.loss_types = loss_types
        self.weights = weights or [1.0 / len(loss_types)] * len(loss_types)
        self.temperature = temperature

        if len(self.weights) != len(self.loss_types):
            raise ValueError("Number of weights must match number of loss types")

    def l2_loss(self, student_attn: torch.Tensor, teacher_attn: torch.Tensor) -> torch.Tensor:
        """L2 distance between attention maps."""
        return F.mse_loss(student_attn, teacher_attn)

    def kl_loss(self, student_attn: torch.Tensor, teacher_attn: torch.Tensor) -> torch.Tensor:
        """KL divergence (probabilistic matching)."""
        # Ensure valid probability distributions
        student_log_prob = F.log_softmax(
            student_attn.view(student_attn.size(0), -1) / self.temperature, dim=-1
        )
        teacher_prob = F.softmax(
            teacher_attn.view(teacher_attn.size(0), -1) / self.temperature, dim=-1
        )
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (
            self.temperature**2
        )

    def contrastive_loss(
        self, student_attn: torch.Tensor, teacher_attn: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss for cross-modal attention."""
        # Flatten and normalize
        student_flat = F.normalize(student_attn.view(student_attn.size(0), -1), p=2, dim=1)
        teacher_flat = F.normalize(teacher_attn.view(teacher_attn.size(0), -1), p=2, dim=1)

        # Cosine similarity
        similarity = (student_flat * teacher_flat).sum(dim=1)

        # Maximize similarity (minimize negative)
        return (1 - similarity).mean()

    def relational_loss(
        self, student_attn: torch.Tensor, teacher_attn: torch.Tensor
    ) -> torch.Tensor:
        """Relational loss using Gram matrices."""
        # Compute Gram matrices
        student_flat = student_attn.view(student_attn.size(0), -1)
        teacher_flat = teacher_attn.view(teacher_attn.size(0), -1)

        student_gram = torch.matmul(student_flat, student_flat.T)
        teacher_gram = torch.matmul(teacher_flat, teacher_flat.T)

        # Frobenius norm of difference
        return F.mse_loss(student_gram, teacher_gram)

    def compute(self, student_attn: torch.Tensor, teacher_attn: torch.Tensor) -> torch.Tensor:
        """Compute weighted combination of losses."""
        total_loss = torch.tensor(0.0, device=student_attn.device, requires_grad=True)

        for loss_type, weight in zip(self.loss_types, self.weights):
            if loss_type == "l2":
                loss = self.l2_loss(student_attn, teacher_attn)
            elif loss_type == "kl":
                loss = self.kl_loss(student_attn, teacher_attn)
            elif loss_type == "contrastive":
                loss = self.contrastive_loss(student_attn, teacher_attn)
            elif loss_type == "relational":
                loss = self.relational_loss(student_attn, teacher_attn)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            total_loss = total_loss + weight * loss

        return total_loss


# ============================================================================
# 4. MAIN DISTILLER (ENHANCED)
# ============================================================================


class AttentionTransferDistiller(BaseDistiller):
    """
    Multi-Mode Attention Transfer (Zynthe-AT) - ENHANCED
    ------------------------------------------------
    A unified attention transfer framework supporting:

    CLASSICAL METHODS:
    - Classical Spatial Attention (Zagoruyko & Komodakis, 2017)
    - Relational / Affinity Attention (A2T, RKD)
    - Self-Attention Matching (SAD)
    - Spatial–Channel Split Attention (SCAT)
    - Probabilistic Attention Transfer (PAT)

    ADVANCED METHODS (NEW):
    - Attention Rollout (Transformer interpretability)
    - Cross-layer Attention Flow
    - Dual Attention Matching (feature + token space)
    - Temporal Attention Transfer (video models)

    CONFIGURATION:
    Supports dynamic configuration via config.yaml:
        attention_transfer:
          enabled: true
          type: ["spatial", "self", "relational", "contrastive"]
          layers: ["block3", "block4"]
          normalization: "softmax"
          weight: 0.25
          loss_types: ["l2", "kl"]
          use_attention_rollout: true
          use_dual_matching: true
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        alpha: Union[float, Dict[str, Any]] = 1.0,
        mode: str = "hybrid",
        temperature: float = 1.0,
        use_attention_rollout: bool = False,
        use_dual_matching: bool = False,
        use_cross_layer_flow: bool = False,
        use_temporal_attention: bool = False,
        # New parameters for advanced features
        teacher_layers: Optional[List[str]] = None,
        student_layers: Optional[List[str]] = None,
        layer_mapping: Optional[Dict[str, str]] = None,
        normalization: str = "softmax",
        loss_types: Optional[List[str]] = None,
        loss_weights: Optional[List[float]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        parsed_config: Dict[str, Any] = {}

        if isinstance(alpha, dict):
            parsed_config.update(alpha)
            alpha = parsed_config.get("weight", parsed_config.get("alpha", 1.0))
            mode = parsed_config.get("mode", parsed_config.get("type", mode))
            temperature = parsed_config.get("temperature", temperature)
            use_attention_rollout = parsed_config.get(
                "use_attention_rollout", use_attention_rollout
            )
            use_dual_matching = parsed_config.get("use_dual_matching", use_dual_matching)
            use_cross_layer_flow = parsed_config.get("use_cross_layer_flow", use_cross_layer_flow)
            use_temporal_attention = parsed_config.get(
                "use_temporal_attention", use_temporal_attention
            )
            teacher_layers = parsed_config.get("teacher_layers", teacher_layers)
            student_layers = parsed_config.get("student_layers", student_layers)
            layer_mapping = parsed_config.get("layer_mapping", layer_mapping)
            normalization = parsed_config.get("normalization", normalization)
            loss_types = parsed_config.get("loss_types", loss_types)
            loss_weights = parsed_config.get("loss_weights", loss_weights)

        if config:
            cfg_block = config.get("attention_transfer", config)
            parsed_config.update(cfg_block)
            alpha = parsed_config.get("weight", alpha)
            mode = parsed_config.get("mode", parsed_config.get("type", mode))
            temperature = parsed_config.get("temperature", temperature)
            use_attention_rollout = parsed_config.get(
                "use_attention_rollout", use_attention_rollout
            )
            use_dual_matching = parsed_config.get("use_dual_matching", use_dual_matching)
            use_cross_layer_flow = parsed_config.get("use_cross_layer_flow", use_cross_layer_flow)
            use_temporal_attention = parsed_config.get(
                "use_temporal_attention", use_temporal_attention
            )
            teacher_layers = parsed_config.get("teacher_layers", teacher_layers)
            student_layers = parsed_config.get("student_layers", student_layers)
            layer_mapping = parsed_config.get("layer_mapping", layer_mapping)
            normalization = parsed_config.get("normalization", normalization)
            loss_types = parsed_config.get("loss_types", loss_types)
            loss_weights = parsed_config.get("loss_weights", loss_weights)

        parsed_config = self._validate_attention_config(parsed_config)

        self.config = parsed_config

        super().__init__(teacher, student, parsed_config or None)

        self.alpha = alpha if isinstance(alpha, (float, int)) else 1.0
        self.mode = mode
        self.temperature = temperature
        self.use_attention_rollout = use_attention_rollout
        self.use_dual_matching = use_dual_matching
        self.use_cross_layer_flow = use_cross_layer_flow
        self.use_temporal_attention = use_temporal_attention
        self.entropy_regularizer = parsed_config.get("entropy_regularizer", 0.0)

        # Initialize extractors
        self.teacher_layers = teacher_layers or []
        self.student_layers = student_layers or []

        auto_detect_layers = parsed_config.get("auto_detect_layers", True)
        if auto_detect_layers:
            if not self.teacher_layers:
                self.teacher_layers = self._auto_detect_attention_layers(self.teacher)
            if not self.student_layers:
                self.student_layers = self._auto_detect_attention_layers(self.student)

        if not layer_mapping and self.teacher_layers and self.student_layers:
            layer_mapping = self._build_relative_layer_mapping(
                self.teacher_layers, self.student_layers
            )

        if not self.teacher_layers or not self.student_layers:
            warnings.warn(
                "Attention layer hooks are partially/unconfigured; output-attention based losses will still run. "
                "Set teacher_layers/student_layers or keep auto_detect_layers=true for hook alignment."
            )

        # Validate that every requested layer actually exists on its model.
        # Default stays backward-compatible (warn), but a strict mode raises.
        self.strict_layer_match = bool(
            parsed_config.get(
                "strict_layer_match",
                config.get("strict_layer_match", False) if config else False,
            )
        )
        self._validate_attention_layers(strict=self.strict_layer_match)

        if self.teacher_layers:
            self.teacher_extractor = AttentionExtractor(teacher, self.teacher_layers)
        else:
            self.teacher_extractor = None  # type: ignore[assignment]

        if self.student_layers:
            self.student_extractor = AttentionExtractor(student, self.student_layers)
        else:
            self.student_extractor = None  # type: ignore[assignment]

        # Initialize matcher
        interp_mode = parsed_config.get("interpolation_mode", "bilinear")
        self.matcher = AttentionMatcher(
            normalization=normalization, interpolation_mode=interp_mode, layer_mapping=layer_mapping
        )

        # Initialize loss composer
        default_loss_types = ["l2"] if not loss_types else loss_types
        self.loss_composer = AttentionLossComposer(
            loss_types=default_loss_types, weights=loss_weights, temperature=temperature
        )

    def _validate_attention_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize attention transfer config."""
        validated = dict(config or {})

        allowed_norms = {"l2", "softmax", "sigmoid", "none"}
        normalization = validated.get("normalization", "softmax")
        if normalization not in allowed_norms:
            warnings.warn(f"Invalid normalization '{normalization}'. Falling back to 'softmax'.")
            validated["normalization"] = "softmax"

        allowed_loss_types = {"l2", "kl", "contrastive", "relational"}
        loss_types = validated.get("loss_types")
        if loss_types is not None:
            if isinstance(loss_types, str):
                loss_types = [loss_types]
            filtered_loss_types = [lt for lt in loss_types if lt in allowed_loss_types]
            if not filtered_loss_types:
                warnings.warn("No valid loss_types found. Falling back to ['l2'].")
                filtered_loss_types = ["l2"]
            validated["loss_types"] = filtered_loss_types

        mode = validated.get("mode", validated.get("type", "hybrid"))
        allowed_modes = {"spatial", "affinity", "probabilistic", "scat", "hybrid"}
        if mode not in allowed_modes:
            warnings.warn(f"Invalid attention mode '{mode}'. Falling back to 'hybrid'.")
            validated["mode"] = "hybrid"

        for key in ("teacher_layers", "student_layers"):
            layers = validated.get(key)
            if isinstance(layers, str):
                validated[key] = [layers]

        weights = validated.get("loss_weights")
        if isinstance(weights, list) and validated.get("loss_types"):
            if len(weights) != len(validated["loss_types"]):
                warnings.warn(
                    "loss_weights length does not match loss_types. Ignoring custom loss_weights."
                )
                validated.pop("loss_weights", None)

        return validated

    def _validate_attention_layers(self, *, strict: bool) -> None:
        """Verify every requested ``teacher_layer`` / ``student_layer`` exists.

        Mirrors the :class:`FeatureDistiller` behavior: warnings by default,
        :class:`~zynthe.core.utils.ConfigError` when ``strict=True``.
        """
        teacher_modules = dict(self.teacher.named_modules())
        student_modules = dict(self.student.named_modules())

        missing: list[str] = []
        for name in self.teacher_layers:
            if name not in teacher_modules:
                missing.append(f"teacher:{name}")
        for name in self.student_layers:
            if name not in student_modules:
                missing.append(f"student:{name}")

        if not missing:
            return

        if strict:
            from zynthe.core.utils import ConfigError, format_missing_layers

            raise ConfigError(
                "AttentionTransferDistiller could not find requested layers",
                context={
                    "missing": missing,
                    "missing_summary": format_missing_layers(missing),
                    "hint": (
                        "Use model.named_modules() to find valid attention "
                        "layer names, or set strict_layer_match=False."
                    ),
                },
            )
        warnings.warn(
            f"[AttentionTransferDistiller] skipping unmatched layers: {missing}. "
            "Set strict_layer_match=True to raise instead.",
            stacklevel=2,
        )

    def _auto_detect_attention_layers(self, model: nn.Module, max_layers: int = 4) -> List[str]:
        """Auto-detect likely attention layers for hook-based matching."""
        named_modules = list(model.named_modules())
        exact_suffixes = (
            "attention.self",
            "self_attn",
            "self_attention",
            "attn",
        )

        candidates: List[str] = []
        for name, _ in named_modules:
            lname = name.lower()
            if lname.endswith(exact_suffixes):
                candidates.append(name)

        if not candidates:
            for name, _ in named_modules:
                lname = name.lower()
                if "attention" in lname and (
                    "encoder" in lname or "layer" in lname or "block" in lname
                ):
                    candidates.append(name)

        if not candidates:
            return []

        if len(candidates) <= max_layers:
            return candidates

        # Select layers uniformly across depth.
        step = max(1, len(candidates) // max_layers)
        selected = [candidates[i] for i in range(0, len(candidates), step)]
        return selected[-max_layers:]

    def _build_relative_layer_mapping(
        self, teacher_layers: List[str], student_layers: List[str]
    ) -> Dict[str, str]:
        """Build relative-depth layer mapping when teacher/student layer names differ."""
        if not teacher_layers or not student_layers:
            return {}

        mapping: Dict[str, str] = {}
        if len(teacher_layers) == len(student_layers):
            for t_name, s_name in zip(teacher_layers, student_layers):
                mapping[t_name] = s_name
            return mapping

        ratio = len(teacher_layers) / max(len(student_layers), 1)
        for idx, s_name in enumerate(student_layers):
            t_idx = min(len(teacher_layers) - 1, int(round(idx * ratio)))
            mapping[teacher_layers[t_idx]] = s_name
        return mapping

    def prepare_for_forward_pass(self) -> None:
        """Reset extractor state before each teacher/student forward pair."""
        if self.teacher_extractor:
            self.teacher_extractor.clear()
        if self.student_extractor:
            self.student_extractor.clear()

    def _move_to_device(self, data: Any) -> Any:
        """Recursively move tensors (or collections of tensors) to the distiller device."""
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            moved = [self._move_to_device(v) for v in data]
            return type(data)(moved) if isinstance(data, tuple) else moved
        return data

    # ---------------------- Classical Attention Computation ----------------------

    def compute_spatial_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Classical Spatial Attention (AT)."""
        if features.dim() == 4:
            att_map = features.pow(2).mean(dim=1)
        elif features.dim() == 3:
            att_map = features.pow(2).mean(dim=-1)
        else:
            raise ValueError(f"Unsupported feature dim: {features.shape}")
        att_map = F.normalize(att_map.view(att_map.size(0), -1), p=2, dim=1)
        return att_map

    def compute_affinity_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Relational Attention (A2T/RKD): Cosine relational structure."""
        norm_feats = F.normalize(features.view(features.size(0), -1), p=2, dim=1)
        return norm_feats @ norm_feats.T

    def compute_self_attention(self, att_matrices: torch.Tensor) -> torch.Tensor:
        """Self-Attention Based (Transformer attention heads)."""
        return F.normalize(att_matrices.mean(dim=1), p=2, dim=-1)

    def compute_spatial_channel_attention(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Spatial–Channel Split Attention (SCAT)."""
        if features.dim() != 4:
            raise ValueError("SCAT expects 4D convolutional features")
        spatial = features.pow(2).mean(dim=1)
        channel = features.pow(2).mean(dim=(2, 3))
        return {
            "spatial": F.normalize(spatial.view(spatial.size(0), -1), p=2, dim=1),
            "channel": F.normalize(channel, p=2, dim=1),
        }

    def compute_prob_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Probabilistic Attention (PAT)."""
        att_map = self.compute_spatial_attention(features)
        prob = F.softmax(att_map / self.temperature, dim=-1)
        return prob

    # ---------------------- Advanced Methods (NEW) ----------------------

    def attention_rollout(
        self, attn_weights: List[torch.Tensor], residual: bool = True
    ) -> torch.Tensor:
        """
        Attention Rollout (Transformer Interpretability)

        Aggregates multi-head attention across layers to trace information flow.
        Reference: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)

        Args:
            attn_weights: List of attention weight tensors from each transformer layer
            residual: Whether to include residual connections in rollout

        Returns:
            Aggregated attention map [B, seq_len, seq_len]
        """
        if not attn_weights:
            raise ValueError("No attention weights provided for rollout")

        # Start with identity matrix for residual connections
        batch_size, num_heads, seq_len, _ = attn_weights[0].shape

        if residual:
            # Add identity to account for residual connections
            eye = torch.eye(seq_len, device=attn_weights[0].device).unsqueeze(0).unsqueeze(0)
            result = (attn_weights[0] + eye) / 2
        else:
            result = attn_weights[0]

        # Average over heads for first layer
        result = result.mean(dim=1)  # [B, seq_len, seq_len]

        # Roll through subsequent layers
        for i in range(1, len(attn_weights)):
            attn_layer = attn_weights[i].mean(dim=1)  # Average over heads

            if residual:
                eye = torch.eye(seq_len, device=attn_layer.device).unsqueeze(0)
                attn_layer = (attn_layer + eye) / 2

            # Matrix multiplication to accumulate attention flow
            result = torch.matmul(attn_layer, result)

        return result

    def cross_layer_attention_flow(
        self, teacher_attentions: List[torch.Tensor], student_attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Cross-layer Attention Flow

        Propagates teacher's attention backward to guide earlier student layers.
        Uses gradient flow to align attention patterns across depth.

        Args:
            teacher_attentions: List of teacher attention tensors (deep to shallow)
            student_attentions: List of student attention tensors (deep to shallow)

        Returns:
            Flow alignment loss
        """
        if len(teacher_attentions) != len(student_attentions):
            # If layer counts differ, match based on relative depth
            teacher_step = len(teacher_attentions) / len(student_attentions)
            matched_teacher = [
                teacher_attentions[int(i * teacher_step)] for i in range(len(student_attentions))
            ]
            teacher_attentions = matched_teacher

        # Initialize flow_loss as tensor
        device = student_attentions[0].device
        flow_loss = torch.tensor(0.0, device=device, requires_grad=True)
        propagated_teacher = teacher_attentions[-1].mean(dim=1)  # Start from last layer

        # Backward flow through layers
        for i in range(len(student_attentions) - 1, -1, -1):
            student_attn = student_attentions[i].mean(dim=1)

            # Resize if needed
            if propagated_teacher.shape != student_attn.shape:
                propagated_teacher = self.matcher.resize(propagated_teacher, student_attn)

            # Compute alignment loss
            flow_loss = flow_loss + F.mse_loss(student_attn, propagated_teacher)

            # Propagate to earlier layer (if not the first)
            if i > 0:
                teacher_attn = teacher_attentions[i - 1].mean(dim=1)
                propagated_teacher = torch.matmul(teacher_attn, propagated_teacher)

        num_layers = len(student_attentions)
        return flow_loss / num_layers if num_layers > 0 else flow_loss

    def dual_attention_matching(
        self, teacher_features: Dict[str, torch.Tensor], student_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Dual Attention Matching (Feature + Token Space)

        Combines:
        1. Feature-space attention (spatial/channel activations)
        2. Token-space attention (self-attention weights)

        Very useful for multimodal KD where both visual features and
        text tokens need alignment.

        Args:
            teacher_features: Dict with "feature_map" and "attn_matrix"
            student_features: Dict with "feature_map" and "attn_matrix"

        Returns:
            Combined dual attention loss
        """
        losses = []

        # 1. Feature-space attention (spatial)
        if "feature_map" in teacher_features and "feature_map" in student_features:
            teacher_feat_attn = self.compute_spatial_attention(teacher_features["feature_map"])
            student_feat_attn = self.compute_spatial_attention(student_features["feature_map"])

            # Resize and compute loss
            student_feat_attn = self.matcher.resize(student_feat_attn, teacher_feat_attn)
            feature_loss = self.loss_composer.compute(student_feat_attn, teacher_feat_attn)
            losses.append(feature_loss)

        # 2. Token-space attention (self-attention)
        if "attn_matrix" in teacher_features and "attn_matrix" in student_features:
            teacher_token_attn = self.compute_self_attention(teacher_features["attn_matrix"])
            student_token_attn = self.compute_self_attention(student_features["attn_matrix"])

            # Resize and compute loss
            student_token_attn = self.matcher.resize(student_token_attn, teacher_token_attn)
            token_loss = self.loss_composer.compute(student_token_attn, teacher_token_attn)
            losses.append(token_loss)

        # 3. Cross-attention (for multimodal models)
        if "cross_attn" in teacher_features and "cross_attn" in student_features:
            teacher_cross = teacher_features["cross_attn"].mean(dim=1)  # Average over heads
            student_cross = student_features["cross_attn"].mean(dim=1)

            student_cross = self.matcher.resize(student_cross, teacher_cross)
            cross_loss = self.loss_composer.compute(student_cross, teacher_cross)
            losses.append(cross_loss)

        if not losses:
            warnings.warn("No valid attention features found for dual matching")
            return torch.tensor(0.0, device=next(self.student.parameters()).device)

        # Use torch.stack to maintain tensor type
        return torch.stack(losses).mean()

    def temporal_attention_transfer(
        self, teacher_temporal_attns: torch.Tensor, student_temporal_attns: torch.Tensor
    ) -> torch.Tensor:
        """
        Temporal Attention Transfer (for Video Models)

        Aligns temporal attention weights across video frames/time steps.
        Handles the time axis in addition to spatial dimensions.

        Args:
            teacher_temporal_attns: [B, T, H, W] or [B, T, L] temporal attention
            student_temporal_attns: [B, T', H', W'] or [B, T', L'] temporal attention

        Returns:
            Temporal alignment loss
        """
        # Handle temporal dimension mismatch
        if teacher_temporal_attns.size(1) != student_temporal_attns.size(1):
            # Interpolate along time axis by flattening spatial dims first
            b, t_s, *spatial = student_temporal_attns.shape
            student_flat = student_temporal_attns.reshape(b, t_s, -1).transpose(1, 2)  # [B, S, T]
            student_flat = student_flat.reshape(-1, 1, 1, t_s)  # [B*S, 1, 1, T]
            student_resampled = (
                F.interpolate(
                    student_flat,
                    size=(1, teacher_temporal_attns.size(1)),
                    mode="bilinear",
                    align_corners=False,
                )
                .reshape(b, -1, teacher_temporal_attns.size(1))
                .transpose(1, 2)
            )
            student_temporal_attns = student_resampled.reshape(
                b, teacher_temporal_attns.size(1), *spatial
            )

        # Handle spatial dimension mismatch
        if teacher_temporal_attns.shape[2:] != student_temporal_attns.shape[2:]:
            # Reshape for interpolation: [B*T, C, H, W]
            B, T = teacher_temporal_attns.size(0), teacher_temporal_attns.size(1)
            student_reshaped = student_temporal_attns.reshape(
                B * T, 1, *student_temporal_attns.shape[2:]
            )
            target_shape = teacher_temporal_attns.shape[2:]

            interp_mode = "bilinear" if len(target_shape) == 2 else "linear"
            student_resized = F.interpolate(
                student_reshaped, size=target_shape, mode=interp_mode, align_corners=False
            )
            student_temporal_attns = student_resized.reshape(B, T, *target_shape)

        # Normalize temporal attention
        teacher_norm = self.matcher.normalize(teacher_temporal_attns)
        student_norm = self.matcher.normalize(student_temporal_attns)

        # Compute temporal consistency loss
        return self.loss_composer.compute(student_norm, teacher_norm)

    # ---------------------- Loss Computation ----------------------

    def compute_loss(
        self,
        student_outputs: Any,
        teacher_outputs: Any,
        targets: Optional[torch.Tensor] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute unified AT loss with all enabled methods.

        Dynamically combines classical and advanced attention transfer methods
        based on configuration and available features.

        Returns:
            Tuple of (total_loss, loss_dict) for compatibility with BaseDistiller
        """
        # Convert outputs to dict format for internal processing
        teacher_feats = dict(self._output_to_dict(teacher_outputs) if not isinstance(teacher_outputs, dict) else teacher_outputs)
        student_feats = dict(self._output_to_dict(student_outputs) if not isinstance(student_outputs, dict) else student_outputs)

        # Merge with provided features if available
        if teacher_features:
            teacher_feats.update(teacher_features)
        if student_features:
            student_feats.update(student_features)

        losses = []
        loss_details: Dict[str, float] = {}

        # Classical Methods (backward compatible)
        if self.mode in ["spatial", "hybrid"]:
            if teacher_feats.get("last_hidden_state") is not None and student_feats.get("last_hidden_state") is not None:
                t_map = self.compute_spatial_attention(teacher_feats["last_hidden_state"])
                s_map = self.compute_spatial_attention(student_feats["last_hidden_state"])
                s_map = self.matcher.resize(s_map, t_map)
                losses.append(self.loss_composer.compute(s_map, t_map))

        if self.mode in ["affinity", "hybrid"]:
            if teacher_feats.get("last_hidden_state") is not None and student_feats.get("last_hidden_state") is not None:
                t_aff = self.compute_affinity_attention(teacher_feats["last_hidden_state"])
                s_aff = self.compute_affinity_attention(student_feats["last_hidden_state"])
                losses.append(self.loss_composer.compute(s_aff, t_aff))

        if self.mode in ["probabilistic", "hybrid"]:
            if teacher_feats.get("last_hidden_state") is not None and student_feats.get("last_hidden_state") is not None:
                t_prob = self.compute_prob_attention(teacher_feats["last_hidden_state"])
                s_prob = self.compute_prob_attention(student_feats["last_hidden_state"])
                s_prob = self.matcher.resize(s_prob, t_prob)
                losses.append(self.loss_composer.compute(s_prob, t_prob))

        if self.mode in ["scat", "hybrid"]:
            if "feature_map" in teacher_feats and "feature_map" in student_feats:
                try:
                    t_scat = self.compute_spatial_channel_attention(teacher_feats["feature_map"])
                    s_scat = self.compute_spatial_channel_attention(student_feats["feature_map"])
                    spatial_loss = self.loss_composer.compute(s_scat["spatial"], t_scat["spatial"])
                    channel_loss = self.loss_composer.compute(s_scat["channel"], t_scat["channel"])
                    losses.append((spatial_loss + channel_loss) / 2)
                except ValueError:
                    pass  # Skip if not 4D features

        # Advanced Methods (NEW)

        # 1. Attention Rollout
        if (
            self.use_attention_rollout
            and "attentions" in teacher_feats
            and "attentions" in student_feats
        ):
            teacher_attns = teacher_feats["attentions"]
            student_attns = student_feats["attentions"]

            if isinstance(teacher_attns, (list, tuple)) and isinstance(
                student_attns, (list, tuple)
            ):
                rollout_t = self.attention_rollout(list(teacher_attns))
                rollout_s = self.attention_rollout(list(student_attns))
                rollout_s = self.matcher.resize(rollout_s, rollout_t)
                losses.append(self.loss_composer.compute(rollout_s, rollout_t))

        # 2. Cross-layer Attention Flow
        if (
            self.use_cross_layer_flow
            and "attentions" in teacher_feats
            and "attentions" in student_feats
        ):
            teacher_attns = teacher_feats["attentions"]
            student_attns = student_feats["attentions"]

            if isinstance(teacher_attns, (list, tuple)) and isinstance(
                student_attns, (list, tuple)
            ):
                flow_loss = self.cross_layer_attention_flow(
                    list(teacher_attns), list(student_attns)
                )
                losses.append(flow_loss)

        # 3. Dual Attention Matching
        if self.use_dual_matching:
            dual_loss = self.dual_attention_matching(teacher_feats, student_feats)
            losses.append(dual_loss)

        # 4. Temporal Attention (if temporal dimensions exist)
        if self.use_temporal_attention:
            if "temporal_attn" in teacher_feats and "temporal_attn" in student_feats:
                temporal_loss = self.temporal_attention_transfer(
                    teacher_feats["temporal_attn"], student_feats["temporal_attn"]
                )
                losses.append(temporal_loss)

        if self.entropy_regularizer > 0 and isinstance(
            student_feats.get("attentions"), (list, tuple)
        ):
            entropy_penalty = torch.zeros((), device=self.device)
            attentions_seq = student_feats["attentions"]
            for attn in attentions_seq:
                prob = F.softmax(attn, dim=-1)
                entropy_penalty = entropy_penalty + (
                    -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
                )
            entropy_penalty = entropy_penalty / max(len(attentions_seq), 1)
            losses.append(self.entropy_regularizer * entropy_penalty)
            loss_details["attention_entropy"] = entropy_penalty.detach().item()

        # Extract attention maps from hooks (if extractors are configured)
        if self.teacher_extractor and self.student_extractor:
            teacher_attention_maps = self.teacher_extractor.extract_attention_maps()
            student_attention_maps = self.student_extractor.extract_attention_maps()

            # Match and align layers
            matched_pairs = self.matcher.match_layers(
                teacher_attention_maps, student_attention_maps
            )

            for layer_name, teacher_attn, student_attn in matched_pairs:
                layer_loss = self.loss_composer.compute(student_attn, teacher_attn)
                losses.append(layer_loss)

        if not losses:
            # Fallback: basic spatial attention
            warnings.warn("No attention features available, using basic spatial attention")
            if teacher_feats.get("last_hidden_state") is not None and student_feats.get("last_hidden_state") is not None:
                t_map = self.compute_spatial_attention(teacher_feats["last_hidden_state"])
                s_map = self.compute_spatial_attention(student_feats["last_hidden_state"])
                s_map = self.matcher.resize(s_map, t_map)
                loss = self.alpha * F.mse_loss(s_map, t_map)
                self.prepare_for_forward_pass()
                return loss, {"attention_transfer": loss.item(), **loss_details}
            else:
                loss = torch.tensor(0.0, device=next(self.student.parameters()).device, requires_grad=True)
                self.prepare_for_forward_pass()
                return loss, {"attention_transfer": 0.0, **loss_details}

        total_loss = self.alpha * torch.stack(losses).mean()
        loss_details["attention_transfer"] = total_loss.item()
        self.prepare_for_forward_pass()
        return total_loss, loss_details

    # ---------------------- Evaluation Metrics ----------------------

    def compute_attention_alignment_score(
        self,
        teacher_attentions: Dict[str, torch.Tensor],
        student_attentions: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Measure attention alignment between teacher and student.

        Returns metrics:
        - cosine_similarity: Average cosine similarity across layers
        - l2_distance: Average L2 distance
        - kl_divergence: Average KL divergence
        - correlation: Pearson correlation coefficient
        """
        matched_pairs = self.matcher.match_layers(teacher_attentions, student_attentions)

        if not matched_pairs:
            return {
                "cosine_similarity": 0.0,
                "l2_distance": float("inf"),
                "kl_divergence": float("inf"),
                "correlation": 0.0,
            }

        cosine_sims = []
        l2_dists = []
        kl_divs = []
        correlations = []

        for layer_name, teacher_attn, student_attn in matched_pairs:
            # Flatten for metrics
            t_flat = teacher_attn.view(-1)
            s_flat = student_attn.view(-1)

            # Cosine similarity
            cosine_sim = F.cosine_similarity(t_flat.unsqueeze(0), s_flat.unsqueeze(0))
            cosine_sims.append(cosine_sim.item())

            # L2 distance
            l2_dist = torch.norm(t_flat - s_flat, p=2)
            l2_dists.append(l2_dist.item())

            # KL divergence (with softmax normalization)
            t_prob = F.softmax(t_flat, dim=0)
            s_prob = F.softmax(s_flat, dim=0)
            kl_div = F.kl_div(s_prob.log(), t_prob, reduction="sum")
            kl_divs.append(kl_div.item())

            # Pearson correlation
            t_mean = t_flat.mean()
            s_mean = s_flat.mean()
            t_centered = t_flat - t_mean
            s_centered = s_flat - s_mean
            corr = (t_centered * s_centered).sum() / (
                torch.sqrt((t_centered**2).sum() * (s_centered**2).sum()) + 1e-8
            )
            correlations.append(corr.item())

        return {
            "cosine_similarity": sum(cosine_sims) / len(cosine_sims),
            "l2_distance": sum(l2_dists) / len(l2_dists),
            "kl_divergence": sum(kl_divs) / len(kl_divs),
            "correlation": sum(correlations) / len(correlations),
            "num_layers": len(matched_pairs),
        }

    def compute_interpretability_score(
        self, student_attentions: torch.Tensor, gradients: torch.Tensor
    ) -> float:
        """
        Compute Grad-CAM style interpretability score.

        Measures how well student attention aligns with gradient-based importance.
        Higher score = better interpretability.

        Args:
            student_attentions: Student attention maps
            gradients: Gradients w.r.t. student predictions

        Returns:
            Interpretability score (0-1)
        """
        # Normalize both to [0, 1]
        attn_norm = (student_attentions - student_attentions.min()) / (
            student_attentions.max() - student_attentions.min() + 1e-8
        )
        grad_norm = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

        # Compute alignment (cosine similarity)
        attn_flat = attn_norm.view(-1)
        grad_flat = grad_norm.view(-1)

        similarity = F.cosine_similarity(attn_flat.unsqueeze(0), grad_flat.unsqueeze(0))

        # Convert to 0-1 range (from -1 to 1)
        score = (similarity.item() + 1) / 2

        return score

    def visualize_attention_comparison(
        self,
        teacher_attentions: Dict[str, torch.Tensor],
        student_attentions: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Visualize attention map comparison between teacher and student.

        Generates side-by-side heatmaps for debugging.
        Requires matplotlib (optional dependency).

        Args:
            teacher_attentions: Teacher attention maps
            student_attentions: Student attention maps
            save_path: Optional path to save visualization

        Returns:
            Matplotlib figure if available, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available, skipping visualization")
            return None

        matched_pairs = self.matcher.match_layers(teacher_attentions, student_attentions)

        if not matched_pairs:
            warnings.warn("No matched layers for visualization")
            return None

        num_layers = len(matched_pairs)
        fig, axes = plt.subplots(num_layers, 2, figsize=(10, 4 * num_layers))

        if num_layers == 1:
            axes = axes.reshape(1, -1)

        for idx, (layer_name, teacher_attn, student_attn) in enumerate(matched_pairs):
            # Convert to numpy and take first sample
            t_np = teacher_attn[0].detach().cpu().numpy()
            s_np = student_attn[0].detach().cpu().numpy()

            # Plot teacher
            im1 = axes[idx, 0].imshow(t_np, cmap="viridis", aspect="auto")
            axes[idx, 0].set_title(f"Teacher - {layer_name}")
            axes[idx, 0].axis("off")
            plt.colorbar(im1, ax=axes[idx, 0], fraction=0.046)

            # Plot student
            im2 = axes[idx, 1].imshow(s_np, cmap="viridis", aspect="auto")
            axes[idx, 1].set_title(f"Student - {layer_name}")
            axes[idx, 1].axis("off")
            plt.colorbar(im2, ax=axes[idx, 1], fraction=0.046)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Attention comparison saved to: {save_path}")
        return fig

    # ---------------------- Forward Pass ----------------------

    def forward(
        self, inputs: Any, return_features: bool = False, return_loss: bool = False, **kwargs
    ) -> Any:  # type: ignore[override]
        """
        Forward pass with comprehensive attention-based distillation.

        Automatically extracts and aligns attention from both models,
        applies all enabled distillation methods, and optionally returns
        evaluation metrics.

        Note: This override has a different signature than BaseDistiller.forward()
        for attention-specific functionality.
        """
        inputs = self._move_to_device(inputs)
        device_kwargs = self._move_to_device(kwargs)
        if isinstance(device_kwargs, dict):
            device_kwargs.pop("return_features", None)
            device_kwargs.pop("return_loss", None)
            device_kwargs.pop("output_attentions", None)
            device_kwargs.pop("output_hidden_states", None)

        def _run(model, model_inputs):
            if isinstance(model_inputs, dict):
                return model(
                    output_attentions=True,
                    output_hidden_states=True,
                    **model_inputs,
                    **(device_kwargs or {}),
                )
            return model(
                model_inputs,
                output_attentions=True,
                output_hidden_states=True,
                **(device_kwargs or {}),
            )

        # Forward pass through student
        student_out = _run(self.student, inputs)

        # Forward pass through teacher (no gradients)
        with torch.no_grad():
            teacher_out = _run(self.teacher, inputs)

        if return_loss and self.training:
            # Convert outputs to dict format for compatibility
            teacher_feats = self._output_to_dict(teacher_out)
            student_feats = self._output_to_dict(student_out)

            # Compute comprehensive attention loss (returns tuple)
            loss_tensor, loss_dict = self.compute_loss(
                student_outputs=student_out,
                teacher_outputs=teacher_out,
                student_features=student_feats,
                teacher_features=teacher_feats,
            )

            return loss_tensor

        if return_features:
            teacher_feats = self._output_to_dict(teacher_out)
            student_feats = self._output_to_dict(student_out)
            return student_out, teacher_out, teacher_feats, student_feats

        return student_out, teacher_out

    def _output_to_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        """Convert model output to standardized dict format."""
        result = {}

        if hasattr(model_output, "last_hidden_state"):
            result["last_hidden_state"] = model_output.last_hidden_state

        if hasattr(model_output, "hidden_states") and model_output.hidden_states is not None:
            result["hidden_states"] = model_output.hidden_states

        if hasattr(model_output, "attentions") and model_output.attentions is not None:
            result["attentions"] = model_output.attentions

        if hasattr(model_output, "cross_attentions") and model_output.cross_attentions is not None:
            result["cross_attentions"] = model_output.cross_attentions

        # For backward compatibility
        if isinstance(model_output, dict):
            result.update(model_output)

        return result

    def evaluate_attention_quality(
        self, dataloader: Any, device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of attention transfer quality.

        Measures:
        - Attention alignment scores (cosine, L2, KL)
        - Layer-wise correlation
        - Interpretability score (optional Grad-CAM)

        Args:
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on

        Returns:
            Dictionary with comprehensive attention metrics
        """
        self.teacher.eval()
        self.student.eval()

        if device is None:
            device = next(self.student.parameters()).device

        all_alignment_scores = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                else:
                    inputs = batch.to(device)

                # Forward pass
                teacher_out = self.teacher(
                    **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
                    output_attentions=True,
                    output_hidden_states=True,
                )
                student_out = self.student(
                    **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
                    output_attentions=True,
                    output_hidden_states=True,
                )

                # Extract attentions
                teacher_feats = self._output_to_dict(teacher_out)
                student_feats = self._output_to_dict(student_out)

                # Extract attention maps
                teacher_attentions = {}
                student_attentions = {}

                if "attentions" in teacher_feats and teacher_feats["attentions"]:
                    for i, attn in enumerate(teacher_feats["attentions"]):
                        teacher_attentions[f"layer_{i}"] = attn.mean(dim=1)  # Average over heads

                if "attentions" in student_feats and student_feats["attentions"]:
                    for i, attn in enumerate(student_feats["attentions"]):
                        student_attentions[f"layer_{i}"] = attn.mean(dim=1)

                # Compute alignment scores
                if teacher_attentions and student_attentions:
                    alignment = self.compute_attention_alignment_score(
                        teacher_attentions, student_attentions
                    )
                    all_alignment_scores.append(alignment)

        # Aggregate results
        if all_alignment_scores:
            avg_alignment = {
                key: sum(score[key] for score in all_alignment_scores) / len(all_alignment_scores)
                for key in all_alignment_scores[0].keys()
            }
        else:
            avg_alignment = {
                "cosine_similarity": 0.0,
                "l2_distance": float("inf"),
                "kl_divergence": float("inf"),
                "correlation": 0.0,
                "num_layers": 0,
            }

        return {
            "alignment_scores": avg_alignment,
            "num_batches_evaluated": len(all_alignment_scores),
        }

    @classmethod
    def from_config(
        cls, teacher: nn.Module, student: nn.Module, config: Dict[str, Any]
    ) -> "AttentionTransferDistiller":
        """
        Factory method to create AttentionTransferDistiller from config dict.

        Supports configuration like:
        ```yaml
        attention_transfer:
          enabled: true
          type: ["spatial", "self", "relational"]
          layers: ["layer.3", "layer.4"]
          normalization: "softmax"
          weight: 0.25
          temperature: 2.0
          loss_types: ["l2", "kl"]
          loss_weights: [0.7, 0.3]
          use_attention_rollout: true
          use_dual_matching: true
          use_cross_layer_flow: false
          use_temporal_attention: false
        ```

        Args:
            teacher: Teacher model
            student: Student model
            config: Configuration dictionary

        Returns:
            Configured AttentionTransferDistiller instance
        """
        at_config = config.get("attention_transfer", {})

        # Parse mode from type list
        mode_types = at_config.get("type", ["spatial"])
        if isinstance(mode_types, str):
            mode = mode_types
        elif len(mode_types) == 1:
            mode = mode_types[0]
        else:
            mode = "hybrid"  # Multiple types = hybrid mode

        return cls(
            teacher=teacher,
            student=student,
            alpha=at_config.get("weight", 1.0),
            mode=mode,
            temperature=at_config.get("temperature", 1.0),
            use_attention_rollout=at_config.get("use_attention_rollout", False),
            use_dual_matching=at_config.get("use_dual_matching", False),
            use_cross_layer_flow=at_config.get("use_cross_layer_flow", False),
            use_temporal_attention=at_config.get("use_temporal_attention", False),
            teacher_layers=at_config.get("teacher_layers"),
            student_layers=at_config.get("student_layers", at_config.get("layers")),
            layer_mapping=at_config.get("layer_mapping"),
            normalization=at_config.get("normalization", "softmax"),
            loss_types=at_config.get("loss_types", ["l2"]),
            loss_weights=at_config.get("loss_weights"),
        )

    def __del__(self):
        """Cleanup: remove hooks when distiller is destroyed."""
        if hasattr(self, "teacher_extractor") and self.teacher_extractor:
            self.teacher_extractor.remove_hooks()
        if hasattr(self, "student_extractor") and self.student_extractor:
            self.student_extractor.remove_hooks()
