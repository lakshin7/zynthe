"""
Feature Distillation - Advanced Multi-Stage Feature Transfer
==============================================================

Implements the 4 stages of feature distillation:
1. Vanilla Feature Regression (L2)
2. Correlation & Similarity Matching (CKA, Cosine, Gram)
3. Attention-Augmented Feature Distillation
4. Advanced Paradigms (FitNets, FSP, AB, VID, CRD)

Architecture:
```
FeatureDistiller (extends BaseDistiller)
 FeatureExtractor (hook system)
 LayerAligner (dimension matching)
 MetricSelector (L2, CKA, Gram, etc.)
 FeatureAdapter (auto 1x1 conv)
 FeatureLossComposer (weighted multi-metric)
 Evaluation Metrics (CKA, MI, cosine)
```

Reference: Zynthe Feature Distillation Blueprint
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_distiller import BaseDistiller

# ============================================================================
# FEATURE ADAPTER - Auto Dimension Alignment
# ============================================================================


class FeatureAdapter(nn.Module):
    """
    Automatic feature dimension adapter using 1x1 convolution.
    Handles channel mismatch between teacher and student layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adapter_type: str = "1x1conv",
        use_bn: bool = False,
    ):
        """
        Args:
            in_channels: Student feature channels
            out_channels: Teacher feature channels (target)
            adapter_type: Type of adapter ('1x1conv', 'linear', 'mlp')
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        self.adapter_type = adapter_type

        if adapter_type == "1x1conv":
            layers: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, 1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            self.adapter = nn.Sequential(*layers)

        elif adapter_type == "linear":
            self.adapter = nn.Linear(in_channels, out_channels, bias=not use_bn)
            if use_bn:
                self.bn = nn.BatchNorm1d(out_channels)

        elif adapter_type == "mlp":
            hidden_dim = (in_channels + out_channels) // 2
            self.adapter = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_channels),
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt feature dimensions."""
        return self.adapter(x)


# ============================================================================
# METRIC FUNCTIONS - Stage 2 Correlation & Similarity
# ============================================================================


class FeatureMetrics:
    """Collection of feature similarity metrics."""

    @staticmethod
    def l2_distance(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """Stage 1: Vanilla L2 loss."""
        return F.mse_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """Stage 2: Cosine similarity loss."""
        # Flatten spatial dimensions
        f_t_flat = f_t.flatten(2)  # [B, C, H*W]
        f_s_flat = f_s.flatten(2)

        # Cosine similarity per sample
        cos_sim = F.cosine_similarity(f_t_flat, f_s_flat, dim=1).mean(dim=1)

        # Convert to loss (1 - similarity)
        return (1 - cos_sim).mean()

    @staticmethod
    def gram_matrix(x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style matching."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    @staticmethod
    def gram_loss(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """Stage 2: Gram matrix matching (style transfer-like)."""
        gram_t = FeatureMetrics.gram_matrix(f_t)
        gram_s = FeatureMetrics.gram_matrix(f_s)
        return F.mse_loss(gram_s, gram_t)

    @staticmethod
    def centered_kernel_alignment(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """
        Stage 2: CKA (Centered Kernel Alignment) - robust similarity metric.

        Reference: "Similarity of Neural Network Representations Revisited" (Kornblith et al., 2019)
        """
        # Flatten to [batch, features]
        f_t_flat = f_t.flatten(1)
        f_s_flat = f_s.flatten(1)

        # Center the features
        f_t_centered = f_t_flat - f_t_flat.mean(dim=0, keepdim=True)
        f_s_centered = f_s_flat - f_s_flat.mean(dim=0, keepdim=True)

        # Compute kernel matrices (linear kernel)
        k_t = f_t_centered @ f_t_centered.T
        k_s = f_s_centered @ f_s_centered.T

        # CKA = HSIC(K_t, K_s) / sqrt(HSIC(K_t, K_t) * HSIC(K_s, K_s))
        hsic_ts = (k_t * k_s).sum()
        hsic_tt = (k_t * k_t).sum()
        hsic_ss = (k_s * k_s).sum()

        cka = hsic_ts / (torch.sqrt(hsic_tt * hsic_ss) + 1e-10)

        # Convert to loss (1 - CKA for maximization)
        return 1 - cka

    @staticmethod
    def fsp_matrix(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Stage 4: FSP Matrix - Flow of Solution Process.
        Captures relationship between two consecutive feature maps.

        Reference: "A Gift from Knowledge Distillation" (Yim et al., 2017)
        """
        # f1: [B, C1, H, W], f2: [B, C2, H, W]
        b, c1, h1, w1 = f1.size()
        _, c2, h2, w2 = f2.size()

        # Resize f2 to match f1 spatial dimensions if needed
        if h1 != h2 or w1 != w2:
            f2 = F.interpolate(f2, size=(h1, w1), mode="bilinear", align_corners=False)

        # Reshape to [B, C, H*W]
        f1_flat = f1.view(b, c1, -1)
        f2_flat = f2.view(b, c2, -1)

        # Compute FSP matrix: [B, C1, C2]
        fsp = torch.bmm(f1_flat, f2_flat.transpose(1, 2)) / (h1 * w1)

        return fsp

    @staticmethod
    def fsp_loss(
        f_t_pair: Tuple[torch.Tensor, torch.Tensor], f_s_pair: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """FSP loss between teacher and student layer pairs."""
        fsp_t = FeatureMetrics.fsp_matrix(f_t_pair[0], f_t_pair[1])
        fsp_s = FeatureMetrics.fsp_matrix(f_s_pair[0], f_s_pair[1])

        # Handle dimension mismatch by flattening and using normalized similarity
        # FSP matrices have shape [B, C1, C2]
        b_t, c1_t, c2_t = fsp_t.shape
        b_s, c1_s, c2_s = fsp_s.shape

        if fsp_t.shape[1:] != fsp_s.shape[1:]:
            # Flatten and normalize both
            fsp_t_flat = fsp_t.view(b_t, -1)  # [B, C1*C2]
            fsp_s_flat = fsp_s.view(b_s, -1)  # [B, C1*C2]

            # Pad smaller one or pool larger one to match
            if fsp_t_flat.shape[1] > fsp_s_flat.shape[1]:
                # Pool teacher to match student
                pool_factor = fsp_t_flat.shape[1] // fsp_s_flat.shape[1]  # noqa: F841
                fsp_t_flat = F.adaptive_avg_pool1d(
                    fsp_t_flat.unsqueeze(1), fsp_s_flat.shape[1]
                ).squeeze(1)
            elif fsp_s_flat.shape[1] > fsp_t_flat.shape[1]:
                # Pool student to match teacher
                fsp_s_flat = F.adaptive_avg_pool1d(
                    fsp_s_flat.unsqueeze(1), fsp_t_flat.shape[1]
                ).squeeze(1)

            # Normalize and compute similarity
            fsp_t_norm = F.normalize(fsp_t_flat, p=2, dim=1)
            fsp_s_norm = F.normalize(fsp_s_flat, p=2, dim=1)

            # Cosine similarity loss
            return (1 - F.cosine_similarity(fsp_t_norm, fsp_s_norm, dim=1)).mean()

        return F.mse_loss(fsp_s, fsp_t)

    @staticmethod
    def activation_boundary_loss(f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """
        Stage 4: AB Distillation - Transfer activation boundary instead of magnitudes.

        Reference: "Knowledge Transfer via Distillation of Activation Boundaries" (Heo et al., 2019)
        """
        # Compute binary activation boundaries (sign)
        ab_t = (f_t > 0).float()

        # Binary cross-entropy on activation boundaries
        return F.binary_cross_entropy_with_logits(f_s, ab_t, reduction="mean")

    @staticmethod
    def contrastive_loss(
        f_t: torch.Tensor, f_s: torch.Tensor, temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Stage 4: CRD - Contrastive Representation Distillation.
        Uses InfoNCE-style contrastive loss.

        Reference: "Contrastive Representation Distillation" (Tian et al., 2020)
        """
        # Flatten features
        f_t_flat = f_t.flatten(2).mean(dim=2)  # [B, C]
        f_s_flat = f_s.flatten(2).mean(dim=2)  # [B, C]

        # Normalize
        f_t_norm = F.normalize(f_t_flat, dim=1)
        f_s_norm = F.normalize(f_s_flat, dim=1)

        # Compute similarity matrix
        logits = torch.mm(f_s_norm, f_t_norm.T) / temperature  # [B, B]

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


# ============================================================================
# FEATURE LOSS COMPOSER - Unified Multi-Metric Loss
# ============================================================================


class FeatureLossComposer(nn.Module):
    """
    Composes multiple feature distillation losses with configurable weights.

    Supports all metrics from Stage 1-4:
    - l2, cosine, gram, cka (Stage 1-2)
    - fsp, ab, contrastive (Stage 4)
    - attention_weighted (Stage 3)
    """

    def __init__(
        self,
        metrics: List[str] = ["l2"],
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 0.07,
        use_attention: bool = False,
    ):
        """
        Args:
            metrics: List of metrics to use ['l2', 'cosine', 'gram', 'cka', 'fsp', 'ab', 'contrastive']
            weights: Weight for each metric (default: equal weights)
            temperature: Temperature for contrastive loss
            use_attention: Whether to use attention-weighted losses (Stage 3)
        """
        super().__init__()
        self.metrics = metrics
        self.weights = weights or {m: 1.0 for m in metrics}
        self.temperature = temperature
        self.use_attention = use_attention

        # Metric function mapping
        self.metric_fns = {
            "l2": FeatureMetrics.l2_distance,
            "cosine": FeatureMetrics.cosine_similarity_loss,
            "gram": FeatureMetrics.gram_loss,
            "cka": FeatureMetrics.centered_kernel_alignment,
            "ab": FeatureMetrics.activation_boundary_loss,
            "contrastive": lambda f_t, f_s: FeatureMetrics.contrastive_loss(
                f_t, f_s, self.temperature
            ),
        }

    def forward(
        self,
        f_t: torch.Tensor,
        f_s: torch.Tensor,
        attention_t: Optional[torch.Tensor] = None,
        attention_s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted feature loss.

        Args:
            f_t: Teacher features
            f_s: Student features
            attention_t: Teacher attention maps (optional, for Stage 3)
            attention_s: Student attention maps (optional, for Stage 3)

        Returns:
            total_loss: Weighted sum of all metric losses
            loss_dict: Individual loss values for logging
        """
        # Stage 3: Attention-weighted features
        if self.use_attention and attention_t is not None and attention_s is not None:
            f_t = attention_t * f_t
            f_s = attention_s * f_s

        total_loss = torch.zeros((), device=f_t.device)
        loss_dict = {}

        for metric in self.metrics:
            if metric not in self.metric_fns:
                warnings.warn(f"Unknown metric: {metric}, skipping")
                continue

            metric_loss = self.metric_fns[metric](f_t, f_s)
            weight = self.weights.get(metric, 1.0)

            total_loss = total_loss + weight * metric_loss
            loss_dict[f"feat_{metric}"] = metric_loss.item()

        loss_dict["feat_total"] = total_loss.item()

        return total_loss, loss_dict


# ============================================================================
# LAYER ALIGNER - Intelligent Layer Matching
# ============================================================================


class LayerAligner:
    """
    Aligns teacher and student layers for feature distillation.
    Handles dimension mismatches and spatial size differences.
    """

    def __init__(
        self,
        layer_pairs: Optional[List[Tuple[str, str]]] = None,
        auto_align: bool = True,
        adapter_type: str = "1x1conv",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            layer_pairs: Explicit teacher-student layer pairs [(t_layer, s_layer), ...]
            auto_align: Auto-detect layer alignment from architecture
            adapter_type: Type of dimension adapter
            device: Device to put adapters on
        """
        self.layer_pairs = layer_pairs or []
        self.auto_align = auto_align
        self.adapter_type = adapter_type
        self.device = device or torch.device("cpu")
        self.adapters: Dict[str, FeatureAdapter] = {}

    def create_adapter(
        self, student_channels: int, teacher_channels: int, layer_name: str
    ) -> FeatureAdapter:
        """Create and cache an adapter for dimension mismatch."""
        if layer_name not in self.adapters:
            adapter = FeatureAdapter(
                in_channels=student_channels,
                out_channels=teacher_channels,
                adapter_type=self.adapter_type,
            )
            adapter = adapter.to(self.device)  # Move to device immediately
            self.adapters[layer_name] = adapter
        return self.adapters[layer_name]

    def align_features(
        self, f_t: torch.Tensor, f_s: torch.Tensor, layer_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align teacher and student features.

        Handles:
        - Spatial dimension mismatch (resize)
        - Channel dimension mismatch (adapter)
        """
        # Handle spatial size mismatch
        if f_t.shape[2:] != f_s.shape[2:]:
            target_shape = f_t.shape[2:]
            if f_t.dim() == 4:
                f_s = F.interpolate(f_s, size=target_shape, mode="bilinear", align_corners=False)
            elif f_t.dim() == 3:
                # Treat sequence length as spatial width for interpolation
                seq = f_s.unsqueeze(-2)  # [B, C, 1, L]
                seq = F.interpolate(
                    seq, size=(1, target_shape[0]), mode="bilinear", align_corners=False
                )
                f_s = seq.squeeze(-2)
            else:
                f_s = F.interpolate(f_s.unsqueeze(1), size=target_shape, mode="nearest").squeeze(1)

        # Handle channel mismatch
        if f_t.shape[1] != f_s.shape[1]:
            adapter = self.create_adapter(
                student_channels=f_s.shape[1], teacher_channels=f_t.shape[1], layer_name=layer_name
            )
            f_s = adapter(f_s)

        return f_t, f_s


# ============================================================================
# MAIN FEATURE DISTILLER - Extends BaseDistiller
# ============================================================================


class FeatureDistiller(BaseDistiller):
    """
    Advanced Feature Distillation supporting all 4 stages:

    Stage 1: Vanilla L2 feature regression
    Stage 2: Correlation matching (CKA, Cosine, Gram)
    Stage 3: Attention-augmented feature distillation
    Stage 4: Advanced paradigms (FitNets, FSP, AB, VID, CRD)

    Integrates seamlessly with Zynthe's BaseDistiller architecture.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initialize feature distiller.

        Config structure:
        ```yaml
        feature_distillation:
          enabled: true
          layers:
            - teacher: "encoder.layer.6"
              student: "encoder.layer.3"
              weight: 1.0
          metrics: ["l2", "cka", "cosine"]
          metric_weights:
            l2: 1.0
            cka: 0.5
            cosine: 0.3
          auto_align: true
          adapter_type: "1x1conv"
          use_attention: false
          fsp_pairs: [[0, 1], [2, 3]]  # Layer pairs for FSP
          contrastive_temperature: 0.07
        ```
        """
        # Parse feature distillation config BEFORE calling super().__init__
        # This ensures attributes are available when _register_hooks() is called
        if config is None:
            config = {}

        feat_config = config.get("feature_distillation", {})
        self.feat_enabled = feat_config.get("enabled", True)
        self.feat_layers = feat_config.get("layers", [])
        # When True, missing layer names raise ConfigError instead of being
        # silently skipped. Defaults to False to preserve historical behavior,
        # but a strict mode is the recommended setting for new configs.
        self.strict_layer_match = bool(
            feat_config.get(
                "strict_layer_match",
                config.get("strict_layer_match", False),
            )
        )

        # Now call parent __init__ which will call _register_hooks()
        super().__init__(teacher, student, config, device, **kwargs)

        # Initialize remaining components
        self.feat_metrics = feat_config.get(
            "metrics", ["l2"]
        )  # Renamed to avoid conflict with parent's self.metrics
        self.metric_weights = feat_config.get("metric_weights", {})
        self.use_attention = feat_config.get("use_attention", False)
        self.fsp_pairs = feat_config.get("fsp_pairs", [])

        # Loss composer
        self.feature_loss_composer = FeatureLossComposer(
            metrics=self.feat_metrics,
            weights=self.metric_weights,
            temperature=feat_config.get("contrastive_temperature", 0.07),
            use_attention=self.use_attention,
        ).to(self.device)

        # Layer aligner with auto-adapters
        self.layer_aligner = LayerAligner(
            auto_align=feat_config.get("auto_align", True),
            adapter_type=feat_config.get("adapter_type", "1x1conv"),
            device=self.device,  # Pass device to aligner
        )

    def _register_hooks(self) -> None:
        """Register hooks for specified feature layers.

        When ``strict_layer_match=True`` (set via
        ``feature_distillation.strict_layer_match`` or top-level
        ``strict_layer_match``), missing ``teacher`` or ``student`` layer names
        raise :class:`~zynthe.core.utils.ConfigError` instead of being
        silently skipped. The error message lists every missing layer so the
        user can fix the config in one round-trip.
        """
        if not self.feat_enabled or not self.feat_layers:
            return

        teacher_modules = dict(self.teacher.named_modules())
        student_modules = dict(self.student.named_modules())

        missing: list[str] = []

        for layer_config in self.feat_layers:
            t_layer_name = layer_config["teacher"]
            s_layer_name = layer_config["student"]

            t_ok = t_layer_name in teacher_modules
            s_ok = s_layer_name in student_modules

            if not t_ok:
                missing.append(f"teacher:{t_layer_name}")
            if not s_ok:
                missing.append(f"student:{s_layer_name}")

            if t_ok:
                t_handle = teacher_modules[t_layer_name].register_forward_hook(
                    self._get_teacher_hook(t_layer_name)
                )
                self._hook_handles.append(t_handle)

            if s_ok:
                s_handle = student_modules[s_layer_name].register_forward_hook(
                    self._get_student_hook(s_layer_name)
                )
                self._hook_handles.append(s_handle)

        if missing and self.strict_layer_match:
            from zynthe.core.utils import ConfigError, format_missing_layers

            raise ConfigError(
                "FeatureDistiller could not find requested layers",
                context={
                    "missing": missing,
                    "hint": (
                        "Use named_modules() on the teacher/student to pick "
                        "valid layer names, or set strict_layer_match=False."
                    ),
                    "missing_summary": format_missing_layers(missing),
                },
            )
        elif missing:
            import warnings

            warnings.warn(
                f"[FeatureDistiller] skipping unmatched layers: {missing}. "
                f"Set strict_layer_match=True to raise instead.",
                stacklevel=2,
            )

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
        Compute feature distillation loss.

        Combines:
        - Supervised loss (if targets provided)
        - Feature matching losses (L2, CKA, etc.)
        - FSP matrix losses (if configured)
        - Attention-weighted losses (if enabled)
        """
        total_loss = torch.zeros((), device=self.device)
        loss_dict: Dict[str, Any] = {}

        # Supervised loss - handle dict, object with logits attr, or tensor
        if targets is not None:
            logits = self._extract_logits_tensor(student_outputs)
            task_type = self._resolve_task_type(logits)

            if task_type == "causal_lm" and logits.dim() == 3:
                ignore_index = int(self.config.get("distillation", {}).get("ignore_index", -100))
                shift_labels = bool(self.config.get("distillation", {}).get("shift_labels", True))
                flat_logits, flat_targets = self._flatten_lm_logits_and_targets(
                    logits,
                    targets,
                    ignore_index=ignore_index,
                    shift_labels=shift_labels,
                )
                if flat_logits.numel() == 0:
                    loss_ce = torch.zeros((), device=self.device)
                else:
                    loss_ce = F.cross_entropy(flat_logits, flat_targets)
                loss_dict["task_type"] = "causal_lm"  # type: ignore[index]
            else:
                loss_ce = F.cross_entropy(logits, targets)
                loss_dict["task_type"] = "classification"  # type: ignore[index]
            total_loss = total_loss + loss_ce
            loss_dict["supervised"] = loss_ce.item()

        # Feature distillation losses
        if self.feat_enabled and teacher_features and student_features:
            feat_loss_total = torch.zeros((), device=self.device)

            for layer_config in self.feat_layers:
                t_layer = layer_config["teacher"]
                s_layer = layer_config["student"]
                weight = layer_config.get("weight", 1.0)

                if t_layer in teacher_features and s_layer in student_features:
                    f_t = teacher_features[t_layer]
                    f_s = student_features[s_layer]

                    # Align dimensions
                    f_t, f_s = self.layer_aligner.align_features(f_t, f_s, s_layer)

                    # Dynamic Layer Weighting based on variance magnitude difference
                    # This increases importance of layers that have higher divergence
                    with torch.no_grad():
                        norm_diff = torch.abs(torch.norm(f_t) - torch.norm(f_s))
                        # Scale weight by relative divergence (clamped to prevent explosion)
                        dynamic_weight = weight * torch.clamp(1.0 + 0.1 * norm_diff, 1.0, 3.0)

                    # Compute loss using composer
                    layer_loss, layer_loss_dict = self.feature_loss_composer(f_t, f_s)

                    feat_loss_total = feat_loss_total + dynamic_weight * layer_loss

                    # Log individual layer losses
                    for key, value in layer_loss_dict.items():
                        loss_dict[f"{s_layer}_{key}"] = value

            total_loss = total_loss + feat_loss_total
            loss_dict["feature_total"] = feat_loss_total.item()

        # FSP matrix losses (Stage 4)
        if self.fsp_pairs and teacher_features and student_features:
            fsp_loss_total = torch.zeros((), device=self.device)

            for idx1, idx2 in self.fsp_pairs:
                if idx1 < len(self.feat_layers) and idx2 < len(self.feat_layers):
                    t_layer1 = self.feat_layers[idx1]["teacher"]
                    t_layer2 = self.feat_layers[idx2]["teacher"]
                    s_layer1 = self.feat_layers[idx1]["student"]
                    s_layer2 = self.feat_layers[idx2]["student"]

                    if all(lyr in teacher_features for lyr in [t_layer1, t_layer2]) and all(
                        lyr in student_features for lyr in [s_layer1, s_layer2]
                    ):
                        fsp_loss = FeatureMetrics.fsp_loss(
                            (teacher_features[t_layer1], teacher_features[t_layer2]),
                            (student_features[s_layer1], student_features[s_layer2]),
                        )
                        fsp_loss_total = fsp_loss_total + fsp_loss

            if fsp_loss_total.item() > 0:
                total_loss = total_loss + fsp_loss_total
                loss_dict["fsp_loss"] = fsp_loss_total.item()

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    @torch.no_grad()
    def compute_feature_alignment_metrics(
        self, teacher_features: Dict[str, torch.Tensor], student_features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for feature alignment.

        Returns:
            Dictionary with CKA, cosine similarity, and MI scores
        """
        metrics = {}

        for layer_config in self.feat_layers:
            t_layer = layer_config["teacher"]
            s_layer = layer_config["student"]

            if t_layer in teacher_features and s_layer in student_features:
                f_t = teacher_features[t_layer]
                f_s = student_features[s_layer]

                # Align first
                f_t, f_s = self.layer_aligner.align_features(f_t, f_s, s_layer)

                # CKA score
                cka_loss = FeatureMetrics.centered_kernel_alignment(f_t, f_s)
                cka_score = 1 - cka_loss.item()
                metrics[f"{s_layer}_cka"] = cka_score

                # Cosine similarity
                cos_loss = FeatureMetrics.cosine_similarity_loss(f_t, f_s)
                cos_score = 1 - cos_loss.item()
                metrics[f"{s_layer}_cosine"] = cos_score

                # L2 distance (normalized)
                l2_dist = FeatureMetrics.l2_distance(f_t, f_s).item()
                metrics[f"{s_layer}_l2"] = l2_dist

        # Average scores
        if metrics:
            metrics["avg_cka"] = np.mean([v for k, v in metrics.items() if "cka" in k])
            metrics["avg_cosine"] = np.mean([v for k, v in metrics.items() if "cosine" in k])

        return metrics


# ============================================================================
# LEGACY COMPATIBILITY WRAPPER
# ============================================================================


class LegacyFeatureDistiller:
    """
    Legacy feature distiller for backward compatibility.
    Wraps the new FeatureDistiller with a simpler interface.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        feature_layers: list,
        loss_fn=nn.MSELoss(),
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.feature_layers = feature_layers
        self.loss_fn = loss_fn
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        for layer_name in self.feature_layers:
            teacher_layer = dict(self.teacher.named_modules())[layer_name]
            student_layer = dict(self.student.named_modules())[layer_name]

            teacher_layer.register_forward_hook(self._get_teacher_hook(layer_name))
            student_layer.register_forward_hook(self._get_student_hook(layer_name))

    def _get_teacher_hook(self, name):
        def hook(module, input, output):
            self.teacher_features[name] = output

        return hook

    def _get_student_hook(self, name):
        def hook(module, input, output):
            self.student_features[name] = output

        return hook

    def compute_loss(self) -> torch.Tensor:
        total_loss: Optional[torch.Tensor] = None
        for layer_name in self.feature_layers:
            teacher_feat = self.teacher_features.get(layer_name)
            student_feat = self.student_features.get(layer_name)
            if teacher_feat is not None and student_feat is not None:
                loss = self.loss_fn(student_feat, teacher_feat.detach())
                total_loss = loss if total_loss is None else total_loss + loss
        if total_loss is None:
            return torch.zeros(())
        return total_loss

    def step(
        self, student_inputs: dict, return_feats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Forward pass student and teacher with inputs and compute feature distillation loss.

        Args:
            student_inputs (dict): Input data for student and teacher models.
            return_feats (bool, optional): If True, also return student and teacher features.
                Useful for multi-stage distillation aggregation. Defaults to False.

        Returns:
            torch.Tensor: The computed distillation loss.
            (optional) dict: Student features.
            (optional) dict: Teacher features.
        """
        # Clear previous features
        self.teacher_features = {}
        self.student_features = {}
        with torch.no_grad():
            _ = self.teacher(**student_inputs)
        _ = self.student(**student_inputs)
        loss = self.compute_loss()
        if return_feats:
            # Return copies to avoid accidental overwrites in aggregation
            student_feats = {k: v for k, v in self.student_features.items()}
            teacher_feats = {k: v for k, v in self.teacher_features.items()}
            return loss, student_feats, teacher_feats
        return loss
