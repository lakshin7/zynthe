"""
Similarity Transfer - Relational Knowledge Distillation

Transfers structural relationships between samples from teacher to student.
Goes beyond individual features to capture geometric and semantic relationships.

Key Features:
1. Pairwise Similarity Matrices - Captures sample relationships
2. Cross-Modality Alignment - For multimodal models (CLIP-style)
3. Progressive Layer Transfer - Builds hierarchical relationships
4. Graph-based Similarity - Research-grade structural distillation
5. Integration with Multi-Stage Pipeline - Stage 3 distillation

Mathematical Foundation:
    L_sim = ||S_t - S_s||²_F
    where S_t, S_s are teacher/student similarity matrices

    S = normalize(F) @ normalize(F)^T
    F = feature embeddings [batch_size, feature_dim]

Example:
    >>> distiller = SimilarityTransfer(teacher, student, {
    ...     'layer': 8,
    ...     'similarity_metric': 'cosine',
    ...     'weight': 0.6,
    ...     'progressive': True
    ... })
    >>> loss = distiller.compute_loss(inputs, labels)
"""

from __future__ import annotations


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import warnings

from .base_distiller import BaseDistiller
import logging

logger = logging.getLogger(__name__)


class SimilarityTransfer(BaseDistiller):
    """
    Similarity Transfer Distiller - Relational Knowledge Distillation.

    Teaches the student to preserve semantic relationships between samples,
    not just individual predictions. This captures the "geometric soul" of
    the teacher's understanding.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Similarity Transfer distiller.

        Args:
            teacher: Teacher model
            student: Student model
            config: Configuration dictionary with keys:
                - layer: Layer index to extract features from (default: -1)
                - layers: List of layers for multi-layer similarity (optional)
                - similarity_metric: 'cosine', 'euclidean', or 'graph' (default: 'cosine')
                - weight: Loss weight (default: 1.0)
                - temperature: Temperature for softmax normalization (default: 1.0)
                - progressive: Enable progressive layer transfer (default: False)
                - cross_modality: Enable cross-modal similarity (default: False)
                - graph_mode: Enable graph-based similarity (default: False)
                - kd_weight: Weight for KD loss if combined (default: 0.3)
                - normalize: Normalize features before similarity (default: True)
        """
        if config is None:
            config = {}
        transfer_cfg = config.get("similarity_transfer", {})
        if transfer_cfg:
            base_cfg = transfer_cfg
        else:
            base_cfg = config

        # Set config attributes BEFORE calling super().__init__()
        # because BaseDistiller calls _register_hooks() which needs these
        self.config = config
        self.layer = base_cfg.get("layer", -1)
        layers_cfg = base_cfg.get("layers", [self.layer] if self.layer != -1 else [])
        self.layers = layers_cfg
        self.similarity_metric = base_cfg.get("similarity_metric", "cosine")
        self.weight = base_cfg.get("weight", 1.0)
        self.weight_schedule = base_cfg.get("weight_schedule")
        self.temperature = base_cfg.get("temperature", 1.0)
        self.progressive = base_cfg.get("progressive", False)
        self.cross_modality = base_cfg.get("cross_modality", False)
        self.graph_mode = base_cfg.get("graph_mode", False)
        self.graph_threshold = base_cfg.get("graph_threshold", 0.0)
        self.kd_weight = base_cfg.get("kd_weight", 0.3)
        self.normalize = base_cfg.get("normalize", True)
        self.cross_modality_weight = base_cfg.get("cross_modality_weight", 0.5)
        self.use_hidden_state_fallback = base_cfg.get("fallback_to_hidden_states", True)
        self.auto_layer_strategy = base_cfg.get("auto_layers")
        self.auto_layer_count = base_cfg.get("auto_layer_count", 2)

        # Progressive training state
        self.current_epoch = 0
        self.total_epochs = base_cfg.get("total_epochs", 100)
        self.progressive_epochs = base_cfg.get("progressive_epochs", 3)
        self.current_layers = [self.layers[0]] if self.progressive and self.layers else self.layers

        # Feature extraction hooks
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}

        # Metrics tracking
        self.structural_alignment_scores: List[float] = []

        if (not self.layers) and self.auto_layer_strategy:
            self.layers = self._infer_auto_layers(self.auto_layer_strategy, self.auto_layer_count)
            self.current_layers = (
                [self.layers[0]] if self.progressive and self.layers else self.layers
            )

        # Now call super().__init__() which will call _register_hooks()
        super().__init__(teacher, student, config=config, device=device)

        # Adapters must be created after nn.Module initialization
        self.adapters = nn.ModuleDict()

        logger.info("[SIM] Similarity Transfer initialized:")
        logger.info(f"   Metric: {self.similarity_metric}")
        logger.info(f"   Layers: {self.layers}")
        logger.info(f"   Progressive: {self.progressive}")
        logger.info(f"   Cross-modality: {self.cross_modality}")
        logger.info(f"   Graph mode: {self.graph_mode}")

    def _get_scheduled_weight(self) -> float:
        schedule = self.weight_schedule
        if not schedule:
            return float(self.weight)

        if isinstance(schedule, str):
            schedule = {"type": schedule}

        sched_type = str(schedule.get("type", "linear")).lower()
        start = float(schedule.get("start", schedule.get("from", self.weight)))
        end = float(schedule.get("end", self.weight))

        total_epochs = max(1, int(self.total_epochs or 1))
        epoch = max(1, int(self.current_epoch or 1))
        warmup_epochs = int(schedule.get("warmup_epochs", 0))

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return start * float(epoch) / float(max(warmup_epochs, 1))

        if total_epochs <= 1:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, float(epoch - 1) / float(total_epochs - 1)))

        if sched_type == "cosine":
            weight = start + (end - start) * (1.0 - math.cos(math.pi * progress)) / 2.0
        elif sched_type == "step":
            step_epoch = int(schedule.get("step_epoch", max(1, total_epochs // 2)))
            weight = end if epoch >= step_epoch else start
        else:
            weight = start + (end - start) * progress

        min_weight = schedule.get("min", schedule.get("min_weight"))
        max_weight = schedule.get("max", schedule.get("max_weight"))
        if min_weight is not None:
            weight = max(weight, float(min_weight))
        if max_weight is not None:
            weight = min(weight, float(max_weight))
        return float(weight)

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""

        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output

            return hook

        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output

            return hook

        # Register hooks for specified layers by name
        for layer_name in self.layers:
            if isinstance(layer_name, str) and layer_name.startswith("hidden:"):
                continue  # Hidden-state shorthand handled dynamically
            # Get module by name (e.g., "layer_5" or "transformer.layer.5")
            try:
                teacher_module = dict(self.teacher.named_modules()).get(layer_name)
                if teacher_module is not None:
                    teacher_module.register_forward_hook(get_teacher_hook(layer_name))
                else:
                    warnings.warn(f"Teacher layer '{layer_name}' not found")

                student_module = dict(self.student.named_modules()).get(layer_name)
                if student_module is not None:
                    student_module.register_forward_hook(get_student_hook(layer_name))
                else:
                    warnings.warn(f"Student layer '{layer_name}' not found")
            except Exception as e:
                warnings.warn(f"Failed to register hook for layer '{layer_name}': {e}")

    def _infer_auto_layers(self, strategy: str, count: int) -> List[str]:
        """Infer layer identifiers when user opts into auto selection."""
        count = max(1, int(count))
        strategy = strategy.lower()
        if strategy in ("last", "tail", "default"):
            return [f"hidden:-{idx + 1}" for idx in range(count)]
        if strategy in ("first", "head"):
            return [f"hidden:{idx}" for idx in range(count)]
        if strategy == "mixed":
            layers = ["hidden:0", "hidden:-1"]
            if count > 2:
                mid = count - 2
                layers.extend([f"hidden:-{idx + 2}" for idx in range(mid)])
            return layers[:count]
        if strategy == "uniform":
            return [f"hidden:-{idx + 1}" for idx in range(count)]
        if strategy == "attn":
            # Attention heads typically live in the last few hidden states
            return [f"hidden:-{idx + 1}" for idx in range(count)]
        return [f"hidden:-{idx + 1}" for idx in range(count)]

    @staticmethod
    def _extract_hidden_states(output: Any) -> Optional[Tuple[torch.Tensor, ...]]:
        if isinstance(output, dict):
            return output.get("hidden_states")
        return getattr(output, "hidden_states", None)

    @staticmethod
    def _extract_logits(output: Any) -> Any:
        if isinstance(output, dict) and "logits" in output:
            logits = output["logits"]
        elif getattr(output, "logits", None) is not None:
            logits = output.logits
        elif isinstance(output, tuple) and len(output) > 0:
            logits = output[0]
        else:
            logits = output
        # Upcast to float32 and clamp inf/nan for stable loss computation
        if isinstance(logits, torch.Tensor):
            if logits.dtype != torch.float32:
                logits = logits.float()
            if torch.isinf(logits).any() or torch.isnan(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits

    def _ensure_adapter(
        self,
        key: str,
        student_dim: int,
        teacher_dim: int,
        *,
        adapter_type: str,
    ) -> nn.Module:
        if key in self.adapters:
            return self.adapters[key]

        if adapter_type == "conv2d":
            adapter: nn.Module = nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False)
        else:
            adapter = nn.Linear(student_dim, teacher_dim, bias=False)

        student_dtype = self._get_model_dtype(self.student)
        adapter = adapter.to(device=self.device, dtype=student_dtype)
        self.adapters[key] = adapter
        return adapter

    def _align_features(
        self, teacher_feats: torch.Tensor, student_feats: torch.Tensor, layer_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align teacher/student feature shapes (seq length, spatial, channel)."""
        if teacher_feats.shape == student_feats.shape:
            return teacher_feats, student_feats

        # CNN-style features
        if teacher_feats.dim() == 4 and student_feats.dim() == 4:
            if teacher_feats.shape[2:] != student_feats.shape[2:]:
                student_feats = F.interpolate(
                    student_feats, size=teacher_feats.shape[2:], mode="bilinear", align_corners=False
                )
            if teacher_feats.shape[1] != student_feats.shape[1]:
                key = f"{layer_name}_conv2d_{student_feats.shape[1]}_{teacher_feats.shape[1]}"
                adapter = self._ensure_adapter(
                    key,
                    student_feats.shape[1],
                    teacher_feats.shape[1],
                    adapter_type="conv2d",
                )
                student_feats = adapter(student_feats)
            return teacher_feats, student_feats

        # Transformer-style features
        if teacher_feats.dim() == 3 and student_feats.dim() == 3:
            if teacher_feats.shape[1] != student_feats.shape[1]:
                student_feats = F.interpolate(
                    student_feats.transpose(1, 2),
                    size=teacher_feats.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            if teacher_feats.shape[2] != student_feats.shape[2]:
                key = f"{layer_name}_linear_{student_feats.shape[2]}_{teacher_feats.shape[2]}"
                adapter = self._ensure_adapter(
                    key,
                    student_feats.shape[2],
                    teacher_feats.shape[2],
                    adapter_type="linear",
                )
                student_feats = adapter(student_feats)
            return teacher_feats, student_feats

        # Vector features
        if teacher_feats.dim() == 2 and student_feats.dim() == 2:
            if teacher_feats.shape[1] != student_feats.shape[1]:
                key = f"{layer_name}_linear_{student_feats.shape[1]}_{teacher_feats.shape[1]}"
                adapter = self._ensure_adapter(
                    key,
                    student_feats.shape[1],
                    teacher_feats.shape[1],
                    adapter_type="linear",
                )
                student_feats = adapter(student_feats)
            return teacher_feats, student_feats

        return teacher_feats, student_feats

    def _get_feature(
        self,
        layer_name: str,
        cache: Dict[str, torch.Tensor],
        hidden_states: Optional[Tuple[torch.Tensor, ...]],
    ) -> Optional[torch.Tensor]:
        if layer_name in cache:
            return cache[layer_name]
        if layer_name.startswith("hidden:") and hidden_states is not None:
            try:
                index = int(layer_name.split(":", 1)[1])
            except ValueError:
                return None
            return hidden_states[index]
        return None

    def compute_similarity_matrix(
        self, features: torch.Tensor, metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix between samples.

        Args:
            features: Feature tensor [batch_size, feature_dim]
            metric: Similarity metric ('cosine', 'euclidean', 'graph')

        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        # Flatten spatial dimensions if present (for CNNs)
        if features.dim() > 2:
            batch_size = features.size(0)
            features = features.view(batch_size, -1)

        # Normalize features
        if self.normalize:
            features = F.normalize(features, p=2, dim=-1)

        if metric == "cosine":
            # Cosine similarity: normalized dot product
            sim_matrix = torch.matmul(features, features.T)

        elif metric == "euclidean":
            # Euclidean distance (convert to similarity)
            # S = exp(-d²/T)
            dist_sq = torch.cdist(features, features, p=2).pow(2)
            sim_matrix = torch.exp(-dist_sq / self.temperature)

        elif metric == "graph":
            # Graph-based similarity with adaptive threshold
            sim_matrix = torch.matmul(features, features.T)

            # Apply threshold to create sparse adjacency
            if self.graph_threshold and self.graph_threshold > 0:
                threshold = self.graph_threshold
            else:
                threshold = sim_matrix.mean() + sim_matrix.std()
            sim_matrix = torch.where(
                sim_matrix > threshold, sim_matrix, torch.zeros_like(sim_matrix)
            )

        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return sim_matrix

    def compute_similarity_loss(
        self, teacher_feats: torch.Tensor, student_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity transfer loss.

        Args:
            teacher_feats: Teacher features
            student_feats: Student features

        Returns:
            Similarity loss scalar
        """
        # Compute similarity matrices
        teacher_sim = self.compute_similarity_matrix(teacher_feats, self.similarity_metric)
        student_sim = self.compute_similarity_matrix(student_feats, self.similarity_metric)

        # Frobenius norm (MSE for matrices)
        if self.graph_mode:
            # For graph mode, use masked loss on non-zero entries
            mask = (teacher_sim != 0).float()
            loss = F.mse_loss(student_sim * mask, teacher_sim * mask)
        else:
            loss = F.mse_loss(student_sim, teacher_sim)

        return loss

    def compute_cross_modality_loss(
        self,
        teacher_feats_1: torch.Tensor,
        teacher_feats_2: torch.Tensor,
        student_feats_1: torch.Tensor,
        student_feats_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-modal similarity alignment loss.

        For multimodal models (e.g., CLIP), ensures student preserves
        cross-modal relationships.

        Args:
            teacher_feats_1: Teacher features from modality 1 (e.g., vision)
            teacher_feats_2: Teacher features from modality 2 (e.g., text)
            student_feats_1: Student features from modality 1
            student_feats_2: Student features from modality 2

        Returns:
            Cross-modal similarity loss
        """
        # Normalize features
        t1_norm = F.normalize(teacher_feats_1, p=2, dim=-1)
        t2_norm = F.normalize(teacher_feats_2, p=2, dim=-1)
        s1_norm = F.normalize(student_feats_1, p=2, dim=-1)
        s2_norm = F.normalize(student_feats_2, p=2, dim=-1)

        # Cross-modal similarity matrices
        teacher_cross_sim = torch.matmul(t1_norm, t2_norm.T)
        student_cross_sim = torch.matmul(s1_norm, s2_norm.T)

        # Alignment loss
        loss = F.mse_loss(student_cross_sim, teacher_cross_sim)

        return loss

    def get_progressive_layers(self, epoch: int) -> List[str]:
        """
        Get layers to use based on progressive training schedule.

        Starts with shallow layers, gradually adds deeper layers.

        Args:
            epoch: Current epoch

        Returns:
            List of layer names to use
        """
        if not self.progressive or not self.layers:
            return self.layers

        # Calculate how many layers to use based on epoch
        epochs_per_layer = self.progressive_epochs
        num_layers_to_use = min(len(self.layers), 1 + (epoch // epochs_per_layer))

        # Use layers progressively (first to num_layers_to_use)
        return self.layers[:num_layers_to_use]

    def compute_structural_alignment_score(
        self, teacher_sim: torch.Tensor, student_sim: torch.Tensor
    ) -> float:
        """
        Compute Structural Alignment Score (SAS).

        Measures how well student preserves teacher's structural relationships.
        Range: [0, 1], higher is better.

        Args:
            teacher_sim: Teacher similarity matrix
            student_sim: Student similarity matrix

        Returns:
            Alignment score
        """
        # Flatten matrices
        t_flat = teacher_sim.flatten()
        s_flat = student_sim.flatten()

        # Cosine similarity between flattened matrices
        cos_sim = F.cosine_similarity(t_flat.unsqueeze(0), s_flat.unsqueeze(0))

        # Convert to [0, 1] range
        sas = (cos_sim + 1) / 2

        return sas.item()

    def forward(  # type: ignore[override]
        self,
        x: Any,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_features: bool = False,
        **kwargs,
    ) -> Any:
        """
        Forward pass with similarity transfer.

        Note: When ``return_features=True`` this method returns the same
        tuple signature as :meth:`BaseDistiller.forward` to remain compatible
        with the training loop.

        Args:
            x: Input tensor
            labels: Ground truth labels (optional)
            return_dict: Whether to return dictionary (default: True)
            return_features: When True, return (student_outputs, teacher_outputs,
                teacher_features, student_features)

        Returns:
            Dictionary with:
                - loss: Total loss
                - similarity_loss: Similarity transfer loss
                - kd_loss: Knowledge distillation loss (if applicable)
                - sas_score: Structural Alignment Score
                - logits: Student predictions
        """
        if return_features:
            # Ensure hook caches are clean before the base forward pass
            self.teacher_features.clear()
            self.student_features.clear()
            student_out, teacher_out, _, _ = super().forward(
                x, return_features=True, **kwargs
            )
            return student_out, teacher_out, self.teacher_features, self.student_features

        # Clear feature caches
        self.teacher_features.clear()
        self.student_features.clear()

        # Forward passes
        with torch.no_grad():
            teacher_output = self._safe_forward(self.teacher, x, kwargs)
        student_output = self._safe_forward(self.student, x, kwargs)

        # Extract logits based on output type
        teacher_hidden_states = self._extract_hidden_states(teacher_output)
        student_hidden_states = self._extract_hidden_states(student_output)

        teacher_logits = self._extract_logits(teacher_output)
        student_logits = self._extract_logits(student_output)

        # Compute similarity loss
        sim_loss_total = 0.0
        loss_components = {}
        sas_scores = []

        # Get active layers (progressive or all)
        active_layers = self.get_progressive_layers(self.current_epoch)

        # Multi-layer similarity loss
        for layer_name in active_layers:
            t_feats = self._get_feature(layer_name, self.teacher_features, teacher_hidden_states)
            s_feats = self._get_feature(layer_name, self.student_features, student_hidden_states)

            if t_feats is not None and s_feats is not None:
                t_feats, s_feats = self._align_features(t_feats, s_feats, layer_name)
                if t_feats.shape != s_feats.shape:
                    warnings.warn(f"Feature shape mismatch at layer {layer_name}")
                    continue

                # Compute similarity loss for this layer
                sim_loss = self.compute_similarity_loss(t_feats, s_feats)
                sim_loss_total += sim_loss

                loss_components[f"sim_loss_{layer_name}"] = sim_loss.item()

                # Compute and track structural alignment score
                with torch.no_grad():
                    t_sim = self.compute_similarity_matrix(t_feats, self.similarity_metric)
                    s_sim = self.compute_similarity_matrix(s_feats, self.similarity_metric)
                    sas = self.compute_structural_alignment_score(t_sim, s_sim)
                    sas_scores.append(sas)
                    loss_components[f"sas_{layer_name}"] = sas

        # Average across layers
        if len(active_layers) > 0:
            sim_loss_avg = sim_loss_total / len(active_layers)
            avg_sas = sum(sas_scores) / len(sas_scores) if sas_scores else 0.0
        else:
            sim_loss_avg = torch.tensor(0.0, device=x.device)
            avg_sas = 0.0

        # Combine with KD loss if specified
        kd_loss = torch.tensor(0.0, device=x.device)
        ce_loss = torch.tensor(0.0, device=x.device)

        if labels is not None:
            # Cross-entropy with labels
            ce_loss = F.cross_entropy(student_logits, labels)

            # KL divergence with teacher
            T = self.temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)

        # Combined loss
        sim_weight = self._get_scheduled_weight()
        total_loss = sim_weight * sim_loss_avg + self.kd_weight * kd_loss
        if labels is not None:
            ce_weight = max(0.0, 1.0 - sim_weight - self.kd_weight)
            total_loss += ce_weight * ce_loss

        # Return dictionary
        return {
            "loss": total_loss,
            "similarity_loss": (
                sim_loss_avg.item() if isinstance(sim_loss_avg, torch.Tensor) else sim_loss_avg
            ),
            "kd_loss": kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss,
            "ce_loss": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            "sas_score": avg_sas,
            "logits": student_logits,
            "sim_weight": sim_weight,
            **loss_components,
        }

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
        Compute similarity transfer loss (compatible with BaseDistiller).

        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            targets: Ground truth labels
            student_features: Student intermediate features (unused, uses hooks)
            teacher_features: Teacher intermediate features (unused, uses hooks)
            **kwargs: Additional arguments

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        # Extract logits
        teacher_hidden_states = self._extract_hidden_states(teacher_outputs)
        student_hidden_states = self._extract_hidden_states(student_outputs)

        student_logits = self._extract_logits(student_outputs)
        teacher_logits = self._extract_logits(teacher_outputs)
        task_type = self._resolve_task_type(
            student_logits if isinstance(student_logits, torch.Tensor) else None
        )
        ignore_index = int(self.config.get("distillation", {}).get("ignore_index", -100))
        shift_labels = bool(self.config.get("distillation", {}).get("shift_labels", True))

        # Compute similarity loss across registered layers
        sim_loss_total = 0.0
        loss_components = {}
        sas_scores = []

        # Get active layers (progressive or all)
        active_layers = self.get_progressive_layers(self.current_epoch)

        # Multi-layer similarity loss
        for layer_name in active_layers:
            t_feats = self._get_feature(layer_name, self.teacher_features, teacher_hidden_states)
            s_feats = self._get_feature(layer_name, self.student_features, student_hidden_states)

            if t_feats is not None and s_feats is not None:
                t_feats, s_feats = self._align_features(t_feats, s_feats, layer_name)
                if t_feats.shape != s_feats.shape:
                    warnings.warn(f"Feature shape mismatch at layer {layer_name}")
                    continue

                # Compute similarity loss for this layer
                sim_loss = self.compute_similarity_loss(t_feats, s_feats)
                sim_loss_total += sim_loss

                loss_components[f"sim_loss_{layer_name}"] = sim_loss.item()

                # Compute and track structural alignment score
                with torch.no_grad():
                    t_sim = self.compute_similarity_matrix(t_feats, self.similarity_metric)
                    s_sim = self.compute_similarity_matrix(s_feats, self.similarity_metric)
                    sas = self.compute_structural_alignment_score(t_sim, s_sim)
                    sas_scores.append(sas)
                    loss_components[f"sas_{layer_name}"] = sas

        # Average across layers
        if len(active_layers) > 0:
            sim_loss_avg = sim_loss_total / len(active_layers)
            avg_sas = sum(sas_scores) / len(sas_scores) if sas_scores else 0.0
        else:
            device = student_logits.device if hasattr(student_logits, "device") else "cpu"
            sim_loss_avg = torch.tensor(0.0, device=device)
            avg_sas = 0.0

        # Combine with KD loss if specified
        device = student_logits.device if hasattr(student_logits, "device") else "cpu"
        kd_loss = torch.tensor(0.0, device=device)
        ce_loss = torch.tensor(0.0, device=device)

        if targets is not None:
            if (
                task_type == "causal_lm"
                and isinstance(student_logits, torch.Tensor)
                and isinstance(teacher_logits, torch.Tensor)
                and student_logits.dim() == 3
                and teacher_logits.dim() == 3
            ):
                flat_student, flat_targets = self._flatten_lm_logits_and_targets(
                    student_logits,
                    targets,
                    ignore_index=ignore_index,
                    shift_labels=shift_labels,
                )
                flat_teacher, _ = self._flatten_lm_logits_and_targets(
                    teacher_logits,
                    targets,
                    ignore_index=ignore_index,
                    shift_labels=shift_labels,
                )
                if flat_student.numel() > 0:
                    ce_loss = F.cross_entropy(flat_student, flat_targets)
                    T = self.temperature
                    kd_loss = F.kl_div(
                        F.log_softmax(flat_student / T, dim=-1),
                        F.softmax(flat_teacher / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)
                else:
                    ce_loss = torch.zeros((), device=device)
                    kd_loss = torch.zeros((), device=device)
            else:
                # Cross-entropy with labels
                ce_loss = F.cross_entropy(student_logits, targets)

                # KL divergence with teacher
                T = self.temperature
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)

        # Combined loss
        sim_weight = self._get_scheduled_weight()
        total_loss = sim_weight * sim_loss_avg + self.kd_weight * kd_loss
        if targets is not None:
            ce_weight = max(0.0, 1.0 - sim_weight - self.kd_weight)
            total_loss += ce_weight * ce_loss

        # Metrics
        metrics = {
            "similarity_loss": (
                sim_loss_avg.item() if isinstance(sim_loss_avg, torch.Tensor) else sim_loss_avg
            ),
            "kd_loss": kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss,
            "ce_loss": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            "sas_score": avg_sas,
            "sim_weight": sim_weight,
            **loss_components,
        }

        return total_loss, metrics

    def train_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: (inputs, labels) tuple
            optimizer: Optimizer instance

        Returns:
            Dictionary of metrics
        """
        inputs, labels = batch

        optimizer.zero_grad()

        # Forward pass with loss computation
        outputs = self.forward(inputs, labels)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            student_logits = outputs["logits"]
            _, predicted = torch.max(student_logits, 1)
            accuracy = (predicted == labels).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "similarity_loss": outputs["similarity_loss"],
            "kd_loss": outputs["kd_loss"],
            "sas_score": outputs["sas_score"],
        }

        return metrics

    def update_epoch(self, epoch: int):
        """
        Update current epoch for progressive training.

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch

        if self.progressive:
            self.current_layers = self.get_progressive_layers(epoch)
            logger.info(f"  Progressive: Using layers {self.current_layers}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get similarity transfer metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "similarity_metric": self.similarity_metric,
            "layers": self.layers,
            "weight": self.weight,
            "progressive": self.progressive,
            "cross_modality": self.cross_modality,
            "graph_mode": self.graph_mode,
        }

        if self.structural_alignment_scores:
            metrics["mean_sas"] = sum(self.structural_alignment_scores) / len(
                self.structural_alignment_scores
            )
            metrics["final_sas"] = self.structural_alignment_scores[-1]

        return metrics

    @classmethod
    def from_config(
        cls, teacher: nn.Module, student: nn.Module, config: Optional[Dict[str, Any]] = None
    ) -> "SimilarityTransfer":
        return cls(teacher=teacher, student=student, config=config or {})


# Backward compatibility alias
SimilarityTransferDistiller = SimilarityTransfer


def create_similarity_config(
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    similarity_metric: str = "cosine",
    weight: float = 0.6,
    temperature: float = 4.0,
    kd_weight: float = 0.3,
    normalize: bool = True,
    progressive: bool = False,
    progressive_epochs: int = 3,
    cross_modality: bool = False,
    cross_modality_weight: float = 0.5,
    graph_mode: bool = False,
    graph_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Create configuration for similarity transfer.

    Args:
        layer: Single layer name to extract features from
        layers: List of layer names for multi-layer mode
        similarity_metric: 'cosine', 'euclidean', or 'graph'
        weight: Similarity loss weight
        temperature: Temperature for KD loss
        kd_weight: KD loss weight
        normalize: Normalize features before similarity
        progressive: Enable progressive layer transfer
        progressive_epochs: Epochs between layer additions
        cross_modality: Enable cross-modal similarity
        cross_modality_weight: Cross-modal loss weight
        graph_mode: Enable graph-based similarity
        graph_threshold: Threshold for graph edges

    Returns:
        Configuration dictionary
    """
    config = {
        "similarity_metric": similarity_metric,
        "weight": weight,
        "temperature": temperature,
        "kd_weight": kd_weight,
        "normalize": normalize,
        "progressive": progressive,
        "progressive_epochs": progressive_epochs,
        "cross_modality": cross_modality,
        "cross_modality_weight": cross_modality_weight,
        "graph_mode": graph_mode,
        "graph_threshold": graph_threshold,
    }

    # Add layer configuration
    if layers is not None:
        config["layers"] = layers
    elif layer is not None:
        config["layer"] = layer
    else:
        config["layer"] = -1  # Default to last layer

    return config
