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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import warnings

from .base_distiller import BaseDistiller


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
        config: Dict[str, Any]
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
        # Set config attributes BEFORE calling super().__init__()
        # because BaseDistiller calls _register_hooks() which needs these
        self.config = config
        self.layer = config.get('layer', -1)
        self.layers = config.get('layers', [self.layer] if self.layer != -1 else [])
        self.similarity_metric = config.get('similarity_metric', 'cosine')
        self.weight = config.get('weight', 1.0)
        self.temperature = config.get('temperature', 1.0)
        self.progressive = config.get('progressive', False)
        self.cross_modality = config.get('cross_modality', False)
        self.graph_mode = config.get('graph_mode', False)
        self.kd_weight = config.get('kd_weight', 0.3)
        self.normalize = config.get('normalize', True)
        
        # Progressive training state
        self.current_epoch = 0
        self.total_epochs = config.get('total_epochs', 100)
        self.progressive_epochs = config.get('progressive_epochs', 3)
        self.current_layers = [self.layers[0]] if self.progressive and self.layers else self.layers
        
        # Feature extraction hooks
        self.teacher_features = {}
        self.student_features = {}
        
        # Metrics tracking
        self.structural_alignment_scores = []
        
        # Now call super().__init__() which will call _register_hooks()
        super().__init__(teacher, student)
        
        print(f"🧬 Similarity Transfer initialized:")
        print(f"   Metric: {self.similarity_metric}")
        print(f"   Layers: {self.layers}")
        print(f"   Progressive: {self.progressive}")
        print(f"   Cross-modality: {self.cross_modality}")
        print(f"   Graph mode: {self.graph_mode}")
    
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
            # Get module by name (e.g., "layer_5" or "transformer.layer.5")
            try:
                teacher_module = dict(self.teacher.named_modules()).get(layer_name)
                if teacher_module is not None:
                    teacher_module.register_forward_hook(
                        get_teacher_hook(layer_name)
                    )
                else:
                    warnings.warn(f"Teacher layer '{layer_name}' not found")
                    
                student_module = dict(self.student.named_modules()).get(layer_name)
                if student_module is not None:
                    student_module.register_forward_hook(
                        get_student_hook(layer_name)
                    )
                else:
                    warnings.warn(f"Student layer '{layer_name}' not found")
            except Exception as e:
                warnings.warn(f"Failed to register hook for layer '{layer_name}': {e}")
    
    def compute_similarity_matrix(
        self,
        features: torch.Tensor,
        metric: str = 'cosine'
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
        
        if metric == 'cosine':
            # Cosine similarity: normalized dot product
            sim_matrix = torch.matmul(features, features.T)
            
        elif metric == 'euclidean':
            # Euclidean distance (convert to similarity)
            # S = exp(-d²/T)
            dist_sq = torch.cdist(features, features, p=2).pow(2)
            sim_matrix = torch.exp(-dist_sq / self.temperature)
            
        elif metric == 'graph':
            # Graph-based similarity with adaptive threshold
            sim_matrix = torch.matmul(features, features.T)
            
            # Apply threshold to create sparse adjacency
            threshold = sim_matrix.mean() + sim_matrix.std()
            sim_matrix = torch.where(
                sim_matrix > threshold,
                sim_matrix,
                torch.zeros_like(sim_matrix)
            )
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return sim_matrix
    
    def compute_similarity_loss(
        self,
        teacher_feats: torch.Tensor,
        student_feats: torch.Tensor
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
        student_feats_2: torch.Tensor
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
        num_layers_to_use = min(
            len(self.layers),
            1 + (epoch // epochs_per_layer)
        )
        
        # Use layers progressively (first to num_layers_to_use)
        return self.layers[:num_layers_to_use]
    
    def compute_structural_alignment_score(
        self,
        teacher_sim: torch.Tensor,
        student_sim: torch.Tensor
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
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass with similarity transfer.
        
        Args:
            x: Input tensor
            labels: Ground truth labels (optional)
            return_dict: Whether to return dictionary (default: True)
            
        Returns:
            Dictionary with:
                - loss: Total loss
                - similarity_loss: Similarity transfer loss
                - kd_loss: Knowledge distillation loss (if applicable)
                - sas_score: Structural Alignment Score
                - logits: Student predictions
        """
        # Clear feature caches
        self.teacher_features.clear()
        self.student_features.clear()
        
        # Forward passes
        with torch.no_grad():
            teacher_output = self.teacher(x)
        student_output = self.student(x)
        
        # Extract logits based on output type
        if isinstance(teacher_output, dict):
            teacher_logits = teacher_output['logits']
            student_logits = student_output['logits']
        else:
            teacher_logits = teacher_output
            student_logits = student_output
        
        # Compute similarity loss
        sim_loss_total = 0.0
        loss_components = {}
        sas_scores = []
        
        # Get active layers (progressive or all)
        active_layers = self.get_progressive_layers(self.current_epoch)
        
        # Multi-layer similarity loss
        for layer_name in active_layers:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                t_feats = self.teacher_features[layer_name]
                s_feats = self.student_features[layer_name]
                
                # Ensure matching dimensions
                if t_feats.shape != s_feats.shape:
                    warnings.warn(f"Feature shape mismatch at layer {layer_name}")
                    continue
                
                # Compute similarity loss for this layer
                sim_loss = self.compute_similarity_loss(t_feats, s_feats)
                sim_loss_total += sim_loss
                
                loss_components[f'sim_loss_{layer_name}'] = sim_loss.item()
                
                # Compute and track structural alignment score
                with torch.no_grad():
                    t_sim = self.compute_similarity_matrix(t_feats, self.similarity_metric)
                    s_sim = self.compute_similarity_matrix(s_feats, self.similarity_metric)
                    sas = self.compute_structural_alignment_score(t_sim, s_sim)
                    sas_scores.append(sas)
                    loss_components[f'sas_{layer_name}'] = sas
        
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
                reduction='batchmean'
            ) * (T * T)
        
        # Combined loss
        total_loss = self.weight * sim_loss_avg + self.kd_weight * kd_loss
        if labels is not None:
            total_loss += (1.0 - self.weight - self.kd_weight) * ce_loss
        
        # Return dictionary
        return {
            'loss': total_loss,
            'similarity_loss': sim_loss_avg.item() if isinstance(sim_loss_avg, torch.Tensor) else sim_loss_avg,
            'kd_loss': kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss,
            'ce_loss': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            'sas_score': avg_sas,
            'logits': student_logits,
            **loss_components
        }
    
    def compute_loss(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute similarity transfer loss (compatible with MultiStageDistiller).
        
        Args:
            inputs: Input tensor
            labels: Ground truth labels
            
        Returns:
            Loss tensor
        """
        _, loss = self.forward(inputs, labels, return_loss=True, training=True)
        return loss
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer
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
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            student_logits = outputs['logits']
            _, predicted = torch.max(student_logits, 1)
            accuracy = (predicted == labels).float().mean().item()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'similarity_loss': outputs['similarity_loss'],
            'kd_loss': outputs['kd_loss'],
            'sas_score': outputs['sas_score']
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
            print(f"  Progressive: Using layers {self.current_layers}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get similarity transfer metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'similarity_metric': self.similarity_metric,
            'layers': self.layers,
            'weight': self.weight,
            'progressive': self.progressive,
            'cross_modality': self.cross_modality,
            'graph_mode': self.graph_mode
        }
        
        if self.structural_alignment_scores:
            metrics['mean_sas'] = sum(self.structural_alignment_scores) / len(self.structural_alignment_scores)
            metrics['final_sas'] = self.structural_alignment_scores[-1]
        
        return metrics


# Backward compatibility alias
SimilarityTransferDistiller = SimilarityTransfer


def create_similarity_config(
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    similarity_metric: str = 'cosine',
    weight: float = 0.6,
    temperature: float = 4.0,
    kd_weight: float = 0.3,
    normalize: bool = True,
    progressive: bool = False,
    progressive_epochs: int = 3,
    cross_modality: bool = False,
    cross_modality_weight: float = 0.5,
    graph_mode: bool = False,
    graph_threshold: float = 0.5
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
        'similarity_metric': similarity_metric,
        'weight': weight,
        'temperature': temperature,
        'kd_weight': kd_weight,
        'normalize': normalize,
        'progressive': progressive,
        'progressive_epochs': progressive_epochs,
        'cross_modality': cross_modality,
        'cross_modality_weight': cross_modality_weight,
        'graph_mode': graph_mode,
        'graph_threshold': graph_threshold
    }
    
    # Add layer configuration
    if layers is not None:
        config['layers'] = layers
    elif layer is not None:
        config['layer'] = layer
    else:
        config['layer'] = -1  # Default to last layer
        3
    
    return config
