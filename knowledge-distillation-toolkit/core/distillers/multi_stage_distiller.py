from typing import List, Dict, Any
import torch
import torch.nn as nn
from .base_distiller import BaseDistiller
from .kd_hinton import KDHintonDistiller
from .attention_transfer import AttentionTransferDistiller
from .feature_distiller import FeatureDistiller
from .similarity_transfer import SimilarityTransferDistiller

class MultiStageDistiller(BaseDistiller):
    """
    Combines multiple distillation strategies intelligently using reasoning rules.
    Distillers are applied based on the config and training context.
    """

    def __init__(self, student: nn.Module, teacher: nn.Module, config: Dict[str, Any]):
        super().__init__(student, teacher)
        self.config = config
        self.distillers: List[BaseDistiller] = []
        self._init_distillers(config)

    def _init_distillers(self, config: Dict[str, Any]):
        """
        Dynamically instantiate distillers based on reasoning over config.
        Rules:
          1. If method is 'kd_hinton', always include KDHintonDistiller.
          2. Include AttentionTransferDistiller if attention_transfer is True.
          3. Include FeatureDistiller if feature_transfer is True.
          4. Include SimilarityTransferDistiller if similarity_transfer is True.
          5. Order: KD -> Attention -> Feature -> Similarity.
        """
        distill_cfg = config.get("distillation", {})
        reasoning_order = []

        if distill_cfg.get("method") == "kd_hinton":
            # KDHintonDistiller has different constructor - create wrapper
            kd_wrapper = lambda s, t, c: KDHintonDistiller(
                temperature=c.get('temperature', 2.0),
                alpha=c.get('alpha', 0.5)
            )
            reasoning_order.append(kd_wrapper)
        if distill_cfg.get("attention_transfer", False):
            reasoning_order.append(AttentionTransferDistiller)
        if distill_cfg.get("feature_transfer", False):
            reasoning_order.append(FeatureDistiller)
        if distill_cfg.get("similarity_transfer", False):
            reasoning_order.append(SimilarityTransferDistiller)

        # Instantiate all selected distillers
        for distiller_cls in reasoning_order:
            self.distillers.append(distiller_cls(self.student, self.teacher, distill_cfg))

    def forward(self, x, labels=None, return_loss=False, training=False):
        """
        Forward pass with reasoning-guided multi-stage distillation.
        Aggregates loss from all active distillers when training.
        """
        student_output = self.student(x)
        total_loss = 0.0

        if training and return_loss:
            for distiller in self.distillers:
                # Use reasoning: dynamically check if distiller can produce a loss
                try:
                    loss = distiller(x, labels=labels, return_loss=True, training=True)
                    if isinstance(loss, torch.Tensor):
                        total_loss += loss
                except TypeError:
                    # fallback in case a distiller does not accept return_loss argument
                    pass
            return student_output, total_loss

        return student_output