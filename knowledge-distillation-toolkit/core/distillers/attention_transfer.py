from typing import Optional, Union, Dict, Any
import torch
import torch.nn.functional as F
from core.distillers.base_distiller import BaseDistiller

class AttentionTransferDistiller(BaseDistiller):
    """
    Distiller implementing the Attention Transfer (AT) method.
    Transfers knowledge by matching attention maps between teacher and student.
    """
    def __init__(self, teacher, student, alpha: float = 1.0):
        """
        Args:
            teacher: teacher model
            student: student model
            alpha: weight for attention loss
        """
        super().__init__(teacher, student)
        self.alpha = alpha

    def compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention map from intermediate features.

        Args:
            features: shape (batch_size, channels, height, width) or (batch, seq_len, hidden)

        Returns:
            attention_map: normalized attention map
        """
        if features.dim() == 3:
            # sequence output (batch, seq_len, hidden)
            att_map = features.pow(2).mean(dim=-1)  # average over hidden
        elif features.dim() == 4:
            # conv-like features (batch, channels, H, W)
            att_map = features.pow(2).mean(dim=1)  # average over channels
        else:
            raise ValueError(f"Unsupported feature dimension: {features.shape}")
        # normalize per sample
        att_map = F.normalize(att_map.view(att_map.size(0), -1), p=2, dim=1)
        return att_map

    def compute_loss(self, teacher_feats: torch.Tensor, student_feats: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute attention transfer loss between teacher and student.

        Args:
            teacher_feats: features from teacher model
            student_feats: features from student model

        Returns:
            Attention transfer loss tensor
        """
        teacher_map = self.compute_attention_map(teacher_feats)
        student_map = self.compute_attention_map(student_feats)
        loss = F.mse_loss(student_map, teacher_map)
        return self.alpha * loss

    def forward(self, x: torch.Tensor, return_loss: bool = True, return_feats: bool = False, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional AT loss computation and feature return for MultiStageDistiller.

        Args:
            x: input tensor
            return_loss: whether to compute and return the loss
            return_feats: whether to return extracted features instead of loss

        Returns:
            If return_feats is True, returns dict with keys 'student_feats' and 'teacher_feats'.
            Else if return_loss is True and in training mode, returns loss tensor.
            Otherwise returns student model output.
        """
        student_feats = self.student(x)
        if return_feats:
            with torch.no_grad():
                teacher_feats = self.teacher(x)
            return {'student_feats': student_feats, 'teacher_feats': teacher_feats}

        if self.training and return_loss:
            with torch.no_grad():
                teacher_feats = self.teacher(x)
            loss = self.compute_loss(teacher_feats, student_feats)
            return loss
        return student_feats