

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityTransferDistiller(nn.Module):
    """
    Similarity-based distillation using cosine similarity between teacher and student features.
    This class computes the similarity loss between the representations of teacher and student models.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(SimilarityTransferDistiller, self).__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, student_features, teacher_features):
        """
        Compute the similarity loss between student and teacher features.
        Args:
            student_features (Tensor): Student feature representations, shape (batch, dim, ...)
            teacher_features (Tensor): Teacher feature representations, shape (batch, dim, ...)
        Returns:
            Tensor: Similarity loss.
        """
        return self.similarity_loss(student_features, teacher_features)

    def similarity_loss(self, student_features, teacher_features):
        """
        Cosine similarity loss between student and teacher features.
        Args:
            student_features (Tensor): Student feature representations, shape (batch, dim, ...)
            teacher_features (Tensor): Teacher feature representations, shape (batch, dim, ...)
        Returns:
            Tensor: Loss value.
        """
        # Flatten all dimensions except batch
        student_flat = student_features.view(student_features.size(0), -1)
        teacher_flat = teacher_features.view(teacher_features.size(0), -1)

        # Normalize features
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)

        # Cosine similarity for each sample in the batch
        cosine_sim = (student_norm * teacher_norm).sum(dim=1)

        # Loss: 1 - cosine similarity
        loss = 1.0 - cosine_sim

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss