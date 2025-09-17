


import torch
import torch.nn.functional as F

class KDHintonDistiller:
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, student_logits, teacher_logits, labels, loss_fn_ce=None):
        """
        Compute the knowledge distillation loss as described by Hinton et al.
        """
        T = self.temperature
        alpha = self.alpha

        # Soft targets with temperature scaling
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL divergence loss for distillation
        kl_div = torch.nn.KLDivLoss(reduction="batchmean")
        distill_loss = kl_div(student_soft, teacher_soft) * (T * T)

        # Cross-entropy loss with ground truth
        if loss_fn_ce is None:
            loss_fn_ce = torch.nn.CrossEntropyLoss()
        ce_loss = loss_fn_ce(student_logits, labels)

        # Weighted sum
        return alpha * distill_loss + (1 - alpha) * ce_loss