


import torch
import torch.nn.functional as F

class KDHintonDistiller:
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, class_weights=None):
        self.temperature = temperature
        self.alpha = alpha
        # Default class weights for binary classification (adjust based on data distribution)
        if class_weights is None:
            self.class_weights = torch.tensor([0.9766, 1.0246], dtype=torch.float32)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

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

        # Class-balanced cross-entropy loss with ground truth
        if loss_fn_ce is None:
            # Move class weights to the same device as the logits
            device = student_logits.device
            class_weights = self.class_weights.to(device)
            loss_fn_ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        ce_loss = loss_fn_ce(student_logits, labels)

        # Weighted sum
        return alpha * distill_loss + (1 - alpha) * ce_loss