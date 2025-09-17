

import torch
import torch.nn as nn

class FeatureDistiller:
    """
    Implements feature-based knowledge distillation between a teacher and student model.
    """
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, feature_layers: list, loss_fn=nn.MSELoss()):
        self.teacher = teacher_model
        self.student = student_model
        self.feature_layers = feature_layers
        self.loss_fn = loss_fn
        self.teacher_features = {}
        self.student_features = {}
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

    def compute_loss(self):
        total_loss = 0.0
        for layer_name in self.feature_layers:
            teacher_feat = self.teacher_features.get(layer_name)
            student_feat = self.student_features.get(layer_name)
            if teacher_feat is not None and student_feat is not None:
                total_loss += self.loss_fn(student_feat, teacher_feat.detach())
        return total_loss

    def step(
        self,
        student_inputs: dict,
        return_feats: bool = False
    ) -> 'tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]] | torch.Tensor':
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