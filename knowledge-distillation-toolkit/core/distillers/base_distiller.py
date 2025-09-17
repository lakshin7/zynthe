


from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

class BaseDistiller(nn.Module):
    """
    Base class for distillers. Provides the basic structure and hooks for
    knowledge distillation, attention, feature, and multi-stage distillers.
    """
    def __init__(self, student: nn.Module, teacher: nn.Module):
        """
        Initialize the distiller with student and teacher models.

        Args:
            student (nn.Module): The student model to be trained.
            teacher (nn.Module): The teacher model to provide guidance.
        """
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compute_loss(
        self,
        student_outputs: Any,
        teacher_outputs: Any,
        targets: Any,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the distillation loss.

        Args:
            student_outputs: Output from the student model.
            teacher_outputs: Output from the teacher model.
            targets: Ground truth targets.
            **kwargs: Additional arguments for loss computation.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss components.
        """
        raise NotImplementedError("compute_loss must be implemented in subclasses.")

    def pre_forward_hook(self, *args, **kwargs) -> None:
        """
        Hook to be executed before the forward pass.
        Can be used for setup or input preprocessing.
        """
        pass

    def post_forward_hook(self, *args, **kwargs) -> None:
        """
        Hook to be executed after the forward pass.
        Can be used for cleanup or logging.
        """
        pass

    def forward(
        self,
        inputs: Any,
        targets: Any = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward pass through the distiller.

        Args:
            inputs: Input data for the models.
            targets: Ground truth targets, if available.
            **kwargs: Additional arguments.

        Returns:
            Tuple[Any, Dict[str, torch.Tensor]]: Tuple of student outputs and losses.
        """
        self.pre_forward_hook(inputs=inputs, targets=targets, **kwargs)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        student_outputs = self.student(inputs)
        losses = self.compute_loss(student_outputs, teacher_outputs, targets, **kwargs)
        self.post_forward_hook(
            inputs=inputs,
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=targets,
            losses=losses,
            **kwargs
        )
        return student_outputs, losses