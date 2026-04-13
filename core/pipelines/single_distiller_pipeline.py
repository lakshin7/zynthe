"""
Single Distiller Pipeline - Wrapper for Individual Distillers
==============================================================

Wraps existing distillers (KD-Hinton, Feature, Attention, etc.) in the
pipeline interface for backward compatibility and unified API.

This allows existing distillers to work seamlessly in the new pipeline system.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from .base_pipeline import BasePipeline, PipelineMetrics
from core.distillers.base_distiller import BaseDistiller


class SingleDistillerPipeline(BasePipeline):
    """
    Pipeline wrapper for a single distiller.
    
    Provides pipeline interface for existing distillers without modification.
    Acts as an adapter between the old distiller API and new pipeline API.
    
    Usage:
        from core.distillers import KDHinton
        from core.pipelines import SingleDistillerPipeline
        
        distiller = KDHinton(teacher, student, config, device)
        pipeline = SingleDistillerPipeline(distiller)
        
        # Use pipeline API
        loss, metrics = pipeline(batch)
    """
    
    def __init__(
        self,
        distiller: BaseDistiller,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize single distiller pipeline.
        
        Args:
            distiller: An instance of BaseDistiller or any subclass
            config: Optional configuration (merged with distiller config)
            name: Pipeline name (defaults to distiller class name)
        """
        # Validate distiller
        if not isinstance(distiller, BaseDistiller):
            raise TypeError(
                f"Expected BaseDistiller instance, got {type(distiller).__name__}"
            )
        
        # Extract teacher, student, device from distiller
        teacher = distiller.teacher
        student = distiller.student
        device = distiller.device
        
        # Merge configs
        merged_config = {**(distiller.config or {}), **(config or {})}
        
        # Initialize base pipeline
        super().__init__(
            teacher=teacher,
            student=student,
            config=merged_config,
            device=device,
            name=name or f"{distiller.__class__.__name__}Pipeline",
        )
        
        # Store distiller reference
        self.distiller = distiller
        
        # Track component losses from distiller
        self._component_losses: Dict[str, float] = {}
    
    def setup(self) -> None:
        """
        Setup pipeline (distiller is already initialized).
        
        The distiller handles its own setup in __init__, so we just
        mark the pipeline as ready.
        """
        # Distillers are self-contained and setup in __init__
        # No additional setup needed
        pass
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute distiller forward pass.
        
        Args:
            batch: Input batch with 'input_ids', 'attention_mask', 'labels'
        
        Returns:
            Dictionary with teacher and student outputs
        """
        # Move batch to device
        batch = self._move_to_device(batch)
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            self.teacher.eval()
            teacher_outputs = self.teacher(**batch)
        
        # Student forward (with gradients)
        self.student.train()
        student_outputs = self.student(**batch)
        
        return {
            'teacher_outputs': teacher_outputs,
            'student_outputs': student_outputs,
            'batch': batch,
        }
    
    def _move_to_device(self, data: Any) -> Any:
        """Move tensors to device recursively."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._move_to_device(v) for v in data)
        return data
    
    def compute_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute distillation loss using the wrapped distiller.
        
        Args:
            outputs: Dictionary from forward()
        
        Returns:
            Total loss tensor
        """
        teacher_outputs = outputs['teacher_outputs']
        student_outputs = outputs['student_outputs']
        batch = outputs['batch']
        
        # Call distiller's compute_loss method
        # Different distillers have different signatures, handle gracefully
        try:
            # Try full signature (teacher_outputs, student_outputs, labels, input_ids, ...)
            loss = self.distiller.compute_loss(
                teacher_outputs=teacher_outputs,
                student_outputs=student_outputs,
                labels=batch.get('labels'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
            )
        except TypeError:
            # Fallback: try minimal signature
            try:
                loss = self.distiller.compute_loss(
                    teacher_outputs=teacher_outputs,
                    student_outputs=student_outputs,
                )
            except TypeError:
                # Last resort: call with outputs dict
                loss = self.distiller.compute_loss(outputs)
        
        # Extract component losses if available
        if hasattr(self.distiller, 'last_component_losses'):
            self._component_losses = self.distiller.last_component_losses
        elif hasattr(loss, 'component_losses'):
            self._component_losses = loss.component_losses
        
        return loss
    
    def get_metrics(self) -> PipelineMetrics:
        """
        Collect metrics from distiller and pipeline.
        
        Returns:
            PipelineMetrics with loss components and memory info
        """
        metrics = super().get_metrics()
        
        # Add component losses
        metrics.component_losses = self._component_losses.copy()
        
        # Add distiller-specific metrics if available
        if hasattr(self.distiller, 'get_metrics'):
            distiller_metrics = self.distiller.get_metrics()
            if isinstance(distiller_metrics, dict):
                metrics.custom_metrics.update(distiller_metrics)
        
        return metrics
    
    def cleanup(self) -> None:
        """Cleanup distiller and pipeline resources."""
        # Call distiller cleanup if available
        if hasattr(self.distiller, 'cleanup'):
            self.distiller.cleanup()
        
        # Call base cleanup
        super().cleanup()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SingleDistillerPipeline(\n"
            f"  distiller={self.distiller.__class__.__name__},\n"
            f"  device={self.device},\n"
            f"  batches_processed={self._total_batches_processed}\n"
            f")"
        )
