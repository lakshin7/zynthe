"""
Single Distiller Pipeline - Wrapper for Individual Distillers
==============================================================

Wraps existing distillers (KD-Hinton, Feature, Attention, etc.) in the
pipeline interface for backward compatibility and unified API.

This allows existing distillers to work seamlessly in the new pipeline system.
"""

from __future__ import annotations


from typing import Any, Dict, Optional
import inspect
import torch

from .base_pipeline import BasePipeline, PipelineMetrics
from zynthe.core.distillers.base_distiller import BaseDistiller
from zynthe.core.utils.device_utils import move_to_device


class SingleDistillerPipeline(BasePipeline):
    """
    Pipeline wrapper for a single distiller.
    
    Provides pipeline interface for existing distillers without modification.
    Acts as an adapter between the old distiller API and new pipeline API.
    
    Usage:
        from zynthe.core.distillers import KDHinton
        from zynthe.core.pipelines import SingleDistillerPipeline
        
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
            distiller: BaseDistiller or a compatible object with teacher, student, and compute_loss
            config: Optional configuration (merged with distiller config)
            name: Pipeline name (defaults to distiller class name)
        """
        # Accept BaseDistiller subclasses and lightweight duck-typed distillers
        # used by library consumers/tests.
        if not all(hasattr(distiller, attr) for attr in ("teacher", "student", "compute_loss")):
            raise TypeError(
                "Expected a distiller with teacher, student, and compute_loss attributes, "
                f"got {type(distiller).__name__}"
            )
        
        # Extract teacher, student, device from distiller
        teacher = distiller.teacher
        student = distiller.student
        device = getattr(distiller, "device", None)
        if device is None:
            try:
                device = next(student.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        
        # Merge configs
        merged_config = {**(getattr(distiller, "config", {}) or {}), **(config or {})}
        
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
        return move_to_device(data, self.device)
    
    def compute_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute distillation loss using the wrapped distiller.
        
        Uses ``inspect.signature`` to determine which arguments the
        distiller's ``compute_loss`` method actually accepts, then
        builds call kwargs accordingly.  This replaces the previous
        ``try/except TypeError`` chain which silently swallowed real
        errors.
        
        Args:
            outputs: Dictionary from forward()
        
        Returns:
            Total loss tensor
        """
        teacher_outputs = outputs['teacher_outputs']
        student_outputs = outputs['student_outputs']
        batch = outputs['batch']
        
        # Build the pool of available arguments
        available = {
            'teacher_outputs': teacher_outputs,
            'student_outputs': student_outputs,
            'labels': batch.get('labels'),
            'targets': batch.get('labels'),
            'input_ids': batch.get('input_ids'),
            'attention_mask': batch.get('attention_mask'),
            'pixel_values': batch.get('pixel_values'),
            'image': batch.get('image'),
            'batch': batch,
            'student_features': outputs.get('student_features', {}),
            'teacher_features': outputs.get('teacher_features', {}),
        }
        
        # Introspect the distiller's compute_loss signature
        try:
            sig = inspect.signature(self.distiller.compute_loss)
        except (ValueError, TypeError):
            # Fallback: call with the two required positional args
            loss = self.distiller.compute_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
            )
            return loss if isinstance(loss, torch.Tensor) else loss[0]
        
        call_kwargs: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.kind == param.VAR_KEYWORD:  # **kwargs
                continue
            if param.kind == param.VAR_POSITIONAL:  # *args
                continue
            if name in available:
                call_kwargs[name] = available[name]
            elif param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"{self.distiller.__class__.__name__}.compute_loss requires "
                    f"unsupported parameter {name!r}"
                )
        
        result = self.distiller.compute_loss(**call_kwargs)
        
        # Handle tuple return (loss, metrics_dict)
        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result
        
        # Extract component losses if available
        if hasattr(self.distiller, 'last_component_losses'):
            self._component_losses = self.distiller.last_component_losses
        elif isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict):
            self._component_losses = result[1]
        
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
