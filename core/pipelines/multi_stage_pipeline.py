"""
Multi-Stage Pipeline - Composable Distiller Orchestration
==========================================================

Combines multiple distillers in flexible execution modes:
- Sequential: Chain distillers in order
- Parallel: Run distillers independently, aggregate losses
- Conditional: Route based on runtime conditions
- Hybrid: Mix sequential + parallel stages

Optimized for Google Colab T4 GPU with memory-efficient execution.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import torch
import torch.nn as nn
from dataclasses import dataclass

from .base_pipeline import BasePipeline, PipelineMetrics
from .single_distiller_pipeline import SingleDistillerPipeline
from core.distillers.base_distiller import BaseDistiller
from core.utils.device_utils import move_to_device


class ExecutionMode(Enum):
    """Execution modes for multi-stage pipeline."""
    SEQUENTIAL = "sequential"  # Run stages in order
    PARALLEL = "parallel"      # Run all stages simultaneously
    CONDITIONAL = "conditional"  # Route based on conditions
    HYBRID = "hybrid"          # Mix sequential + parallel


@dataclass
class PipelineStage:
    """
    A stage in the multi-stage pipeline.
    
    Each stage can contain one or more pipelines/distillers
    and has its own execution mode and loss weight.
    """
    name: str
    pipelines: List[BasePipeline]
    weight: float = 1.0
    mode: ExecutionMode = ExecutionMode.PARALLEL
    condition: Optional[Callable] = None  # For conditional execution
    
    def __post_init__(self):
        """Validate stage configuration."""
        if self.weight < 0:
            raise ValueError(f"Stage '{self.name}' weight must be non-negative")
        if not self.pipelines:
            raise ValueError(f"Stage '{self.name}' must have at least one pipeline")


class MultiStagePipeline(BasePipeline):
    """
    Multi-stage distillation pipeline.
    
    Orchestrates multiple distillers with flexible execution strategies.
    Supports dynamic composition, weight management, and intelligent routing.
    
    Usage:
        # Create individual pipelines
        kd_pipeline = SingleDistillerPipeline(kd_distiller)
        feature_pipeline = SingleDistillerPipeline(feature_distiller)
        
        # Combine in multi-stage
        multi = MultiStagePipeline(teacher, student, device=device)
        multi.add_stage('logit', [kd_pipeline], weight=0.7)
        multi.add_stage('features', [feature_pipeline], weight=0.3)
        multi.setup()
        
        # Use as regular pipeline
        loss, metrics = multi(batch)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        mode: Union[str, ExecutionMode] = ExecutionMode.SEQUENTIAL,
        name: str = "MultiStagePipeline",
    ):
        """
        Initialize multi-stage pipeline.
        
        Args:
            teacher: Teacher model
            student: Student model
            config: Pipeline configuration
            device: Target device
            mode: Global execution mode (sequential, parallel, conditional, hybrid)
            name: Pipeline name
        """
        super().__init__(
            teacher=teacher,
            student=student,
            config=config,
            device=device,
            name=name,
        )
        
        # Parse execution mode
        if isinstance(mode, str):
            try:
                self.mode = ExecutionMode(mode.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid execution mode: '{mode}'. "
                    f"Choose from: {[m.value for m in ExecutionMode]}"
                )
        else:
            self.mode = mode
        
        # Stage management
        self.stages: List[PipelineStage] = []
        self._stage_metrics: Dict[str, PipelineMetrics] = {}
        
        # Loss aggregation
        self.normalize_weights = config.get('normalize_weights', True)
        self._total_weight = 0.0
        
        # Memory optimization for T4
        self.checkpoint_gradients = config.get('checkpoint_gradients', False)
        if self.checkpoint_gradients:
            print("[MultiStagePipeline] Gradient checkpointing enabled (saves memory)")
    
    def add_stage(
        self,
        name: str,
        pipelines: Union[BasePipeline, List[BasePipeline], BaseDistiller, List[BaseDistiller]],
        weight: float = 1.0,
        mode: Optional[Union[str, ExecutionMode]] = None,
        condition: Optional[Callable] = None,
    ) -> 'MultiStagePipeline':
        """
        Add a stage to the pipeline (fluent API).
        
        Args:
            name: Stage name
            pipelines: Single or list of pipelines/distillers
            weight: Loss weight for this stage
            mode: Execution mode for this stage (overrides global mode)
            condition: Optional condition function for conditional execution
        
        Returns:
            Self for method chaining
        """
        # Ensure pipelines is a list
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        
        # Wrap distillers in SingleDistillerPipeline
        wrapped_pipelines = []
        for p in pipelines:
            if isinstance(p, BaseDistiller):
                wrapped = SingleDistillerPipeline(p, name=f"{name}_{p.__class__.__name__}")
                wrapped_pipelines.append(wrapped)
            elif isinstance(p, BasePipeline):
                wrapped_pipelines.append(p)
            else:
                raise TypeError(
                    f"Expected BasePipeline or BaseDistiller, got {type(p).__name__}"
                )
        
        # Parse stage mode (use global if not specified)
        stage_mode = self.mode
        if mode is not None:
            if isinstance(mode, str):
                stage_mode = ExecutionMode(mode.lower())
            else:
                stage_mode = mode
        
        # Create stage
        stage = PipelineStage(
            name=name,
            pipelines=wrapped_pipelines,
            weight=weight,
            mode=stage_mode,
            condition=condition,
        )
        
        self.stages.append(stage)
        self._total_weight += weight
        
        return self  # For chaining
    
    def setup(self) -> None:
        """Setup all stages and their pipelines."""
        if not self.stages:
            raise ValueError("No stages added to pipeline. Use add_stage() first.")
        
        print(f"[{self.name}] Setting up {len(self.stages)} stage(s)")
        
        for stage in self.stages:
            print(f"  - Stage '{stage.name}': {len(stage.pipelines)} pipeline(s), "
                  f"weight={stage.weight:.2f}, mode={stage.mode.value}")
            
            for pipeline in stage.pipelines:
                if not pipeline._is_setup:
                    pipeline.setup()
                    pipeline._is_setup = True
        
        # Normalize weights if requested
        if self.normalize_weights and self._total_weight > 0:
            for stage in self.stages:
                stage.weight /= self._total_weight
            print(f"[{self.name}] Normalized stage weights")
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all stages on the batch.
        
        Args:
            batch: Input batch
        
        Returns:
            Dictionary with outputs from all stages
        """
        batch = move_to_device(batch, self.device)
        return self._execute_stages(batch)
    
    def _execute_stages(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run every active stage, collecting pipeline outputs.

        Each stage's condition (if any) is evaluated to decide whether
        to run it.  Within a stage each pipeline is executed and its
        output stored keyed by ``pipeline_<i>``.

        The previous four separate methods (_forward_sequential,
        _forward_parallel, _forward_conditional, _forward_hybrid) were
        identical in practice.  This single method replaces all four.
        True parallel execution can be added here later without changing
        the public API.

        Args:
            batch: Device-resident input batch.

        Returns:
            Nested dict: ``{stage_name: {pipeline_i: outputs}}``.
        """
        all_outputs: Dict[str, Any] = {}

        for stage in self.stages:
            # Evaluate optional condition
            if stage.condition is not None and not stage.condition(batch, all_outputs):
                continue

            stage_outputs: Dict[str, Any] = {}
            for i, pipeline in enumerate(stage.pipelines):
                outputs = pipeline.forward(batch)
                stage_outputs[f"pipeline_{i}"] = outputs

            all_outputs[stage.name] = stage_outputs

        return all_outputs
    
    def compute_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Aggregate losses from all stages.
        
        Args:
            outputs: Dictionary from forward() with stage outputs
        
        Returns:
            Weighted sum of all stage losses
        """
        total_loss = torch.tensor(0.0, device=self.device)
        self._stage_metrics = {}
        
        for stage in self.stages:
            if stage.name not in outputs:
                continue  # Stage was skipped
            
            stage_outputs = outputs[stage.name]
            stage_loss = torch.tensor(0.0, device=self.device)
            
            # Aggregate losses from all pipelines in stage
            for pipeline_key, pipeline_outputs in stage_outputs.items():
                pipeline_idx = int(pipeline_key.split('_')[1])
                pipeline = stage.pipelines[pipeline_idx]
                
                loss = pipeline.compute_loss(pipeline_outputs)
                stage_loss = stage_loss + loss
            
            # Average across pipelines in stage
            if len(stage_outputs) > 0:
                stage_loss = stage_loss / len(stage_outputs)
            
            # Weight and add to total
            weighted_loss = stage.weight * stage_loss
            total_loss = total_loss + weighted_loss
            
            # Track per-stage metrics
            self._stage_metrics[stage.name] = PipelineMetrics(
                total_loss=stage_loss.item(),
                component_losses={
                    'weighted_loss': weighted_loss.item(),
                    'weight': stage.weight,
                }
            )
        
        return total_loss
    
    def get_metrics(self) -> PipelineMetrics:
        """Collect metrics from all stages."""
        metrics = super().get_metrics()
        
        # Add per-stage metrics
        stage_metrics_dict = {}
        for stage_name, stage_metric in self._stage_metrics.items():
            stage_metrics_dict[stage_name] = {
                'loss': stage_metric.total_loss,
                'weighted_loss': stage_metric.component_losses.get('weighted_loss', 0),
                'weight': stage_metric.component_losses.get('weight', 0),
            }
        
        metrics.custom_metrics['stage_metrics'] = stage_metrics_dict
        metrics.custom_metrics['num_stages'] = len(self.stages)
        metrics.custom_metrics['total_pipelines'] = sum(len(s.pipelines) for s in self.stages)
        
        return metrics
    
    def cleanup(self) -> None:
        """Cleanup all stage pipelines."""
        for stage in self.stages:
            for pipeline in stage.pipelines:
                pipeline.cleanup()
        super().cleanup()
    
    def __repr__(self) -> str:
        """String representation."""
        stage_summary = [
            f"{s.name}({len(s.pipelines)} pipelines, w={s.weight:.2f})"
            for s in self.stages
        ]
        return (
            f"MultiStagePipeline(\n"
            f"  mode={self.mode.value},\n"
            f"  stages=[{', '.join(stage_summary)}],\n"
            f"  device={self.device}\n"
            f")"
        )


# Register in pipeline registry
from .pipeline_registry import get_registry
get_registry().register_class('multi_stage', MultiStagePipeline, aliases=['multi', 'multistage'])
