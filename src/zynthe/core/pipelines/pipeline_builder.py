"""
Pipeline Builder - Fluent API for Building Pipelines
=====================================================

Provides an easy-to-use fluent interface for constructing pipelines
from configuration or programmatically.

Supports:
- Configuration-based building (YAML/dict)
- Fluent method chaining
- Auto-suggestion based on dataset characteristics
- Validation and optimization
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import copy

from .base_pipeline import BasePipeline
from .single_distiller_pipeline import SingleDistillerPipeline
from .multi_stage_pipeline import MultiStagePipeline, ExecutionMode
from zynthe.core.distillers.multi_stage_distiller import DistillerRegistry


class PipelineBuilder:
    """
    Fluent API for building distillation pipelines.
    
    Usage:
        # Example 1: Single distiller
        pipeline = PipelineBuilder() \\
            .add_distiller('kd_hinton', temperature=4.0) \\
            .build(teacher, student, device)
        
        # Example 2: Multi-stage
        pipeline = PipelineBuilder() \\
            .add_stage('logit', weight=0.7) \\
                .add_distiller('kd_hinton', alpha=0.8) \\
            .add_stage('features', weight=0.3) \\
                .add_distiller('feature', layers=[6, 8]) \\
            .with_mode('sequential') \\
            .build(teacher, student, device)
        
        # Example 3: From config
        pipeline = PipelineBuilder.from_config(config, teacher, student, device)
    """
    
    def __init__(self):
        """Initialize builder."""
        self._stages: List[Dict[str, Any]] = []
        self._current_stage: Optional[Dict[str, Any]] = None
        self._mode: ExecutionMode = ExecutionMode.SEQUENTIAL
        self._config: Dict[str, Any] = {}
        self._name: str = "CustomPipeline"
        self._normalize_weights: bool = True
        # Adapter support (Phase 2)
        self._teacher_adapter = None
        self._student_adapter = None
        self._auto_detect_adapters_flag: bool = False
    
    def add_stage(
        self,
        name: str,
        weight: float = 1.0,
        mode: Optional[Union[str, ExecutionMode]] = None,
    ) -> 'PipelineBuilder':
        """
        Start a new stage (fluent API).
        
        Args:
            name: Stage name
            weight: Loss weight for this stage
            mode: Execution mode for this stage
        
        Returns:
            Self for chaining
        """
        # Finalize current stage if exists
        if self._current_stage is not None:
            self._stages.append(self._current_stage)
        
        # Parse mode
        stage_mode = None
        if mode is not None:
            if isinstance(mode, str):
                stage_mode = ExecutionMode(mode.lower())
            else:
                stage_mode = mode
        
        # Start new stage
        self._current_stage = {
            'name': name,
            'weight': weight,
            'mode': stage_mode,
            'distillers': [],
        }
        
        return self
    
    def add_distiller(
        self,
        distiller_type: str,
        **distiller_config
    ) -> 'PipelineBuilder':
        """
        Add a distiller to the current stage.
        
        Args:
            distiller_type: Type of distiller (e.g., 'kd_hinton', 'feature')
            **distiller_config: Configuration for the distiller
        
        Returns:
            Self for chaining
        """
        if self._current_stage is None:
            # No explicit stage - create default stage
            self.add_stage('default_stage', weight=1.0)
        
        self._current_stage['distillers'].append({  # type: ignore[index]
            'type': distiller_type,
            'config': distiller_config,
        })
        
        return self
    
    def with_mode(self, mode: Union[str, ExecutionMode]) -> 'PipelineBuilder':
        """
        Set global execution mode.
        
        Args:
            mode: Execution mode (sequential, parallel, conditional, hybrid)
        
        Returns:
            Self for chaining
        """
        if isinstance(mode, str):
            self._mode = ExecutionMode(mode.lower())
        else:
            self._mode = mode
        
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """
        Set pipeline configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Self for chaining
        """
        self._config = config
        return self
    
    def with_name(self, name: str) -> 'PipelineBuilder':
        """
        Set pipeline name.
        
        Args:
            name: Pipeline name
        
        Returns:
            Self for chaining
        """
        self._name = name
        return self
    
    def normalize_weights(self, enable: bool = True) -> 'PipelineBuilder':
        """
        Enable/disable weight normalization.
        
        Args:
            enable: Whether to normalize stage weights
        
        Returns:
            Self for chaining
        """
        self._normalize_weights = enable
        return self
    
    def with_adapters(
        self,
        teacher_adapter=None,
        student_adapter=None,
    ) -> 'PipelineBuilder':
        """
        Set explicit adapters for teacher and student models.
        
        Args:
            teacher_adapter: ModelAdapter instance for the teacher
            student_adapter: ModelAdapter instance for the student
        
        Returns:
            Self for chaining
        """
        self._teacher_adapter = teacher_adapter
        self._student_adapter = student_adapter
        return self
    
    def auto_detect_adapters(self) -> 'PipelineBuilder':
        """
        Auto-detect adapters from model architectures at build time.
        
        Uses :class:`core.adapters.AdapterRegistry` to inspect both
        teacher and student models and select the right adapters.
        
        Returns:
            Self for chaining
        """
        self._auto_detect_adapters_flag = True
        return self
    
    def _attach_adapters(
        self,
        pipeline: BasePipeline,
        teacher: nn.Module,
        student: nn.Module,
    ) -> None:
        """Attach adapters to a built pipeline."""
        if self._auto_detect_adapters_flag:
            from zynthe.core.adapters import AdapterRegistry
            registry = AdapterRegistry()
            self._teacher_adapter = registry.detect(teacher)
            self._student_adapter = registry.detect(student)
            print(f"[PipelineBuilder] Auto-detected adapters: "
                  f"teacher={self._teacher_adapter}, student={self._student_adapter}")
        
        if self._teacher_adapter is not None:
            pipeline.teacher_adapter = self._teacher_adapter
        if self._student_adapter is not None:
            pipeline.student_adapter = self._student_adapter
    
    def build(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: Optional[torch.device] = None,
    ) -> BasePipeline:
        """
        Build the pipeline.
        
        Args:
            teacher: Teacher model
            student: Student model
            device: Target device
        
        Returns:
            Constructed pipeline
        """
        # Finalize current stage
        if self._current_stage is not None:
            self._stages.append(self._current_stage)
            self._current_stage = None
        
        # Determine if single or multi-stage
        if len(self._stages) == 0:
            raise ValueError("No stages or distillers added to pipeline")
        
        if len(self._stages) == 1 and len(self._stages[0]['distillers']) == 1:
            # Single distiller - use SingleDistillerPipeline
            pipeline = self._build_single_distiller(teacher, student, device)
        else:
            # Multi-stage - use MultiStagePipeline
            pipeline = self._build_multi_stage(teacher, student, device)
        
        # Attach adapters if configured
        self._attach_adapters(pipeline, teacher, student)
        
        return pipeline
    
    def _build_single_distiller(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: Optional[torch.device] = None,
    ) -> SingleDistillerPipeline:
        """Build a single distiller pipeline."""
        stage = self._stages[0]
        distiller_spec = stage['distillers'][0]
        
        # Get distiller class from registry
        distiller_registry = DistillerRegistry()
        distiller_class = distiller_registry.get(distiller_spec['type'])
        
        if distiller_class is None:
            raise ValueError(f"Unknown distiller type: {distiller_spec['type']}")
        
        # Create distiller instance
        distiller = distiller_class(
            teacher=teacher,
            student=student,
            config=distiller_spec['config'],
            device=device,
        )
        
        # Wrap in pipeline
        pipeline = SingleDistillerPipeline(
            distiller=distiller,
            config=self._config,
            name=f"{self._name}_{distiller_spec['type']}",
        )
        
        return pipeline
    
    def _build_multi_stage(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: Optional[torch.device] = None,
    ) -> MultiStagePipeline:
        """Build a multi-stage pipeline."""
        # Create multi-stage pipeline
        config = copy.deepcopy(self._config)
        config['normalize_weights'] = self._normalize_weights
        
        multi_pipeline = MultiStagePipeline(
            teacher=teacher,
            student=student,
            config=config,
            device=device,
            mode=self._mode,
            name=self._name,
        )
        
        # Add stages
        distiller_registry = DistillerRegistry()
        
        for stage_spec in self._stages:
            # Create pipelines for this stage
            stage_pipelines = []
            
            for distiller_spec in stage_spec['distillers']:
                # Get distiller class
                distiller_class = distiller_registry.get(distiller_spec['type'])
                if distiller_class is None:
                    raise ValueError(f"Unknown distiller type: {distiller_spec['type']}")
                
                # Create distiller
                distiller = distiller_class(
                    teacher=teacher,
                    student=student,
                    config=distiller_spec['config'],
                    device=device,
                )
                
                # Wrap in single pipeline
                pipeline = SingleDistillerPipeline(
                    distiller=distiller,
                    name=f"{stage_spec['name']}_{distiller_spec['type']}",
                )
                
                stage_pipelines.append(pipeline)
            
            # Add stage to multi-pipeline
            multi_pipeline.add_stage(
                name=stage_spec['name'],
                pipelines=stage_pipelines,
                weight=stage_spec['weight'],
                mode=stage_spec['mode'],
            )
        
        return multi_pipeline
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        teacher: nn.Module,
        student: nn.Module,
        device: Optional[torch.device] = None,
    ) -> BasePipeline:
        """
        Build pipeline from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            teacher: Teacher model
            student: Student model
            device: Target device
        
        Returns:
            Constructed pipeline
        
        Config format:
            distillation:
              pipeline:
                type: multi_stage  # or single
                mode: sequential   # or parallel, conditional, hybrid
                stages:
                  - name: logit_stage
                    weight: 0.7
                    distillers:
                      - type: kd_hinton
                        config:
                          temperature: 4.0
                          alpha: 0.7
                  - name: feature_stage
                    weight: 0.3
                    mode: parallel
                    distillers:
                      - type: feature
                        config:
                          layers: [6, 8]
        """
        builder = cls()
        
        # Extract pipeline config
        distil_config = config.get('distillation', {})
        pipeline_config = distil_config.get('pipeline', {})
        
        # Handle legacy single distiller config
        if 'type' not in pipeline_config or pipeline_config.get('type') == 'single':
            # Legacy: single distiller specified directly
            distiller_type = distil_config.get('method') or distil_config.get('type', 'kd_hinton')
            distiller_config = distil_config.get('config', {})
            
            return builder.add_distiller(distiller_type, **distiller_config).build(
                teacher, student, device
            )
        
        # Multi-stage configuration
        mode = pipeline_config.get('mode', 'sequential')
        builder.with_mode(mode)
        
        # Add stages
        stages = pipeline_config.get('stages', [])
        for stage_spec in stages:
            stage_name = stage_spec.get('name', 'unnamed_stage')
            stage_weight = stage_spec.get('weight', 1.0)
            stage_mode = stage_spec.get('mode', None)
            
            builder.add_stage(stage_name, weight=stage_weight, mode=stage_mode)
            
            # Add distillers to stage
            distillers = stage_spec.get('distillers', [])
            for distiller_spec in distillers:
                distiller_type = distiller_spec.get('type', 'kd_hinton')
                distiller_config = distiller_spec.get('config', {})
                
                builder.add_distiller(distiller_type, **distiller_config)
        
        # Build and return
        return builder.build(teacher, student, device)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PipelineBuilder(\n"
            f"  stages={len(self._stages)},\n"
            f"  mode={self._mode.value if isinstance(self._mode, ExecutionMode) else self._mode}\n"
            f")"
        )
