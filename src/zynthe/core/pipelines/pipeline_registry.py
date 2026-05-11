"""
Pipeline Registry - Discovery and Instantiation System
=======================================================

Central registry for all pipeline types. Enables dynamic pipeline
creation from configuration strings and automatic discovery.

Similar to DistillerRegistry but for the pipeline layer.
"""

from __future__ import annotations


from typing import Any, Dict, Type, Optional, Callable
import torch.nn as nn

from .base_pipeline import BasePipeline


class PipelineRegistry:
    """
    Registry for pipeline types.
    
    Allows registration and instantiation of pipelines by name.
    Supports both built-in and custom pipeline types.
    
    Usage:
        registry = PipelineRegistry()
        
        # Get pipeline class
        pipeline_class = registry.get('single_distiller')
        
        # Create pipeline instance
        pipeline = registry.create(
            'single_distiller',
            teacher=teacher,
            student=student,
            config=config
        )
        
        # Register custom pipeline
        @registry.register('my_custom_pipeline')
        class MyCustomPipeline(BasePipeline):
            ...
    """
    
    _instance = None
    _pipelines: Dict[str, Type[BasePipeline]] = {}
    _aliases: Dict[str, str] = {}
    
    def __new__(cls):
        """Singleton pattern - one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_builtin_pipelines()
        return cls._instance
    
    def _initialize_builtin_pipelines(self) -> None:
        """Register built-in pipeline types."""
        # Import here to avoid circular imports
        from .single_distiller_pipeline import SingleDistillerPipeline
        
        # Register core pipelines
        self._pipelines['single_distiller'] = SingleDistillerPipeline
        
        # Aliases for convenience
        self._aliases['single'] = 'single_distiller'
        self._aliases['wrapper'] = 'single_distiller'
    
    def register(
        self,
        name: str,
        aliases: Optional[list] = None
    ) -> Callable:
        """
        Decorator to register a pipeline class.
        
        Args:
            name: Primary name for the pipeline
            aliases: Optional list of alternative names
        
        Returns:
            Decorator function
        
        Example:
            @registry.register('my_pipeline', aliases=['mp', 'custom'])
            class MyPipeline(BasePipeline):
                ...
        """
        def decorator(cls: Type[BasePipeline]) -> Type[BasePipeline]:
            # Validate class extends BasePipeline
            if not issubclass(cls, BasePipeline):
                raise TypeError(
                    f"Pipeline {cls.__name__} must extend BasePipeline"
                )
            
            # Register primary name
            self._pipelines[name.lower()] = cls
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias.lower()] = name.lower()
            
            return cls
        
        return decorator
    
    def register_class(
        self,
        name: str,
        pipeline_class: Type[BasePipeline],
        aliases: Optional[list] = None
    ) -> None:
        """
        Register a pipeline class directly (non-decorator).
        
        Args:
            name: Primary name for the pipeline
            pipeline_class: The pipeline class
            aliases: Optional list of alternative names
        """
        if not issubclass(pipeline_class, BasePipeline):
            raise TypeError(
                f"Pipeline {pipeline_class.__name__} must extend BasePipeline"
            )
        
        self._pipelines[name.lower()] = pipeline_class
        
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name.lower()
    
    def get(self, name: str) -> Optional[Type[BasePipeline]]:
        """
        Get pipeline class by name or alias.
        
        Args:
            name: Pipeline name or alias
        
        Returns:
            Pipeline class or None if not found
        """
        name = name.lower()
        
        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]
        
        return self._pipelines.get(name)
    
    def create(
        self,
        name: str,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> BasePipeline:
        """
        Create a pipeline instance by name.
        
        Args:
            name: Pipeline name or alias
            teacher: Teacher model
            student: Student model
            config: Pipeline configuration
            device: Target device
            **kwargs: Additional arguments for pipeline constructor
        
        Returns:
            Initialized pipeline instance
        
        Raises:
            ValueError: If pipeline name not found
        """
        pipeline_class = self.get(name)
        
        if pipeline_class is None:
            available = self.list_available()
            raise ValueError(
                f"Unknown pipeline type: '{name}'. "
                f"Available pipelines: {', '.join(available)}"
            )
        
        # Create instance
        return pipeline_class(
            teacher=teacher,
            student=student,
            config=config,
            device=device,
            **kwargs
        )
    
    def list_available(self) -> list:
        """
        List all registered pipeline names.
        
        Returns:
            Sorted list of pipeline names
        """
        return sorted(self._pipelines.keys())
    
    def list_aliases(self) -> Dict[str, str]:
        """
        List all registered aliases and their targets.
        
        Returns:
            Dictionary mapping aliases to pipeline names
        """
        return self._aliases.copy()
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a pipeline by name.
        
        Args:
            name: Pipeline name to unregister
        
        Returns:
            True if unregistered, False if not found
        """
        name = name.lower()
        
        if name in self._pipelines:
            del self._pipelines[name]
            # Remove any aliases pointing to this pipeline
            self._aliases = {
                k: v for k, v in self._aliases.items() if v != name
            }
            return True
        
        return False
    
    def __repr__(self) -> str:
        """String representation showing registered pipelines."""
        pipelines = self.list_available()
        return f"PipelineRegistry(pipelines={pipelines})"


# Global singleton instance
_global_registry = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    """Get the global pipeline registry instance."""
    return _global_registry
