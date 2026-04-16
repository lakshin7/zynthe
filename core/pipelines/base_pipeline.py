"""
Base Pipeline - Abstract Interface for Distillation Pipelines
==============================================================

Provides the core interface that all pipelines must implement.
Optimized for Google Colab T4 GPU (16GB VRAM, limited RAM).

Key Features:
- Memory-efficient batch processing
- Automatic gradient management
- Metrics collection and aggregation
- Lifecycle hooks (setup, forward, cleanup)
- Device management with T4 optimization

All pipeline implementations must extend this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import time


@dataclass
class PipelineMetrics:
    """Container for pipeline metrics."""
    total_loss: float = 0.0
    component_losses: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'total_loss': self.total_loss,
            'component_losses': self.component_losses,
            'custom_metrics': self.custom_metrics,
            'execution_time_ms': self.execution_time_ms,
            'memory_allocated_mb': self.memory_allocated_mb,
            'memory_reserved_mb': self.memory_reserved_mb,
        }


class BasePipeline(ABC, nn.Module):
    """
    Abstract base class for all distillation pipelines.
    
    Provides:
    - Standard lifecycle (setup → forward → compute_loss → cleanup)
    - Memory tracking for T4 GPU
    - Automatic gradient management
    - Metrics collection
    - Device handling
    
    Subclasses must implement:
    - setup(): Initialize pipeline components
    - forward(): Execute pipeline on a batch
    - compute_loss(): Calculate distillation loss
    
    Optional overrides:
    - get_metrics(): Collect custom metrics
    - cleanup(): Release resources
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize base pipeline.
        
        Args:
            teacher: Pre-trained teacher model
            student: Student model to train
            config: Pipeline configuration
            device: Target device (auto-detect if None)
            name: Pipeline name for logging
        """
        super().__init__()
        
        self.teacher = teacher
        self.student = student
        self.config = config or {}
        self.device = device or self._auto_detect_device()
        self.name = name or self.__class__.__name__
        
        # Move models to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        
        # Freeze teacher by default
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # State tracking
        self._is_setup = False
        self._total_batches_processed = 0
        self._cumulative_metrics = PipelineMetrics()
        
        # Memory optimization for T4
        self._enable_memory_optimization()
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect best available device (T4 GPU prioritized)."""
        if torch.cuda.is_available():
            # Print GPU info for Colab users
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Pipeline] Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("[Pipeline] Using Apple MPS")
            return torch.device("mps")
        else:
            print("[Pipeline] Using CPU (WARNING: Training will be slow)")
            return torch.device("cpu")
    
    def _enable_memory_optimization(self):
        """Enable memory optimizations for T4 GPU (16GB VRAM)."""
        if self.device.type == 'cuda':
            # Enable TF32 for faster computation on Ampere GPUs (T4 doesn't support but safe)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuner for optimal convolution algorithms
            torch.backends.cudnn.benchmark = True
            
            # Set memory allocator settings for better memory management
            # Helps prevent fragmentation on T4
            torch.cuda.empty_cache()
            
            print(f"[Pipeline] Memory optimization enabled for {self.device}")
    
    @abstractmethod
    def setup(self) -> None:
        """
        Setup pipeline components.
        
        Called once before training starts. Use this to:
        - Initialize loss functions
        - Register hooks
        - Prepare any stateful components
        - Validate configuration
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pipeline on a batch.
        
        Args:
            batch: Input batch (typically with 'input_ids', 'attention_mask', 'labels')
        
        Returns:
            Dictionary with outputs:
                - teacher_outputs: Teacher model outputs
                - student_outputs: Student model outputs
                - intermediate_features: Any extracted features
                - ... (custom outputs)
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute distillation loss from pipeline outputs.
        
        Args:
            outputs: Dictionary from forward() call
        
        Returns:
            Total loss tensor (scalar)
        
        Must be implemented by subclasses.
        """
        pass
    
    def get_metrics(self) -> PipelineMetrics:
        """
        Collect pipeline metrics.
        
        Returns:
            PipelineMetrics object with loss, timing, memory info
        
        Override to add custom metrics.
        """
        metrics = PipelineMetrics()
        
        # Memory metrics (T4 specific)
        if self.device.type == 'cuda':
            metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1e6
            metrics.memory_reserved_mb = torch.cuda.memory_reserved() / 1e6
        
        return metrics
    
    def cleanup(self) -> None:
        """
        Cleanup pipeline resources.
        
        Called after training completes. Override to:
        - Remove hooks
        - Clear caches
        - Release GPU memory
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def __call__(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, PipelineMetrics]:
        """
        Execute full pipeline: forward + loss + metrics.
        
        Args:
            batch: Input batch
        
        Returns:
            (loss, metrics) tuple
        """
        # Ensure setup was called
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        
        # Track execution time
        start_time = time.time()
        
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute loss
        loss = self.compute_loss(outputs)
        
        # Collect metrics
        metrics = self.get_metrics()
        metrics.total_loss = loss.item()
        metrics.execution_time_ms = (time.time() - start_time) * 1000
        
        # Update tracking
        self._total_batches_processed += 1
        
        return loss, metrics
    
    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._cumulative_metrics = PipelineMetrics()
        self._total_batches_processed = 0
    
    def get_cumulative_metrics(self) -> Dict[str, Any]:
        """Get averaged metrics across all batches."""
        if self._total_batches_processed == 0:
            return {}
        
        return {
            'batches_processed': self._total_batches_processed,
            'avg_loss': self._cumulative_metrics.total_loss / self._total_batches_processed,
            'total_execution_time_ms': self._cumulative_metrics.execution_time_ms,
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.name}(\n"
            f"  teacher={self.teacher.__class__.__name__},\n"
            f"  student={self.student.__class__.__name__},\n"
            f"  device={self.device},\n"
            f"  batches_processed={self._total_batches_processed}\n"
            f")"
        )
