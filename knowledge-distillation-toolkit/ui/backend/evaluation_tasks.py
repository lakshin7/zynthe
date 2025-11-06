"""
Async Evaluation Task Manager for Zynthe UI
Handles non-blocking evaluation execution with progress tracking and cancellation support
"""

import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class TaskStatus(str, Enum):
    """Task status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationType(str, Enum):
    """Types of evaluation tasks"""
    STANDARD = "standard"  # Basic accuracy, f1, precision, recall
    EXTENDED = "extended"  # With DEI, CAS, KL divergence
    DUAL = "dual"  # Teacher-student comparison
    BENCHMARK = "benchmark"  # TruthfulQA, MMLU, GSM8K
    CURRICULUM = "curriculum"  # Multi-stage evaluation


@dataclass
class EvaluationTask:
    """Represents an evaluation task"""
    task_id: str
    experiment_id: str
    eval_type: EvaluationType
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_stage: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Internal
    future: Optional[Future] = None
    progress_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API response"""
        return {
            'task_id': self.task_id,
            'experiment_id': self.experiment_id,
            'eval_type': self.eval_type.value,
            'status': self.status.value,
            'progress': self.progress,
            'current_stage': self.current_stage,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress_data': self.progress_data
        }


class EvaluationTaskManager:
    """
    Manages async evaluation tasks with ThreadPoolExecutor.
    Provides task queuing, progress tracking, and cancellation.
    """
    
    def __init__(self, max_workers: int = 2, websocket_manager=None):
        """
        Args:
            max_workers: Maximum concurrent evaluation tasks
            websocket_manager: Optional WebSocket manager for live updates
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eval_worker")
        self.tasks: Dict[str, EvaluationTask] = {}
        self.websocket_manager = websocket_manager
        self.lock = threading.Lock()
        
        print(f"[TASK MANAGER] Initialized with {max_workers} workers")
    
    def create_task(
        self,
        experiment_id: str,
        eval_type: EvaluationType,
        eval_func: Callable,
        **kwargs
    ) -> str:
        """
        Create and start a new evaluation task.
        
        Args:
            experiment_id: Experiment ID to evaluate
            eval_type: Type of evaluation
            eval_func: Function to execute (should accept progress_callback)
            **kwargs: Additional arguments for eval_func
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        
        # Create task object
        task = EvaluationTask(
            task_id=task_id,
            experiment_id=experiment_id,
            eval_type=eval_type
        )
        
        # Create progress callback that updates task
        def progress_callback(progress_data: Dict[str, Any]):
            with self.lock:
                task.progress_data = progress_data
                task.progress = progress_data.get('progress', 0.0)
                task.current_stage = progress_data.get('stage', '')
                
                # Broadcast to WebSocket if available
                if self.websocket_manager:
                    try:
                        self.websocket_manager.broadcast({
                            'type': 'evaluation_progress',
                            'task_id': task_id,
                            **progress_data
                        })
                    except Exception as e:
                        print(f"[WARNING] WebSocket broadcast failed: {e}")
        
        # Wrap eval_func to handle task lifecycle
        def wrapped_eval():
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                print(f"[TASK {task_id[:8]}] Started {eval_type.value} evaluation")
                
                # Execute evaluation with progress callback
                result = eval_func(progress_callback=progress_callback, **kwargs)
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                task.progress = 100.0
                print(f"[TASK {task_id[:8]}] Completed successfully")
                
                # Broadcast completion
                if self.websocket_manager:
                    self.websocket_manager.broadcast({
                        'type': 'evaluation_completed',
                        'task_id': task_id,
                        'result': result
                    })
                
                return result
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = str(e)
                print(f"[TASK {task_id[:8]}] Failed: {e}")
                
                # Broadcast failure
                if self.websocket_manager:
                    self.websocket_manager.broadcast({
                        'type': 'evaluation_failed',
                        'task_id': task_id,
                        'error': str(e)
                    })
                
                raise
        
        # Submit to executor
        future = self.executor.submit(wrapped_eval)
        task.future = future
        
        # Store task
        with self.lock:
            self.tasks[task_id] = task
        
        print(f"[TASK MANAGER] Created task {task_id[:8]} for {experiment_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Get task by ID"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, EvaluationTask]:
        """Get all tasks"""
        with self.lock:
            return dict(self.tasks)
    
    def get_running_tasks(self) -> Dict[str, EvaluationTask]:
        """Get all running tasks"""
        with self.lock:
            return {
                tid: task for tid, task in self.tasks.items()
                if task.status == TaskStatus.RUNNING
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task to cancel
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        with self.lock:
            task = self.tasks.get(task_id)
            
            if not task:
                return False
            
            if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return False
            
            # Try to cancel future
            if task.future and task.future.cancel():
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                print(f"[TASK {task_id[:8]}] Cancelled")
                
                # Broadcast cancellation
                if self.websocket_manager:
                    self.websocket_manager.broadcast({
                        'type': 'evaluation_cancelled',
                        'task_id': task_id
                    })
                
                return True
            
            return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24, keep_last_n: int = 10):
        """
        Clean up old completed/failed tasks.
        
        Args:
            max_age_hours: Delete tasks older than this
            keep_last_n: Keep at least this many recent tasks
        """
        with self.lock:
            # Sort by completion time
            completed_tasks = [
                (tid, task) for tid, task in self.tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task.completed_at is not None
            ]
            # Sort with explicit key handling None
            completed_tasks.sort(key=lambda x: x[1].completed_at or datetime.min, reverse=True)
            
            # Keep recent N
            to_keep = set(tid for tid, _ in completed_tasks[:keep_last_n])
            
            # Delete old tasks
            now = datetime.now()
            deleted = 0
            for tid, task in list(self.tasks.items()):
                if tid in to_keep:
                    continue
                
                if task.completed_at and (now - task.completed_at).total_seconds() > max_age_hours * 3600:
                    del self.tasks[tid]
                    deleted += 1
            
            if deleted > 0:
                print(f"[TASK MANAGER] Cleaned up {deleted} old tasks")
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor and cleanup"""
        print("[TASK MANAGER] Shutting down...")
        self.executor.shutdown(wait=wait)
        print("[TASK MANAGER] Shutdown complete")


# Global instance (initialized in api.py)
_task_manager: Optional[EvaluationTaskManager] = None


def get_task_manager() -> EvaluationTaskManager:
    """Get global task manager instance"""
    global _task_manager
    if _task_manager is None:
        raise RuntimeError("Task manager not initialized. Call init_task_manager() first.")
    return _task_manager


def init_task_manager(max_workers: int = 2, websocket_manager=None) -> EvaluationTaskManager:
    """Initialize global task manager"""
    global _task_manager
    if _task_manager is None:
        _task_manager = EvaluationTaskManager(max_workers=max_workers, websocket_manager=websocket_manager)
    return _task_manager
