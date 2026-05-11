"""
Progress Tracker for Model Downloads and Training
Provides real-time progress updates via WebSocket
"""

from __future__ import annotations


import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class ProgressStage(Enum):
    """Training pipeline stages"""

    INITIALIZING = "initializing"
    DOWNLOADING_TEACHER = "downloading_teacher"
    DOWNLOADING_STUDENT = "downloading_student"
    LOADING_TEACHER = "loading_teacher"
    LOADING_STUDENT = "loading_student"
    LOADING_DATA = "loading_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SAVING = "saving"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    """Progress update message"""

    stage: str
    progress: float  # 0.0 to 1.0
    message: str
    detail: Optional[str] = None
    estimated_time_remaining: Optional[int] = None  # seconds
    current_step: Optional[int] = None
    total_steps: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ProgressTracker:
    """
    Tracks progress through the training pipeline.
    Sends updates via callback (typically WebSocket broadcast)
    """

    def __init__(self, callback: Optional[Callable] = None):
        """
        Args:
            callback: Function to call with progress updates (e.g., websocket.send)
        """
        self.callback = callback
        self.current_stage = ProgressStage.INITIALIZING
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.history: list[dict[str, Any]] = []

    def update(
        self,
        stage: ProgressStage,
        progress: float,
        message: str,
        detail: Optional[str] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ):
        """Send a progress update"""

        # Calculate estimated time remaining
        elapsed = time.time() - self.stage_start_time
        if progress > 0 and progress < 1.0:
            estimated_remaining = int((elapsed / progress) * (1.0 - progress))
        else:
            estimated_remaining = None

        update = ProgressUpdate(
            stage=stage.value,
            progress=progress,
            message=message,
            detail=detail,
            estimated_time_remaining=estimated_remaining,
            current_step=current_step,
            total_steps=total_steps,
        )

        # Record in history
        self.history.append({"timestamp": time.time(), "update": update.to_dict()})

        # Send update via callback
        if self.callback:
            self.callback({"type": "progress", "data": update.to_dict()})

        # Update current stage
        if stage != self.current_stage:
            self.current_stage = stage
            self.stage_start_time = time.time()

    def set_stage(self, stage: ProgressStage, message: str):
        """Change stage (convenience method)"""
        self.update(stage, 0.0, message)

    def download_progress(
        self, model_name: str, downloaded_mb: float, total_mb: float, is_teacher: bool = True
    ):
        """Update download progress"""
        stage = (
            ProgressStage.DOWNLOADING_TEACHER if is_teacher else ProgressStage.DOWNLOADING_STUDENT
        )
        progress = downloaded_mb / total_mb if total_mb > 0 else 0.0

        role = "Teacher" if is_teacher else "Student"
        message = f"Downloading {role} Model: {model_name}"
        detail = f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB ({progress*100:.1f}%)"

        self.update(stage, progress, message, detail)

    def training_progress(
        self,
        epoch: int,
        total_epochs: int,
        batch: int,
        total_batches: int,
        loss: Optional[float] = None,
    ):
        """Update training progress"""
        # Calculate overall progress
        epoch_progress = (epoch - 1) / total_epochs
        batch_progress = batch / total_batches / total_epochs
        overall_progress = epoch_progress + batch_progress

        message = f"Training Epoch {epoch}/{total_epochs}"
        detail = f"Batch {batch}/{total_batches}"
        if loss is not None:
            detail += f" - Loss: {loss:.4f}"

        self.update(
            ProgressStage.TRAINING,
            overall_progress,
            message,
            detail,
            current_step=batch,
            total_steps=total_batches,
        )

    def complete(self, message: str = "Training Complete!"):
        """Mark as complete"""
        self.update(ProgressStage.COMPLETE, 1.0, message)

    def fail(self, error: str):
        """Mark as failed"""
        self.update(ProgressStage.FAILED, 0.0, "Training Failed", detail=error)

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        elapsed = time.time() - self.start_time
        return {
            "current_stage": self.current_stage.value,
            "elapsed_seconds": int(elapsed),
            "total_updates": len(self.history),
            "history": self.history[-10:],  # Last 10 updates
        }


# Convenience function for creating tracker
def create_progress_tracker(websocket_callback: Optional[Callable] = None) -> ProgressTracker:
    """Create a new progress tracker"""
    return ProgressTracker(callback=websocket_callback)
