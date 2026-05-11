"""
Download Progress Hook for HuggingFace Models
Tracks model download progress and reports via ProgressTracker
"""

from __future__ import annotations


import logging
import os
from typing import Optional, Any
from pathlib import Path

from zynthe.core.utils.progress_tracker import ProgressTracker, ProgressStage

logger = logging.getLogger(__name__)


class DownloadProgressCallback:
    """
    Callback for tracking HuggingFace model download progress.

    Usage:
        tracker = ProgressTracker(websocket_callback)
        callback = DownloadProgressCallback(tracker, is_teacher=True)
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=callback.cache_dir
        )
    """

    def __init__(
        self,
        progress_tracker: Optional[ProgressTracker] = None,
        is_teacher: bool = True,
        model_name: str = "model",
    ):
        self.tracker = progress_tracker
        self.is_teacher = is_teacher
        self.model_name = model_name
        self.cache_dir = Path.home() / ".cache" / "huggingface"
        self.initial_cache_size = self._get_cache_size()
        self.expected_size_mb: float | None = None

    def _get_cache_size(self) -> float:
        """Get current cache size in MB"""
        try:
            total = 0
            cache_path = self.cache_dir
            if cache_path.exists():
                for dirpath, dirnames, filenames in os.walk(cache_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total += os.path.getsize(fp)
            return total / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def set_expected_size(self, size_mb: float):
        """Set expected download size"""
        self.expected_size_mb = size_mb

    def check_progress(self):
        """Check download progress by comparing cache size"""
        if not self.tracker or not self.expected_size_mb:
            return

        current_size = self._get_cache_size()
        downloaded = current_size - self.initial_cache_size

        if downloaded > 0:
            self.tracker.download_progress(
                self.model_name, downloaded, self.expected_size_mb, self.is_teacher
            )


def load_model_with_progress(
    model_class,
    model_id: str,
    progress_tracker: Optional[ProgressTracker] = None,
    is_teacher: bool = True,
    expected_size_mb: Optional[float] = None,
    **kwargs,
) -> Any:
    """
    Load a HuggingFace model with progress tracking.

    Args:
        model_class: Model class (e.g., AutoModel, AutoModelForSequenceClassification)
        model_id: Model ID on HuggingFace Hub
        progress_tracker: ProgressTracker instance
        is_teacher: Whether this is the teacher model
        expected_size_mb: Expected download size in MB
        **kwargs: Additional arguments for from_pretrained

    Returns:
        Loaded model
    """
    role = "Teacher" if is_teacher else "Student"
    stage = ProgressStage.DOWNLOADING_TEACHER if is_teacher else ProgressStage.DOWNLOADING_STUDENT

    # Create download callback
    callback = DownloadProgressCallback(
        progress_tracker=progress_tracker, is_teacher=is_teacher, model_name=model_id
    )

    if expected_size_mb:
        callback.set_expected_size(expected_size_mb)

    # Start download stage
    if progress_tracker:
        progress_tracker.set_stage(stage, f"Downloading {role} Model: {model_id}")

    # Check if model is cached
    from transformers.utils.hub import cached_file

    try:
        # Try to get cached path
        cache_path = cached_file(
            model_id,
            "config.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )

        if cache_path and os.path.exists(cache_path):
            # Model is cached, skip download tracking
            if progress_tracker:
                progress_tracker.update(
                    stage, 1.0, f"{role} Model (Cached)", f"Using cached model: {model_id}"
                )
        else:
            # Model needs to be downloaded
            if progress_tracker:
                progress_tracker.update(
                    stage, 0.1, f"Starting {role} Model Download", f"Model: {model_id}"
                )
    except Exception:
        logger.debug("Download progress tracking failed")

    # Load the model
    try:
        model = model_class.from_pretrained(model_id, **kwargs)

        # Mark download complete
        if progress_tracker:
            progress_tracker.update(
                stage, 1.0, f"{role} Model Downloaded", f"Successfully loaded: {model_id}"
            )

        return model

    except Exception as e:
        if progress_tracker:
            progress_tracker.fail(f"Failed to load {role.lower()} model: {str(e)}")
        raise


def load_tokenizer_with_progress(
    tokenizer_class,
    model_id: str,
    progress_tracker: Optional[ProgressTracker] = None,
    is_teacher: bool = True,
    **kwargs,
) -> Any:
    """
    Load a tokenizer with progress tracking.

    Args:
        tokenizer_class: Tokenizer class (e.g., AutoTokenizer)
        model_id: Model ID on HuggingFace Hub
        progress_tracker: ProgressTracker instance
        is_teacher: Whether this is for the teacher model
        **kwargs: Additional arguments for from_pretrained

    Returns:
        Loaded tokenizer
    """
    role = "Teacher" if is_teacher else "Student"

    if progress_tracker:
        stage = (
            ProgressStage.DOWNLOADING_TEACHER if is_teacher else ProgressStage.DOWNLOADING_STUDENT
        )
        progress_tracker.update(stage, 0.5, f"Loading {role} Tokenizer", f"Model: {model_id}")

    tokenizer = tokenizer_class.from_pretrained(model_id, **kwargs)

    return tokenizer
