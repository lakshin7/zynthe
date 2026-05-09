"""
Real-time Download Progress Tracker for HuggingFace Models
Monitors model downloads and emits progress updates
"""

import sys
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import threading
import time
from functools import wraps


class DownloadProgressMonitor:
    """
    Monitors HuggingFace model downloads and emits progress updates.
    
    Usage:
        monitor = DownloadProgressMonitor(model_id="bert-base-uncased", role="teacher")
        # Progress updates are automatically printed to stdout for subprocess parsing
    """
    
    def __init__(self, model_id: str, role: str = "teacher"):
        self.model_id = model_id
        self.role = role  # "teacher" or "student"
        self.start_time = time.time()
        self.last_log_time: float = 0.0
        self.log_interval = 0.5  # Log every 0.5 seconds minimum
        
    def log_progress(self, progress: float, message: str, force: bool = False):
        """
        Log progress in a format that training_manager can parse.
        Throttles logs to avoid flooding output.
        """
        current_time = time.time()
        
        # Skip if too soon since last log (unless forced)
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
            
        stage = f"downloading_{self.role}"
        # Clamp progress between 0 and 1
        progress = max(0.0, min(1.0, progress))
        print(f"[PROGRESS] stage={stage} progress={progress:.3f} message={message}", flush=True)
        self.last_log_time = current_time


def check_model_cached(model_id: str) -> bool:
    """Check if a model is already cached locally"""
    try:
        from transformers.utils.hub import cached_file
        cached_path = cached_file(
            model_id,
            "config.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False
        )
        return cached_path is not None and Path(cached_path).exists()
    except Exception:
        return False


def get_cache_dir_for_model(model_id: str) -> Optional[Path]:
    """Get the cache directory path for a specific model"""
    try:
        cache_home = Path.home() / ".cache" / "huggingface" / "hub"
        # Convert model ID to cache directory name (e.g., "bert-base-uncased" -> "models--bert-base-uncased")
        safe_model_id = model_id.replace("/", "--")
        model_cache = cache_home / f"models--{safe_model_id}"
        return model_cache if model_cache.exists() else None
    except Exception:
        return None


def monitor_cache_directory(model_id: str, role: str, timeout: int = 300):
    """
    Monitor the HuggingFace cache directory for download progress.
    This runs in a background thread while the model is being downloaded.
    """
    monitor = DownloadProgressMonitor(model_id, role)
    cache_dir = get_cache_dir_for_model(model_id)
    
    if cache_dir is None:
        # Cache directory doesn't exist yet, wait for it
        cache_home = Path.home() / ".cache" / "huggingface" / "hub"
        safe_model_id = model_id.replace("/", "--")
        cache_dir = cache_home / f"models--{safe_model_id}"
        
    initial_size = 0
    start_time = time.time()
    last_size = 0
    stall_count = 0
    
    while True:
        if (time.time() - start_time) > timeout:
            monitor.log_progress(0.99, "Download taking longer than expected...", force=True)
            break
            
        try:
            if cache_dir.exists():
                # Calculate total size of all files in cache directory
                current_size = sum(
                    f.stat().st_size 
                    for f in cache_dir.rglob('*') 
                    if f.is_file() and not f.name.endswith('.lock')
                )
                
                if initial_size == 0:
                    initial_size = current_size
                    
                downloaded = current_size - initial_size
                
                # Estimate progress (rough - we don't know total size)
                # Typical model sizes: 100MB to 1GB+
                # Use logarithmic progress estimation
                if downloaded > 0:
                    # Progress increases faster initially, then slows
                    # This gives better UX than linear unknown progress
                    mb_downloaded = downloaded / (1024 * 1024)
                    progress = min(0.95, 0.1 + (0.85 * (mb_downloaded / (mb_downloaded + 100))))
                    
                    # Calculate download speed
                    elapsed = time.time() - start_time
                    speed_mbps = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0.1 else 0
                    
                    monitor.log_progress(
                        progress,
                        f"Downloading {model_id}: {mb_downloaded:.1f}MB ({speed_mbps:.1f}MB/s)"
                    )
                    
                    # Detect stalls
                    if current_size == last_size:
                        stall_count += 1
                    else:
                        stall_count = 0
                    last_size = current_size
                    
                    # If stalled for 10 checks, assume download is complete
                    if stall_count >= 10:
                        break
                        
        except Exception as e:
            print(f"[DEBUG] Cache monitoring error: {e}", file=sys.stderr)
            
        time.sleep(0.5)


def wrap_model_loading(model_loader_func: Callable, model_id: str, *args, role: str = "teacher", **kwargs):
    """
    Wrapper function that monitors model loading progress.
    
    Args:
        model_loader_func: The function to load the model (e.g., AutoModel.from_pretrained)
        model_id: HuggingFace model ID
        role: "teacher" or "student"
        *args, **kwargs: Additional arguments to pass to the loader
    
    Returns:
        Loaded model
    """
    monitor = DownloadProgressMonitor(model_id, role)
    
    # Check if model is cached
    is_cached = check_model_cached(model_id)
    
    if is_cached:
        monitor.log_progress(0.5, f"{model_id} found in cache, loading...", force=True)
        try:
            model = model_loader_func(model_id, *args, **kwargs)
            monitor.log_progress(1.0, f"{model_id} loaded from cache", force=True)
            return model
        except Exception as e:
            monitor.log_progress(0.0, f"Failed to load {model_id}: {str(e)}", force=True)
            raise
    else:
        # Model needs to be downloaded - start monitoring in background
        monitor.log_progress(0.05, f"Starting download of {model_id}", force=True)
        
        # Start cache monitoring thread
        monitor_thread = threading.Thread(
            target=monitor_cache_directory,
            args=(model_id, role),
            daemon=True
        )
        monitor_thread.start()
        
        # Load the model (this will download it)
        try:
            model = model_loader_func(model_id, *args, **kwargs)
            monitor.log_progress(1.0, f"{model_id} downloaded and loaded", force=True)
            return model
        except Exception as e:
            monitor.log_progress(0.0, f"Failed to download {model_id}: {str(e)}", force=True)
            raise


# Global state for monkey-patching
_patched_classes: Dict[str, Any] = {}


def install_progress_hooks():
    """
    Install progress monitoring hooks into transformers library.
    Call this before loading any models.
    """
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification
        
        # Patch AutoModel.from_pretrained
        for cls_name, cls in [
            ('AutoModel', AutoModel),
            ('AutoModelForSequenceClassification', AutoModelForSequenceClassification),
        ]:
            if cls_name not in _patched_classes:
                original_from_pretrained = cls.from_pretrained
                _patched_classes[cls_name] = original_from_pretrained
                
                def make_wrapper(original_method):
                    @wraps(original_method)
                    def from_pretrained_with_progress(pretrained_model_name_or_path, *args, _monitor_role='model', **kwargs):
                        """Wrapped from_pretrained with progress tracking"""
                        # Ensure model_id is a string, not a class object
                        if not isinstance(pretrained_model_name_or_path, str):
                            # Fallback: call original without monitoring
                            return original_method(pretrained_model_name_or_path, *args, **kwargs)
                        
                        return wrap_model_loading(
                            original_method,
                            pretrained_model_name_or_path,
                            *args,
                            role=_monitor_role,
                            **kwargs
                        )
                    return from_pretrained_with_progress
                
                cls.from_pretrained = classmethod(make_wrapper(original_from_pretrained))
        
        print("[DEBUG] Progress hooks installed successfully", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] Failed to install progress hooks: {e}", file=sys.stderr)


def uninstall_progress_hooks():
    """Restore original from_pretrained methods"""
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification
        
        for cls_name, cls in [
            ('AutoModel', AutoModel),
            ('AutoModelForSequenceClassification', AutoModelForSequenceClassification),
        ]:
            if cls_name in _patched_classes:
                cls.from_pretrained = _patched_classes[cls_name]
                del _patched_classes[cls_name]
        
        print("[DEBUG] Progress hooks uninstalled", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] Failed to uninstall progress hooks: {e}", file=sys.stderr)

