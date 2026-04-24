"""
Shared CLI helper functions for Zynthe.
Extracted to reduce duplication between main.py and main_new.py
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from core.config.config_manager import ConfigManager


LOG = logging.getLogger(__name__)


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> ConfigManager:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        overrides: Optional dict of override values
        
    Returns:
        ConfigManager instance
    """
    LOG.info(f"Loading config from {config_path}")
    config_manager = ConfigManager(config_path=config_path, overrides=overrides)
    return config_manager


def detect_device(prefer_mps: bool = False) -> torch.device:
    """
    Detect best available device.
    
    Args:
        prefer_mps: Whether to prefer MPS (Apple Silicon)
        
    Returns:
        torch.device object
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def format_device_info(device: torch.device) -> str:
    """
    Format device information for display.
    
    Args:
        device: torch.device object
        
    Returns:
        Formatted string with device info
    """
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)} ({device.index})"
    elif device.type == "mps":
        return "Apple Silicon MPS"
    else:
        return "CPU"


def validate_config_path(config_path: str) -> bool:
    """
    Validate that config file exists.
    
    Args:
        config_path: Path to config file
        
    Returns:
        True if valid
    """
    path = Path(config_path)
    if not path.exists():
        LOG.error(f"Config file not found: {config_path}")
        return False
    if not path.is_file():
        LOG.error(f"Config path is not a file: {config_path}")
        return False
    return True