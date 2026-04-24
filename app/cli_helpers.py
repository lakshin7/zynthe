"""
Shared CLI helper functions for Zynthe.
Extracted to reduce duplication between main.py and main_new.py
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from core.config.config_manager import ConfigManager


LOG = logging.getLogger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    """
    Parse CLI override arguments in KEY=VALUE format.
    
    Args:
        override_list: List of strings like "train.epochs=3" or "data.batch_size=16"
        
    Returns:
        Nested dict structure matching config hierarchy
    """
    overrides: Dict[str, Any] = {}  # type: ignore[var-annotated]
    for override in override_list:
        if '=' not in override:
            LOG.warning(f"Invalid override format '{override}', expected KEY=VALUE")
            continue
        key, value = override.split('=', 1)
        try:
            if '.' not in value and value.isdigit():
                value = int(value)  # type: ignore[assignment]
            elif value.replace('.', '').replace('-', '').isdigit():
                value = float(value)  # type: ignore[assignment]
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'  # type: ignore[assignment]
        except ValueError:
            pass
        keys = key.split('.')
        current = overrides
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return overrides


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


def use_teacher_agent(config: Dict[str, Any]) -> bool:
    """
    Check if teacher agent is enabled in config.
    
    Args:
        config: Resolved configuration dict
        
    Returns:
        True if teacher agent should be used
    """
    agentic_cfg = config.get('agentic', {}) if isinstance(config, dict) else {}
    return bool(agentic_cfg.get('enable_teacher_agent', False))