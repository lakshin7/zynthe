"""
JSON serialization utilities for handling numpy arrays and other non-serializable types.
"""

import numpy as np
from typing import Any


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and other non-JSON-serializable objects 
    to JSON-serializable Python types.
    
    This function handles:
    - numpy arrays (converted to lists)
    - numpy scalar types (float64, int64, bool_, etc.)
    - nested dictionaries and lists
    - tuples (preserved as tuples)
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, etc.)
    
    Returns:
        JSON-serializable version of the object
        
    Examples:
        >>> import numpy as np
        >>> data = {'array': np.array([1, 2, 3]), 'value': np.float64(3.14)}
        >>> convert_to_serializable(data)
        {'array': [1, 2, 3], 'value': 3.14}
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        # Handle all numpy scalar types (float64, int64, bool_, etc.)
        return obj.item()
    else:
        return obj


def safe_json_dump(data: Any, file_handle, **kwargs):
    """
    Safe wrapper around json.dump that converts numpy types before serialization.
    
    Args:
        data: Data to serialize
        file_handle: File handle to write to
        **kwargs: Additional arguments passed to json.dump (e.g., indent=2)
        
    Example:
        >>> with open('output.json', 'w') as f:
        ...     safe_json_dump(metrics, f, indent=2)
    """
    import json
    serializable_data = convert_to_serializable(data)
    json.dump(serializable_data, file_handle, **kwargs)


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safe wrapper around json.dumps that converts numpy types before serialization.
    
    Args:
        data: Data to serialize
        **kwargs: Additional arguments passed to json.dumps (e.g., indent=2)
        
    Returns:
        JSON string
        
    Example:
        >>> json_str = safe_json_dumps(metrics, indent=2)
    """
    import json
    serializable_data = convert_to_serializable(data)
    return json.dumps(serializable_data, **kwargs)
