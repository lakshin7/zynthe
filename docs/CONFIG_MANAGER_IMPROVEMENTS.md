# ConfigManager and YAML Configuration Improvements

## Date: October 18, 2025

## Summary of Changes

This document outlines the improvements made to the `ConfigManager` and the default YAML configuration file to ensure robustness, clarity, and proper functionality.

---

## Issues Identified and Fixed

### 1. **Incorrect Default Config Path**

**Issue:** The `ConfigManager` was looking for `default.yaml` in `core/config/` directory, but the file actually exists in `configs/` directory. This would cause initialization failures.

**Fix:** Updated the `__init__` method to correctly construct the path to `configs/default.yaml` by going two directories up from `core/config/`:

```python
# Look for default.yaml in configs/ directory (2 levels up from core/config/)
if defaults_path is None:
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
    defaults_path = os.path.join(config_dir, _DEFAULTS_FILENAME)
```

### 2. **Missing `early_stop_patience` Default**

**Issue:** The `early_stop_patience` parameter was used in YAML files and by the `Trainer` class, but not set in the ConfigManager's default train configuration.

**Fix:** Added `early_stop_patience` to the default train configuration and ensured it's properly coerced to an integer:

```python
default_train = {
    # ... other defaults ...
    "early_stop_patience": 2,
}

# Coercion
merged_train["early_stop_patience"] = int(merged_train.get("early_stop_patience", 2))
```

### 3. **Missing Quantization and Similarity Transfer Handling**

**Issue:** The YAML file contained `quantization` and `similarity_transfer` sections, but the ConfigManager didn't validate or provide defaults for these.

**Fix:** Added proper handling for these sections in `_resolve_runtime()`:

```python
# quantization defaults
quant_cfg = self.raw_config.get("quantization", {})
default_quant = {
    "enable": False,
    "mode": "ptq",  # ptq or qat
}
merged_quant = _deep_update(default_quant, quant_cfg)

# similarity transfer default
similarity_transfer = bool(self.raw_config.get("similarity_transfer", False))

# Add to resolved config
self.resolved_config["quantization"] = merged_quant
self.resolved_config["similarity_transfer"] = similarity_transfer
```

### 4. **Insufficient Logging**

**Issue:** The ConfigManager had minimal logging, making debugging difficult.

**Fix:** Added informative logging throughout the loading process:

```python
logger.info(f"Loaded default config from: {self.defaults_path}")
logger.info(f"Loaded user config from: {self.config_path}")
logger.warning(f"Default config not found at: {self.defaults_path}, using minimal defaults")
```

### 5. **YAML File Lacks Documentation**

**Issue:** The original `default.yaml` had minimal comments and was missing several important fields.

**Fix:** Completely rewrote `default.yaml` with:
- Comprehensive comments explaining each parameter
- Structured sections with clear headers
- Additional optional parameters (commented out) for advanced use
- Better organization and readability

---

## New Features Added

### 1. **Nested Configuration Access**

Added `get_nested()` method for safe access to deeply nested configuration values:

```python
def get_nested(self, *keys, default: Any = None) -> Any:
    """
    Safely get nested config values using dot notation.
    
    Example:
        cfg.get_nested("model", "name") -> returns cfg["model"]["name"]
        cfg.get_nested("train", "lr", default=1e-5)
    """
```

**Usage:**
```python
lr = cfg.get_nested("train", "lr", default=5e-5)
model_name = cfg.get_nested("model", "name")
```

### 2. **Path Validation**

Added `validate_required_paths()` method to check if data files exist before training:

```python
def validate_required_paths(self):
    """
    Validate that required data paths exist.
    """
```

**Usage:**
```python
cfg = ConfigManager(config_path="configs/default.yaml")
cfg.validate_required_paths()  # Raises ConfigError if files missing
```

### 3. **Better String Representation**

Added `__repr__()` method for easier debugging:

```python
def __repr__(self) -> str:
    """String representation for debugging."""
    return (
        f"ConfigManager(\n"
        f"  experiment_id={self.experiment_id},\n"
        f"  device={self.get_runtime().get('device', 'unknown')},\n"
        f"  config_path={self.config_path},\n"
        f"  defaults_path={self.defaults_path}\n"
        f")"
    )
```

---

## Enhanced YAML Configuration

### New Sections Added:

1. **`train.save_best_only`**: Only save checkpoints that improve validation loss
2. **`train.log_interval`**: Control logging frequency during training
3. **`model.num_labels`**: Specify number of output classes
4. **`explainability`**: Complete explainability configuration (SHAP/LIME)
5. **Commented optional sections**: TensorBoard, advanced distillation params, checkpointing

### Improved Documentation:

- Every parameter now has an inline comment explaining its purpose
- Section headers with clear descriptions
- Examples of optional parameters for advanced users
- Better default values with rationale

---

## Configuration Validation

The ConfigManager now validates:

1. ✅ Required sections exist (`train`, `model`, `distillation`, `data`)
2. ✅ Required keys within each section
3. ✅ Type coercion (epochs, batch_size, lr, etc.)
4. ✅ Device availability (MPS, CUDA, CPU)
5. ✅ Data file paths (when `validate_required_paths()` is called)
6. ✅ Proper defaults for missing optional fields

---

## Backward Compatibility

All changes maintain backward compatibility:
- Existing YAML files will continue to work
- New optional fields have sensible defaults
- The API remains unchanged for existing code

---

## Testing Recommendations

To verify the improvements:

```python
# Test 1: Load with default config
from core.config.config_manager import ConfigManager
cfg = ConfigManager()
print(cfg)

# Test 2: Load with custom config
cfg = ConfigManager(config_path="configs/default.yaml")
cfg.validate_required_paths()

# Test 3: Access nested values
lr = cfg.get_nested("train", "lr")
device = cfg.device()

# Test 4: Check resolved config
print(cfg.resolved_config["train"])
print(cfg.resolved_config["quantization"])
```

---

## Best Practices Going Forward

1. **Always validate paths** before training:
   ```python
   cfg.validate_required_paths()
   ```

2. **Use `get_nested()` for safe access**:
   ```python
   value = cfg.get_nested("section", "key", default=fallback)
   ```

3. **Check resolved config** to see final merged values:
   ```python
   print(cfg.resolved_config)
   ```

4. **Enable logging** to see config loading process:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

---

## Related Files

- **ConfigManager**: `core/config/config_manager.py`
- **Default Config**: `configs/default.yaml`
- **Other Configs**: `configs/advanced.yaml`, `configs/mac_m2_test.yaml`, etc.

---

## Conclusion

The ConfigManager is now more robust, user-friendly, and production-ready. It includes:
- ✅ Proper path resolution
- ✅ Comprehensive validation
- ✅ Better error messages
- ✅ Enhanced logging
- ✅ Utility methods for common operations
- ✅ Well-documented YAML configuration

These improvements ensure a smoother development experience and reduce configuration-related errors.
