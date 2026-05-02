# tests/test_cpu_smoke.py
"""
CPU-safe smoke tests for Latitude 7490.
These tests should complete in < 5 minutes on 8GB RAM CPU-only hardware.
"""
import torch
from pathlib import Path
from core.config.config_manager import ConfigManager


class TestConfigLoading:
    """Test configuration loading."""
    
    def test_cpu_smoke_config_loads(self):
        """Config file should load without errors."""
        config_path = Path("configs/cpu_smoke_test.yaml")
        assert config_path.exists(), "CPU smoke test config not found"
        
        config_manager = ConfigManager(config_path=str(config_path))
        assert config_manager is not None
    
    def test_cpu_smoke_config_has_correct_structure(self):
        """Config should have required sections."""
        config_path = Path("configs/cpu_smoke_test.yaml")
        config_manager = ConfigManager(config_path=str(config_path))
        config = config_manager.resolved_config
        
        # Check required sections
        assert "model" in config
        assert "data" in config
        assert "train" in config
        assert "device" in config
        
        # Check CPU-specific settings
        assert config["train"]["epochs"] == 1
        assert config["train"]["batch_size"] == 4
        assert config["device"]["cpu_only"]


class TestModelLoading:
    """Test model loading on CPU."""
    
    def test_tiny_bert_loads_on_cpu(self):
        """Tiny BERT should load on CPU without OOM."""
        config_path = Path("configs/cpu_smoke_test.yaml")
        config_manager = ConfigManager(config_path=str(config_path))
        
        # This should not raise OOM error
        from core.models.model_loader import load_models
        teacher, student, tokenizer = load_models(
            config_manager.resolved_config,
            device=torch.device("cpu")
        )
        
        assert teacher is not None
        assert student is not None
        assert tokenizer is not None
        
        # Check model sizes (should be small)
        teacher_params = sum(p.numel() for p in teacher.parameters())
        assert teacher_params < 10_000_000  # < 10M params


class TestCLI:
    """Test CLI commands."""
    
    def test_info_command(self):
        """CLI info command should work."""
        import subprocess
        result = subprocess.run(
            ["python", "app/main.py", "info", "--config", "configs/cpu_smoke_test.yaml"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "model" in result.stdout.lower()
