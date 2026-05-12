import pytest

def test_core_imports():
    """Smoke test to ensure core modules can be imported without errors."""
    import zynthe
    from zynthe import DistillationToolkit
    from zynthe.core.models.model_loader import ModelLoader
    from zynthe.core.distillers.base_distiller import BaseDistiller
    
    assert zynthe.__version__ is not None
    assert DistillationToolkit is not None
    assert ModelLoader is not None
    assert BaseDistiller is not None

def test_pipeline_imports():
    """Smoke test for pipeline modules."""
    from zynthe.core.pipelines import PipelineBuilder, PipelineRegistry
    assert PipelineBuilder is not None
    assert PipelineRegistry is not None
