
from zynthe.data.preprocess import PreprocessConfig, apply_preprocess_pipeline


def test_preprocess_config():
    config = PreprocessConfig()
    assert config is not None


def test_apply_preprocess():
    dataset = [{"text": "Hello, World!"}]
    config = PreprocessConfig()
    result, stats = apply_preprocess_pipeline(dataset, config)
    assert len(result) == 1
