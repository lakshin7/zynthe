
from zynthe.data.dataloaders import JsonlDataset, create_dataloaders


def test_jsonl_dataset():
    assert JsonlDataset is not None


def test_create_dataloaders():
    assert create_dataloaders is not None
