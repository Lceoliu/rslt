from ..dataset.my_dataset import MyDataset
from ..dataset.transform import NormalizeProcessor
from pathlib import Path
import pytest
import torch
import numpy as np

test_data_path = Path(__file__).parent.parent / 'dataset' / 'test_data' / 'shm_overfit'
assert test_data_path.exists(), f"Test data path {test_data_path} does not exist."
test_config = {
    'data_dir': str(test_data_path),
    'window': 32,
    'stride': 16,
    'min_reserved_ratio': 0.6,
    'pad_last': True,
}
transformer = NormalizeProcessor()


def test_dataset_loading():
    dataset = MyDataset(test_config, split='train', transform=transformer)

    print("Dataset loading test passed.")
