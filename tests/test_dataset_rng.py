import torch

from dataset.my_dataset import MyDataset


def test_make_rng_changes_with_epoch() -> None:
    ds = object.__new__(MyDataset)
    ds._base_seed = 123
    ds._epoch = 0
    def fake_get_worker_info():
        class Info:
            def __init__(self):
                self.seed = 999
        return Info()
    original = torch.utils.data.get_worker_info
    try:
        torch.utils.data.get_worker_info = fake_get_worker_info  # type: ignore
        rng0 = ds._make_rng(idx=5)
        v0 = int(rng0.integers(0, 1_000_000))
        ds.set_epoch(1)
        rng1 = ds._make_rng(idx=5)
        v1 = int(rng1.integers(0, 1_000_000))
    finally:
        torch.utils.data.get_worker_info = original  # type: ignore
    assert v0 != v1
