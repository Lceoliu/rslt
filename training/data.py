from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader


class DummyPartsDataset(Dataset):
    def __init__(self, length: int = 64, T: int = 32, nclass: int = 10):
        self.length = length
        self.T = T
        self.K = {
            'body': 13,
            'face': 68,
            'left_hand': 21,
            'right_hand': 21,
            'fullbody': 13 + 68 + 21 + 21,
        }
        self.nclass = nclass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        parts = {}
        for k, v in self.K.items():
            arr = np.random.randn(self.T, v, 3).astype('float32')
            arr[..., 2] = np.random.rand(self.T, v).astype('float32')
            parts[k] = arr
        label = int(np.random.randint(0, self.nclass))
        return parts, label


def parts_collate(batch):
    parts_list, labels = zip(*batch)
    keys = parts_list[0].keys()
    out = {}
    for k in keys:
        arrs = [p[k] for p in parts_list]
        arrs = [a if a.ndim == 3 else a.squeeze(0) for a in arrs]
        out[k] = np.stack(arrs, axis=0)
    return out, np.array(labels, dtype='int64')


def _maybe_import_builder(path: str):
    mod_name, func_name = path.rsplit(':', 1)
    mod = import_module(mod_name)
    return getattr(mod, func_name)


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    data_cfg = cfg.get('data', {})
    batch_size = int(data_cfg.get('batch_size', 8))
    num_workers = int(data_cfg.get('num_workers', 0))
    nclass = int(cfg.get('nclass', data_cfg.get('nclass', 10)))

    builder_path = data_cfg.get('custom_builder')
    if builder_path:
        builder = _maybe_import_builder(builder_path)
        return builder(cfg)

    # Prefer your dataset/my_dataset if provided
    ds_cfg = cfg.get('dataset')
    if ds_cfg and isinstance(ds_cfg, dict) and ds_cfg.get('data_dir'):
        # Late import to avoid heavy deps until needed
        from dataset.my_dataset import create_dataloader
        from dataset.transform import NormalizeProcessor

        transform = NormalizeProcessor()
        train_loader = create_dataloader(
            ds_cfg, split='train', transform=transform,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, verbose=True,
        )
        val_loader = create_dataloader(
            ds_cfg, split='val', transform=transform,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, verbose=True,
        )
        return train_loader, val_loader, nclass

    # Fallback to dummy loaders
    train_set = DummyPartsDataset(length=int(data_cfg.get('train_length', 128)), T=int(data_cfg.get('T', 32)), nclass=nclass)
    val_set = DummyPartsDataset(length=int(data_cfg.get('val_length', 64)), T=int(data_cfg.get('T', 32)), nclass=nclass)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=parts_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=parts_collate)
    return train_loader, val_loader, nclass
