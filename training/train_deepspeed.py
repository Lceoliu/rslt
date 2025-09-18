from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import deepspeed  # type: ignore
except Exception:
    deepspeed = None

from model.config import load_config
from model.embedding import build_model_from_config


class DummyPartsDataset(Dataset):
    """Placeholder dataset. Replace with your real dataloader.
    Each item returns: {part: (T, K, 3)} and an integer label.
    """

    def __init__(self, length: int = 16, T: int = 32):
        import numpy as np

        self.length = length
        self.T = T
        self.K = {'body': 13, 'face': 68, 'left_hand': 21, 'right_hand': 21}
        self.nclass = 10
        self.np = np

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        parts = {}
        for k, v in self.K.items():
            arr = self.np.random.randn(self.T, v, 3).astype('float32')
            arr[..., 2] = self.np.random.rand(self.T, v).astype('float32')
            parts[k] = arr
        label = self.np.random.randint(0, self.nclass)
        return parts, int(label)


class Classifier(nn.Module):
    def __init__(self, embedder: nn.Module, embed_dim: int, nclass: int):
        super().__init__()
        self.embedder = embedder
        self.head = nn.Linear(embed_dim, nclass)

    def forward(self, parts):
        z = self.embedder(parts)
        return self.head(z)


def train(args):
    cfg = load_config(args.config)
    model = build_model_from_config(cfg)
    embed_dim = int(cfg['model'].get('embed_dim', 512))
    nclass = int(cfg.get('nclass', 10))
    net = Classifier(model, embed_dim, nclass)

    ds_config_path = Path(args.deepspeed_config)
    assert ds_config_path.exists(), f"DeepSpeed config not found: {ds_config_path}"

    if deepspeed is None:
        raise RuntimeError("DeepSpeed not installed. Install via: pip install deepspeed")

    engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=net, model_parameters=net.parameters(), config=str(ds_config_path)
    )

    dataset = DummyPartsDataset(length=32, T=32)
    loader = DataLoader(dataset, batch_size=1)  # sample-only; DeepSpeed handles global batch
    loss_fn = nn.CrossEntropyLoss()

    engine.train()
    for step, (parts, label) in enumerate(loader):
        # parts is dict[str, numpy]; pass through as-is; embedder handles numpy inputs
        logits = engine(parts)
        loss = loss_fn(logits, label.to(engine.device))
        engine.backward(loss)
        engine.step()
        if step % 10 == 0:
            engine.print(f"step={step} loss={loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/embedding_default.yaml')
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='configs/ds_config.json')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

