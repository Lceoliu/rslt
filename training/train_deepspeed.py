from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import deepspeed  # type: ignore
except Exception:
    deepspeed = None

from model.config import load_config
from model.embedding import build_model_from_config
from training.data import build_dataloaders
from training.utils import accuracy


class Classifier(nn.Module):
    def __init__(self, embedder: nn.Module, embed_dim: int, nclass: int):
        super().__init__()
        self.embedder = embedder
        self.head = nn.Linear(embed_dim, nclass)

    def forward(self, parts):
        z = self.embedder(parts)
        return self.head(z)


def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def train(args):
    cfg = load_config(args.config)
    if args.train_config:
        train_cfg = load_config(args.train_config)
        cfg = _deep_update(cfg, train_cfg)
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

    train_loader, val_loader, _ = build_dataloaders(cfg)
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(cfg.get('train', {}).get('epochs', args.epochs))
    log_interval = int(cfg.get('train', {}).get('log_interval', 10))
    val_interval = int(cfg.get('train', {}).get('val_interval', 100))
    save_dir = Path(cfg.get('train', {}).get('save_dir', 'runs/ckpts'))
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(epochs):
        engine.train()
        for step, (parts, label) in enumerate(train_loader):
            logits = engine(parts)
            labels_t = torch.as_tensor(label, dtype=torch.long, device=engine.device)
            loss = loss_fn(logits, labels_t)
            engine.backward(loss)
            engine.step()
            global_step += 1
            if global_step % log_interval == 0:
                top1, = accuracy(logits.detach(), labels_t, topk=(1,))
                engine.print(f"epoch={epoch} step={global_step} loss={loss.item():.4f} top1={top1.item():.2f}")
            if global_step % val_interval == 0 and engine.global_rank == 0:
                eval_top1 = evaluate(engine, val_loader)
                engine.print(f"[eval] step={global_step} top1={eval_top1:.2f}")
                engine.save_checkpoint(str(save_dir))

    if engine.global_rank == 0:
        eval_top1 = evaluate(engine, val_loader)
        engine.print(f"[final eval] top1={eval_top1:.2f}")
        engine.save_checkpoint(str(save_dir))


@torch.no_grad()
def evaluate(engine, loader: DataLoader) -> float:
    engine.eval()
    correct = 0
    total = 0
    for parts, label in loader:
        logits = engine(parts)
        pred = logits.argmax(dim=1)
        labels_t = torch.as_tensor(label, dtype=torch.long, device=engine.device)
        total += labels_t.size(0)
        correct += (pred == labels_t).sum().item()
    return 100.0 * correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/embedding_default.yaml')
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='configs/ds_config.json')
    parser.add_argument('--train_config', type=str, default='')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

