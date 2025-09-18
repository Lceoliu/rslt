import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import deepspeed  # type: ignore
except Exception:
    raise RuntimeError("DeepSpeed not installed. Install via: pip install deepspeed")

from model.config import load_config
from model.embedding import build_model_from_config
from training.data import build_dataloaders


class EmbedderWrapper(nn.Module):
    """Wraps the MultiPart embedder to accept both my_dataset dict batches and
    dummy (parts, label) style batches. Returns embeddings tensor.
    """

    def __init__(self, embedder: nn.Module):
        super().__init__()
        self.embedder = embedder

    def forward(self, batch_or_parts):
        # Case 1: dataset/my_dataset batch dict
        if isinstance(batch_or_parts, dict) and 'pose' in batch_or_parts:
            parts = batch_or_parts['pose']  # {part: Tensor[B,T,V,C]}
            pose_len = batch_or_parts.get('pose_len')  # Tensor[B]
            return self.embedder(parts, pose_len=pose_len)
        # Case 2: fallback tuple/list from DummyPartsDataset: (parts, label)
        if isinstance(batch_or_parts, (tuple, list)) and len(batch_or_parts) >= 1:
            parts = batch_or_parts[0]
            return self.embedder(parts)
        # Case 3: directly a parts dict
        return self.embedder(batch_or_parts)


def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def _make_dummy_loss(z: torch.Tensor, mode: str = 'none') -> torch.Tensor:
    """Return a placeholder loss for pipeline verification.
    - 'none': zero loss but keeps graph: (z * 0).sum()
    - 'l2': small L2 to exercise backward: 1e-6 * (z ** 2).mean()
    """
    if mode == 'l2':
        return 1e-6 * (z ** 2).mean()
    return (z * 0.0).sum()


def train(args):
    cfg = load_config(args.config)
    if args.train_config:
        train_cfg = load_config(args.train_config)
        cfg = _deep_update(cfg, train_cfg)

    # Build embedder and wrap for flexible batch handling
    embedder = build_model_from_config(cfg)
    net = EmbedderWrapper(embedder)

    ds_config_path = Path(args.deepspeed_config)
    assert ds_config_path.exists(), f"DeepSpeed config not found: {ds_config_path}"

    engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=net, model_parameters=net.parameters()
    )

    train_loader, val_loader, _ = build_dataloaders(cfg)

    epochs = int(cfg.get('train', {}).get('epochs', args.epochs))
    log_interval = int(cfg.get('train', {}).get('log_interval', 10))
    val_interval = int(cfg.get('train', {}).get('val_interval', 100))
    save_dir = Path(cfg.get('train', {}).get('save_dir', 'runs/ckpts'))
    save_dir.mkdir(parents=True, exist_ok=True)

    dummy_mode = str(cfg.get('train', {}).get('dummy_loss', 'none'))  # 'none' or 'l2'

    global_step = 0
    for epoch in range(epochs):
        engine.train()
        for step, batch in enumerate(train_loader):
            z = engine(batch)  # embeddings [B, D]
            loss = _make_dummy_loss(z, mode=dummy_mode)
            engine.backward(loss)
            engine.step()
            global_step += 1
            if global_step % log_interval == 0:
                with torch.no_grad():
                    mean_norm = z.norm(dim=1).mean().item()
                engine.print(f"epoch={epoch} step={global_step} loss={loss.item():.6f} |z|={mean_norm:.3f}")
            if global_step % val_interval == 0 and engine.global_rank == 0:
                eval_stat = evaluate(engine, val_loader)
                engine.print(f"[eval] step={global_step} mean_|z|={eval_stat:.3f}")
                engine.save_checkpoint(str(save_dir))

    if engine.global_rank == 0:
        eval_stat = evaluate(engine, val_loader)
        engine.print(f"[final eval] mean_|z|={eval_stat:.3f}")
        engine.save_checkpoint(str(save_dir))


@torch.no_grad()
def evaluate(engine, loader: DataLoader) -> float:
    engine.eval()
    norms = []
    for batch in loader:
        z = engine(batch)
        norms.append(z.norm(dim=1).mean().item())
    return float(sum(norms) / max(1, len(norms)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/embedding_default.yaml')
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='configs/ds_config.json')
    parser.add_argument('--train_config', type=str, default='')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1, help='for deepspeed')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
