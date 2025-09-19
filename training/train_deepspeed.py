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
from model.adapter import VisualAdapter
from model.LLM_wrapper import LLMWithVisualPrefix
from training.data import build_dataloaders


class VLLMTrainer(nn.Module):
    """Composite module: MultiPart embedder + adapter + LLM CE loss.

    Forward returns a scalar CE loss computed by the LLM.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        # Build visual embedder
        self.embedder = build_model_from_config(cfg)
        embed_dim = int(cfg.get('model', {}).get('embed_dim', 512))

        # Build LLM
        llm_cfg = cfg.get('llm', {})
        model_name = llm_cfg.get('model_name_or_path', 'Qwen/Qwen2.5-0.5B')
        trust_remote_code = bool(llm_cfg.get('trust_remote_code', True))
        max_text_len = int(llm_cfg.get('max_text_len', 128))
        gradient_checkpointing = bool(llm_cfg.get('gradient_checkpointing', False))
        freeze_lm = bool(llm_cfg.get('freeze_lm', False))

        self.llm = LLMWithVisualPrefix(
            model_name_or_path=model_name,
            trust_remote_code=trust_remote_code,
            max_text_len=max_text_len,
            gradient_checkpointing=gradient_checkpointing,
            freeze_lm=freeze_lm,
        )
        llm_dim = self.llm.hidden_size

        # Build adapter to project visual embedding to LLM hidden size
        num_prefix_tokens = int(llm_cfg.get('num_prefix_tokens', 1))
        hidden_dim = int(llm_cfg.get('adapter_hidden', max(512, embed_dim)))
        self.adapter = VisualAdapter(
            in_dim=embed_dim,
            llm_dim=llm_dim,
            num_prefix_tokens=num_prefix_tokens,
            hidden_dim=hidden_dim,
        )

    def forward(self, batch):
        # Accept my_dataset dict or fallback tuple/list
        if isinstance(batch, (tuple, list)) and len(batch) >= 1 and not isinstance(batch[0], (dict,)):
            # Fallback dummy case: (parts, label)
            parts = batch[0]
            texts = None
            pose_len = None
        elif isinstance(batch, dict):
            parts = batch['pose']
            texts = batch.get('text', None)
            pose_len = batch.get('pose_len', None)
        else:
            parts = batch
            texts = None
            pose_len = None

        z = self.embedder(parts, pose_len=pose_len)  # [B, D]
        prefix = self.adapter(z)  # [B, P, E]

        if texts is None:
            # No texts -> return small L2 to exercise backward
            return 1e-6 * (z ** 2).mean()

        loss = self.llm(prefix, texts)
        return loss


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

    # Build composite trainer module (embedder + adapter + LLM)
    net = VLLMTrainer(cfg)

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
            loss = engine(batch)  # scalar CE loss from LLM
            engine.backward(loss)
            engine.step()
            global_step += 1
            if global_step % log_interval == 0 and engine.global_rank == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")
            if global_step % val_interval == 0 and engine.global_rank == 0:
                eval_stat = evaluate(engine, val_loader)
                print(f"[eval] step={global_step} val_loss={eval_stat:.6f}")
                engine.save_checkpoint(str(save_dir))

    if engine.global_rank == 0:
        eval_stat = evaluate(engine, val_loader)
        print(f"[final eval] val_loss={eval_stat:.6f}")
        engine.save_checkpoint(str(save_dir))


@torch.no_grad()
def evaluate(engine, loader: DataLoader) -> float:
    engine.eval()
    losses = []
    for batch in loader:
        loss = engine(batch)
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


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
