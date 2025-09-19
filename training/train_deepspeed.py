import os

ENABLE_DEBUG = False  # Enable to get more debug info
if ENABLE_DEBUG:
    import debugpy

    if os.environ.get('RANK') == '0' or os.environ.get('LOCAL_RANK') == '0':
        debugpy.listen(('0.0.0.0', 5678))
        print(
            f"Process with RANK {os.environ.get('RANK', 'N/A')} is listening on port 5678. Waiting for debugger attach..."
        )
        debugpy.wait_for_client()
        print("Debugger attached to RANK 0.")


import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

try:
    import deepspeed  # type: ignore
except Exception:
    raise RuntimeError("DeepSpeed not installed. Install via: pip install deepspeed")

from model.config import load_config
from model.embedding import build_model_from_config
from model.adapter import VisualAdapter
from model.LLM_wrapper import LLMWithVisualPrefix
from training.data import build_dataloaders
from utils.set_seed import set_seed


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
        model_name = llm_cfg.get('model_name_or_path', '../Qwen2.5-0.5B')
        print(f"Building LLM: {model_name}")
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
            print(
                "Warning: using fallback batch format (parts, label). No text loss will be computed."
            )
        elif isinstance(batch, dict):
            parts = batch['pose']
            texts = batch.get('text', None)
            pose_len = batch.get('pose_len', None)
        else:
            parts = batch
            texts = None
            pose_len = None
        assert isinstance(parts, dict), f"Expected parts dict, got {type(parts)}"
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
    seed = int(cfg.get('train', {}).get('seed', 42))
    # Build dataloaders first (dataset may spawn workers later)
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Build composite trainer module (embedder + adapter + LLM)
    net = VLLMTrainer(cfg)

    ds_config_path = Path(args.deepspeed_config)
    assert ds_config_path.exists(), f"DeepSpeed config not found: {ds_config_path}"

    engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=net, model_parameters=net.parameters()
    )

    # Prepare run directory with timestamp
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(cfg.get('train', {}).get('save_dir_root', 'runs')) / ts
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer (rank0 only)
    writer = None
    if engine.global_rank == 0 and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(run_dir))
        # Save a copy of config for reproducibility
        try:
            import yaml  # type: ignore

            with open(run_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        except Exception:
            pass

    epochs = int(cfg.get('train', {}).get('epochs', args.epochs))
    log_interval = int(cfg.get('train', {}).get('log_interval', 10))
    val_interval = int(cfg.get('train', {}).get('val_interval', 100))

    global_step = 0
    for epoch in range(epochs):
        engine.train()
        iterable = train_loader
        if engine.global_rank == 0 and tqdm is not None:
            iterable = tqdm(train_loader, desc=f'train epoch {epoch}', leave=False)
        for step, batch in enumerate(iterable):
            loss = engine(batch)  # scalar CE loss from LLM
            engine.backward(loss)
            engine.step()
            global_step += 1
            if writer is not None:
                writer.add_scalar('train/loss', float(loss.item()), global_step)
                # Try to log LR
                try:
                    if hasattr(engine, 'optimizer') and engine.optimizer is not None:
                        pgs = getattr(engine.optimizer, 'param_groups', None)
                        if pgs:
                            writer.add_scalar('train/lr', float(pgs[0].get('lr', 0.0)), global_step)
                except Exception:
                    pass
            if engine.global_rank == 0 and (global_step % log_interval == 0):
                if tqdm is None:
                    print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")
                else:
                    iterable.set_postfix({"loss": f"{loss.item():.6f}"})
            if engine.global_rank == 0 and (global_step % val_interval == 0):
                eval_stat = evaluate(engine, val_loader)
                if writer is not None:
                    writer.add_scalar('val/loss', float(eval_stat), global_step)
                print(f"[eval] step={global_step} val_loss={eval_stat:.6f}")
                engine.save_checkpoint(str(ckpt_dir))

    if engine.global_rank == 0:
        eval_stat = evaluate(engine, val_loader)
        if writer is not None:
            writer.add_scalar('val/final_loss', float(eval_stat), global_step)
            writer.flush()
            writer.close()
        print(f"[final eval] val_loss={eval_stat:.6f}")
        engine.save_checkpoint(str(ckpt_dir))


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
    parser.add_argument('--config', type=str, default='configs/train_default.yaml')
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='configs/ds_config.json')
    parser.add_argument('--train_config', type=str, default='')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1, help='for deepspeed')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
