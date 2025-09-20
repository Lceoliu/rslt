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
import os, sys, random, logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
sys.path.append(Path(__file__).parent.parent.as_posix())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed import ReduceOp
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
from training.utils import set_seed


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
        bot_token = llm_cfg.get('bot_token', '<BOT>')

        self.llm = LLMWithVisualPrefix(
            model_name_or_path=model_name,
            trust_remote_code=trust_remote_code,
            max_text_len=max_text_len,
            gradient_checkpointing=gradient_checkpointing,
            freeze_lm=freeze_lm,
            bot_token=bot_token,
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
        # Streaming config (default enabled)
        s_cfg = cfg.get('streaming', {})
        self.streaming_enabled = bool(s_cfg.get('enabled', True))
        self.window = int(s_cfg.get('window', 16))
        self.stride = int(s_cfg.get('stride', 8))
        self.drop_last = bool(s_cfg.get('drop_last', True))
        self.loss_reduction = str(s_cfg.get('loss_reduction', 'mean'))

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
        if self.streaming_enabled:
            # Sliding-window sequence of prefixes
            z_seq = self.embedder.encode_chunks(
                parts, pose_len=pose_len, window=self.window, stride=self.stride, drop_last=self.drop_last
            )  # [B, N, D]
            prefix_seq = self.adapter(z_seq)  # [B, N, E] (P=1)
            if texts is None:
                return 1e-6 * (z_seq ** 2).mean()
            return self.llm.forward_stream(prefix_seq, texts, reduction=self.loss_reduction)
        else:
            z = self.embedder(parts, pose_len=pose_len)  # [B, D]
            prefix = self.adapter(z)  # [B, P, E]
            if texts is None:
                return 1e-6 * (z ** 2).mean()
            return self.llm(prefix, texts)


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


def get_cast_type(ds_config: Dict[str, Any]) -> torch.dtype:
    if 'bf16' in ds_config and ds_config['bf16'].get('enabled', False):
        return torch.bfloat16
    if 'fp16' in ds_config and ds_config['fp16'].get('enabled', False):
        return torch.float16
    return torch.float32


def cast_model(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    if dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
    elif dtype == torch.float16:
        model = model.to(torch.float16)
    else:
        model = model.to(torch.float32)
    return model


def train(args):
    cfg = load_config(args.config)
    seed = int(cfg.get('train', {}).get('seed', 3407))
    set_seed(seed)

    # Load and normalize DeepSpeed config early so we can align batch sizes and scheduler steps
    ds_config_path = Path(args.deepspeed_config)
    assert ds_config_path.exists(), f"DeepSpeed config not found: {ds_config_path}"
    with open(ds_config_path, 'r', encoding='utf-8') as f:
        import json
        ds_config = json.load(f)

    # Align dataloader batch size to DS micro batch size for correctness
    micro_bs = int(ds_config.get('train_micro_batch_size_per_gpu') or ds_config.get('train_batch_size', 8))
    cfg.setdefault('data', {})
    cfg['data']['batch_size'] = micro_bs

    # Build dataloaders after alignment
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Infer total optimizer steps for scheduler, if provided
    epochs = int(cfg.get('train', {}).get('epochs', args.epochs))
    acc_steps = int(ds_config.get('gradient_accumulation_steps', 1))
    steps_per_epoch = max(1, len(train_loader) // max(1, acc_steps))
    total_num_steps = max(1, epochs * steps_per_epoch)
    if 'scheduler' in ds_config and isinstance(ds_config['scheduler'], dict):
        sch = ds_config['scheduler']
        params = sch.setdefault('params', {})
        # Sync warmup_max_lr with optimizer lr
        if 'optimizer' in ds_config and isinstance(ds_config['optimizer'], dict):
            opt_lr = ds_config['optimizer'].get('params', {}).get('lr')
            if opt_lr is not None:
                params['warmup_max_lr'] = opt_lr
        # If warmup_num_steps is a fraction (<1), convert to integer steps
        wms = params.get('warmup_num_steps', None)
        if isinstance(wms, float) and wms > 0.0 and wms < 1.0:
            params['warmup_num_steps'] = max(1, int(round(wms * total_num_steps)))
        elif isinstance(wms, int):
            # keep as is
            pass
        else:
            # default to 5% if unspecified or invalid
            params['warmup_num_steps'] = max(1, int(round(0.05 * total_num_steps)))
        params['total_num_steps'] = total_num_steps

    # Build composite trainer module (embedder + adapter + LLM)
    net = VLLMTrainer(cfg)
    print(f"Model built. Total params: {sum(p.numel() for p in net.parameters()):,}")

    net = cast_model(net, get_cast_type(ds_config))
    print(f"Model cast to {next(net.parameters()).dtype}.")

    engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args, model=net, model_parameters=net.parameters()
    )

    # Resolve run/ckpt directories (support resume) with consistent RUN_ID across ranks
    train_cfg = cfg.get('train', {})
    resume_from = train_cfg.get('resume_from', '')
    if resume_from:
        rf = Path(resume_from)
        if rf.name == 'checkpoints' and rf.is_dir():
            ckpt_dir = rf
            run_dir = rf.parent
        else:
            run_dir = rf
            ckpt_dir = run_dir / 'checkpoints'
        if engine.global_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
    else:
        base_root = Path(train_cfg.get('save_dir_root', 'runs'))
        run_id = os.environ.get('RUN_ID')
        if not run_id:
            # Rank0 creates a run id; broadcast to all
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S') if (not dist.is_initialized() or dist.get_rank() == 0) else None
            obj = [run_id]
            if dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            run_id = obj[0]  # type: ignore
        run_dir = base_root / str(run_id)
        ckpt_dir = run_dir / 'checkpoints'
        if engine.global_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

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
        # Setup simple file logger for val samples
        log_path = run_dir / 'val_samples.log'
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path, encoding='utf-8')])

    epochs = int(train_cfg.get('epochs', args.epochs))
    log_interval = int(train_cfg.get('log_interval', 10))
    val_interval = int(train_cfg.get('val_interval', 100))
    save_every = int(train_cfg.get('save_every', 0))  # 0 disables periodic save

    # Optionally resume engine state
    global_step = 0
    start_epoch = 0
    if resume_from:
        try:
            load_path, client_sd = engine.load_checkpoint(str(ckpt_dir), tag=None)
            if engine.global_rank == 0:
                print(f"Resumed from checkpoint: {load_path}")
            if isinstance(client_sd, dict):
                global_step = int(client_sd.get('global_step', 0))
                start_epoch = int(client_sd.get('epoch', 0))
        except Exception as e:
            if engine.global_rank == 0:
                print(f"[WARN] Failed to resume from {ckpt_dir}: {e}")
            global_step = 0
            start_epoch = 0

    for epoch in range(start_epoch, epochs):
        engine.train()
        iterable = train_loader
        if engine.global_rank == 0 and tqdm is not None:
            iterable = tqdm(train_loader, desc=f'train epoch {epoch}', leave=False)
        for step, batch in enumerate(iterable):
            # move tensors to correct device
            if isinstance(batch, dict):
                batch = {k: (v.to(engine.local_rank) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            elif isinstance(batch, (tuple, list)):
                batch = [b.to(engine.local_rank) if isinstance(b, torch.Tensor) else b for b in batch]
            with torch.autocast(device_type='cuda', dtype=get_cast_type(ds_config)):
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
            # Periodic checkpointing on all ranks
            if save_every > 0 and (global_step % save_every == 0):
                engine.save_checkpoint(str(ckpt_dir), client_state={'global_step': global_step, 'epoch': epoch})
            if global_step % val_interval == 0:
                eval_stat = evaluate(engine, val_loader)
                if engine.global_rank == 0:
                    if writer is not None:
                        writer.add_scalar('val/loss', float(eval_stat), global_step)
                    print(f"[eval] step={global_step} val_loss={eval_stat:.6f}")
                    # Sample a few predictions for inspection
                    try:
                        sample_and_log_predictions(engine, val_loader, cfg, global_step, writer)
                    except Exception as e:
                        print(f"[warn] sample_and_log_predictions failed: {e}")
                # Save checkpoint on all ranks to avoid deadlocks
                engine.save_checkpoint(str(ckpt_dir), client_state={'global_step': global_step, 'epoch': epoch})

    eval_stat = evaluate(engine, val_loader)
    if engine.global_rank == 0:
        if writer is not None:
            writer.add_scalar('val/final_loss', float(eval_stat), global_step)
            writer.flush()
            writer.close()
        print(f"[final eval] val_loss={eval_stat:.6f}")
    engine.save_checkpoint(str(ckpt_dir), client_state={'global_step': global_step, 'epoch': epochs})


@torch.no_grad()
def evaluate(engine, loader: DataLoader) -> float:
    engine.eval()
    local_sum = 0.0
    local_count = 0
    for batch in loader:
        # move tensors to device
        if isinstance(batch, dict):
            batch = {k: (v.to(engine.local_rank) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            batch = [b.to(engine.local_rank) if isinstance(b, torch.Tensor) else b for b in batch]
        loss = engine(batch)
        local_sum += float(loss.item())
        local_count += 1
    # Aggregate across ranks to avoid desync
    if dist.is_initialized():
        buf = torch.tensor([local_sum, local_count], device=engine.device, dtype=torch.float64)
        dist.all_reduce(buf, op=ReduceOp.SUM)
        total_sum = buf[0].item()
        total_count = max(1.0, buf[1].item())
        return float(total_sum / total_count)
    else:
        return float(local_sum / max(1, local_count))


@torch.no_grad()
def sample_and_log_predictions(engine, loader: DataLoader, cfg: Dict[str, Any], global_step: int, writer=None) -> None:
    """Randomly sample 3~5 items from val loader, run streaming inference and log results.
    Rank0 only. Keeps the engine/module state intact.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    module = engine.module  # VLLMTrainer
    assert hasattr(module, 'embedder') and hasattr(module, 'adapter') and hasattr(module, 'llm')
    window = getattr(module, 'window', 16)
    stride = getattr(module, 'stride', 8)
    drop_last = getattr(module, 'drop_last', True)
    nsamples = random.randint(3, 5)
    # Decoding parameters from config (real inference settings)
    dec_cfg = cfg.get('decoding', {})
    max_new_tokens = int(dec_cfg.get('max_new_tokens', 48))
    do_sample = bool(dec_cfg.get('do_sample', False))
    temperature = float(dec_cfg.get('temperature', 1.0))
    top_k = int(dec_cfg.get('top_k', 0))
    collected = 0
    # Randomly choose starting batch index to avoid always sampling from the beginning
    start_offset = random.randint(0, 3)
    for bidx, batch in enumerate(loader):
        if bidx < start_offset:
            continue
        # Extract candidate indices within this batch
        B = batch['pose_len'].shape[0] if isinstance(batch, dict) and 'pose_len' in batch else None
        if not B:
            continue
        # Move tensors to device
        device = engine.device
        parts = {}
        for k, v in batch['pose'].items():
            parts[k] = v.to(device)
        pose_len = batch.get('pose_len', None)
        if pose_len is not None:
            pose_len = pose_len.to(device)
        texts = batch.get('text', None)
        # Sample one or two indices from this batch
        take = min(nsamples - collected, min(2, B))
        indices = random.sample(range(B), k=take)
        for i in indices:
            # Single-item dict
            single_parts = {k: v[i:i+1] for k, v in parts.items()}
            single_len = pose_len[i:i+1] if pose_len is not None else None
            single_text = [texts[i]] if texts is not None else [""]
            # Streaming encode
            z_seq = module.embedder.encode_chunks(single_parts, single_len, window, stride, drop_last)  # [1, N, D]
            prefix_seq = module.adapter(z_seq)  # [1, N, E]
            # Decode at a few chunk milestones
            N = prefix_seq.shape[1]
            step = max(1, N // 5)
            chunk_points = list(range(0, N, step))
            if (N - 1) not in chunk_points:
                chunk_points.append(N - 1)
            # Log header
            header = f"step {global_step} val, batch_index={bidx}, sample batch id {i}, gt: \"{single_text[0]}\", total chunks: {N}"
            logging.info(header)
            tb_lines = [header]
            # Iterate and generate
            module.llm.reset_prefix_cache()
            for ci in range(N):
                module.llm.step_prefix(prefix_seq[:, ci])
                if ci in chunk_points:
                    try:
                        pred = module.llm.generate_from_prefix(
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_k=top_k,
                        )
                        line = f"  chunk {ci} predict: \"{pred[0]}\""
                        logging.info(line)
                        tb_lines.append(line)
                    except Exception as e:
                        line = f"  chunk {ci} predict: <generation failed: {e}>"
                        logging.info(line)
                        tb_lines.append(line)
            # Write to TensorBoard as a single text block
            if writer is not None:
                writer.add_text('val/sample', "\n".join(tb_lines), global_step)
            collected += 1
            if collected >= nsamples:
                break
        if collected >= nsamples:
            break


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
