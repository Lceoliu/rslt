import os

ENABLE_DEBUG = False  # Enable to get more debug info
if ENABLE_DEBUG:
    import debugpy

    if os.environ.get('RANK') == '0' or os.environ.get('LOCAL_RANK') == '0':
        debugpy.listen(('0.0.0.0', 5678))
        print(
            f"Process with RANK {os.environ.get('RANK', 'N/A')}"  # noqa: E251
            " is listening on port 5678. Waiting for debugger attach..."
        )
        debugpy.wait_for_client()
        print("Debugger attached to RANK 0.")


import argparse
import os, sys, random, logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Sequence, Optional

sys.path.append(Path(__file__).parent.parent.as_posix())

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model.embedding import build_visual_encoder
from model.LLM_wrapper import LLMWithVisualPrefix
from training.data import build_dataloaders
from training.utils import set_seed, log_logits


class VLLMTrainer(nn.Module):
    """Couple the visual encoder with the language model for training."""

    def __init__(self, cfg: Dict[str, Any], verbose: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        llm_cfg = cfg.get('llm', {})
        model_name = llm_cfg.get(
            'model_name_or_path', '../Qwen/Qwen2.5-0.5B'
        )  # default to Qwen2.5-0.5B
        print(f"Building LLM: {model_name}")
        trust_remote_code = bool(llm_cfg.get('trust_remote_code', True))
        gradient_checkpointing = bool(llm_cfg.get('gradient_checkpointing', False))
        self.freeze_lm = bool(llm_cfg.get('freeze_lm', False))
        max_text_len = int(llm_cfg.get('max_text_len', 128))
        self.llm = LLMWithVisualPrefix(
            model_name_or_path=model_name,
            trust_remote_code=trust_remote_code,
            max_text_len=max_text_len,
            gradient_checkpointing=gradient_checkpointing,
            freeze_lm=self.freeze_lm,
            boc_token=llm_cfg.get('boc_token', '<BOC>'),
            eoc_token=llm_cfg.get('eoc_token', '<EOC>'),
            bot_token=llm_cfg.get('bot_token', '<BOT>'),
            eot_token=llm_cfg.get('eot_token', '<EOT>'),
            verbose=verbose,
        )
        self.visual = build_visual_encoder(cfg, llm_dim=self.llm.hidden_size)
        print(f"LLM hidden size: {self.llm.hidden_size}")
        self.current_epoch = 0
        self.tau = llm_cfg.get('contrastive_tau', 0.07)
        self.neg_sample_k = int(llm_cfg.get('contrastive_neg_k', 64))
        self.contrastive_alpha = float(llm_cfg.get('contrastive_alpha', 0.0))
        # For visualization
        self.last_logits = None
        self.last_labels = None
        self.scalers = {}

    def get_parameter_groups(self) -> List[Dict]:
        """Separate parameters for differential learning rates."""
        train_cfg = self.cfg.get("train", {})
        visual_lr = float(train_cfg.get("visual_lr", 5e-5))
        llm_lr = float(train_cfg.get("llm_lr", 1e-6))

        if self.freeze_lm:
            print(f"LLM is frozen. Only training visual encoder with lr={visual_lr}")
            return [
                {
                    "params": self.visual.parameters(),
                    "lr": visual_lr,
                    "initial_lr": visual_lr,
                }
            ]

        print(f"Training with differential LRs: visual_lr={visual_lr}, llm_lr={llm_lr}")
        return [
            {
                "params": self.visual.parameters(),
                "lr": visual_lr,
                "initial_lr": visual_lr,
            },
            {
                "params": self.llm.parameters(),
                "lr": llm_lr,
                "initial_lr": llm_lr,
            },
        ]

    def get_diverse_loss(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encourage diversity among the chunk tokens within a sample."""
        # tokens: [B, N, P, E], token_mask: [B, N, P]
        B, N, P, E = tokens.shape
        pass

    def get_contrastive_loss(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        texts: Sequence[str],
        tau: float = 0.07,
    ) -> torch.Tensor:
        """Encourage similarity between positive pairs and dissimilarity between negative pairs."""
        # tokens: [B, N, P, E], token_mask: [B, N, P]
        B, N, P, E = tokens.shape
        pos_ids = self.llm.get_texts_ids(texts)  # list of [L_i]
        pos_embeds = [
            self.llm.get_id_embeddings(ids) for ids in pos_ids
        ]  # list of [L_i, E]
        sample_neg_k = self.neg_sample_k
        neg_embeds = [
            self.llm.sample_negative_embeddings(sample_neg_k, pos_id).detach()
            for pos_id in pos_ids
        ]  # list of [sample_neg_k, E]
        neg_embeds = torch.stack(neg_embeds, dim=0).to(
            tokens.device
        )  # [B, sample_neg_k, E]
        Lmax = max(pe.shape[0] for pe in pos_embeds)
        pos_pad = torch.zeros(B, Lmax, E, device=tokens.device)  # [B, Lmax, E]
        pos_mask = torch.zeros(
            B, Lmax, dtype=torch.bool, device=tokens.device
        )  # [B, Lmax]
        for b, pe in enumerate(pos_embeds):
            L = pe.shape[0]
            if L <= 0 or L > Lmax:
                raise ValueError("Invalid pos_embeds length.")
            pos_pad[b, :L, :] = pe
            pos_mask[b, :L] = 1

        C = torch.cat([pos_pad, neg_embeds], dim=1)  # [B, Lmax + sample_neg_k, E]
        M = C.shape[1]
        cand_mask = torch.cat(
            [
                pos_mask,
                torch.ones(B, sample_neg_k, dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )  # [B, M]

        C = F.normalize(C, dim=-1)
        Y = tokens[token_mask].reshape(-1, E)  # [sum(B*N*P_valid), E]
        Y = F.normalize(Y, dim=-1)

        with torch.no_grad():
            b_grid = (
                torch.arange(B, device=tokens.device).view(B, 1, 1).expand(B, N, P)
            )  # [B, N, P]
            b_idx = b_grid[token_mask].contiguous()  # [sum(B*N*P_valid)]

        loss_list = []
        start = 0

        for b in range(B):
            Tb = (b_idx == b).sum().item()  # number of valid tokens for sample b
            if Tb <= 0:
                continue
            Yb = Y[start : start + Tb]  # [Tb, E]
            start += Tb

            Cb = C[b]  # [M, E]
            logits = (Yb @ Cb.t()) / tau  # [Tb, M]
            mask_b = cand_mask[b].unsqueeze(0).expand(logits.shape)  # [Tb, M]
            logits = logits.masked_fill(~mask_b, float('-inf'))  # 用-inf填充padding位置
            log_probs = F.log_softmax(logits, dim=-1)  # [Tb, M]

            Lb = pos_mask[b].sum().item()  # length of positive text
            if Lb <= 0:
                continue
            pos_log = log_probs[:, :Lb]  # [Tb, Lb]
            loss_b = -pos_log.mean()  # average over all tokens and positive texts
            loss_list.append(loss_b)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=tokens.device, requires_grad=True)
        loss = torch.stack(loss_list).mean()
        return loss

    def forward(self, batch: Dict[str, Any]):
        if not isinstance(batch, dict):
            raise TypeError('Expected batch dict from dataloader.')
        pose = batch['pose']
        part_lens = batch['part_lens']
        adjacency = batch['adjacency_matrix']
        texts = batch.get('text')
        pose_len = batch.get('pose_len')
        last_chunk_valid_len = batch.get('last_chunk_valid_len')

        if texts is None:
            raise ValueError('Batch must include text field.')

        device = next(self.visual.parameters()).device
        pose = pose.to(device)
        pose_len = pose_len.to(device) if pose_len is not None else None
        last_chunk_valid_len = last_chunk_valid_len.to(device) if last_chunk_valid_len is not None else None
        adjacency = {k: v.to(device) for k, v in adjacency.items()}

        tokens, token_mask, _ = self.visual(
            pose,
            part_lens=part_lens,
            pose_len=pose_len,
            last_chunk_valid_len=last_chunk_valid_len,
            adjacency=adjacency,
        )
        # tokens: [B, N, P, E], token_mask: [B, N, P]
        contrastive_loss = (
            self.get_contrastive_loss(tokens, token_mask, texts, tau=self.tau)
            if self.contrastive_alpha > 0
            else None
        )

        # The llm.forward now returns (outputs, labels)
        output, labels = self.llm(tokens, token_mask, texts)

        # Store for visualization
        self.last_logits = output.logits
        self.last_labels = labels

        loss = output.loss if hasattr(output, 'loss') else output
        if loss is None:
            raise ValueError("LLM forward did not return loss.")
        self.scalers['LM_loss'] = loss.item()
        self.scalers['contrastive_loss'] = (
            contrastive_loss.item() if contrastive_loss is not None else 0.0
        )
        if self.contrastive_alpha > 0:
            loss = loss + self.contrastive_alpha * contrastive_loss
        self.scalers['total_loss'] = loss.item()
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



def _sync_param_group_lrs(engine, target_lrs):
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None:
        return
    groups = optimizer.param_groups
    for group, lr in zip(groups, target_lrs):
        group["lr"] = lr
        group["initial_lr"] = lr
    scheduler = getattr(engine, "lr_scheduler", None)
    if scheduler is not None:
        sched_groups = getattr(scheduler, "param_groups", None)
        if sched_groups:
            for group, lr in zip(sched_groups, target_lrs):
                group["initial_lr"] = lr
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
        # Sync warmup_max_lr with the maximum lr from train config
        train_cfg = cfg.get('train', {})
        visual_lr = float(train_cfg.get('visual_lr', 5e-5))
        llm_lr = float(train_cfg.get('llm_lr', 1e-6))
        max_lr = max(visual_lr, llm_lr)
        params['warmup_max_lr'] = max_lr
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
        if params.get('total_num_steps') == 'auto':
            params['total_num_steps'] = total_num_steps
        elif 'total_num_steps' not in params:
            params['total_num_steps'] = total_num_steps

    # Build composite trainer module (visual encoder + LLM)
    net = VLLMTrainer(cfg)
    print(f"Model built. Total params: {sum(p.numel() for p in net.parameters()):,}")

    net = cast_model(net, get_cast_type(ds_config))
    print(f"Model cast to {next(net.parameters()).dtype}.")

    # Get parameter groups for differential learning rates
    param_groups = net.get_parameter_groups()
    target_lrs = [float(pg.get("lr", 0.0)) for pg in param_groups]

    engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args, model=net, model_parameters=param_groups
    )
    _sync_param_group_lrs(engine, target_lrs)
    device = engine.device
    print(
        f"DeepSpeed engine initialized. Local rank: {engine.local_rank}, Global rank: {engine.global_rank}"
    )
    net.to(device)

    # Resolve run/ckpt directories (support resume) with consistent RUN_ID across ranks
    train_cfg = cfg.get('train', {})
    resume_from = train_cfg.get('resume_from', '')
    run_dir = None
    if resume_from:
        rf = Path(resume_from).resolve()
        if rf.name == 'checkpoints' and rf.is_dir():
            ckpt_dir = rf
            run_dir = rf.parent
        elif "global_step" in rf.name and rf.is_dir():
            ckpt_dir = rf
            run_dir = rf.parent.parent
            if not ckpt_dir.exists():
                raise ValueError(f"resume_from checkpoints dir not found: {ckpt_dir}")
        else:
            run_dir = rf
            ckpt_dir = run_dir / 'checkpoints'
            if not ckpt_dir.exists():
                raise ValueError(f"resume_from checkpoints dir not found: {ckpt_dir}")
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

    # Create directories and define log path on rank 0
    logits_log_path = None
    if engine.global_rank == 0:
        if run_dir is None:
            raise ValueError("run_dir is not set for rank 0.")
        if resume_from:
            run_dir_name = run_dir.name
            run_dir = run_dir.parent / f"{run_dir_name}_resume"
            if run_dir.exists():
                print(
                    f"Resuming run, but {run_dir} already exists. New run will be overwritten."
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = ckpt_dir.resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logits_log_path = run_dir / 'logits_visualization.log'

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
    log_logits = bool(train_cfg.get('log_logits', True))
    val_interval = int(train_cfg.get('val_interval', 100))
    save_every = int(train_cfg.get('save_every', 0))  # 0 disables periodic save
    ckpt_tag = str(train_cfg.get('ckpt_tag', None)).strip()
    resume_for_new_stage = bool(train_cfg.get('resume_for_new_stage', False))

    # Optionally resume engine state
    global_step = 0
    start_epoch = 0
    if resume_from:
        if ckpt_tag == '':
            ckpt_tag = None  # load latest
        print(f"Loading checkpoint from {ckpt_dir}, tag: {ckpt_tag or 'latest'}")
        load_path, client_sd = engine.load_checkpoint(
            str(ckpt_dir),
            tag=ckpt_tag,
            load_module_strict=not resume_for_new_stage,
            load_optimizer_states=not resume_for_new_stage,
            load_lr_scheduler_states=not resume_for_new_stage,
            load_module_only=resume_for_new_stage,
        )
        if engine.global_rank == 0:
            print(50 * '=')
            print(f"Resumed from checkpoint: {load_path}")
            print(50 * '=')
        if isinstance(client_sd, dict):
            global_step = int(client_sd.get('global_step', 0))
            start_epoch = int(client_sd.get('epoch', 0))

    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader, 'sampler') and hasattr(
            train_loader.sampler, 'set_epoch'
        ):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
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
            if hasattr(engine.module, 'current_epoch'):
                engine.module.current_epoch = global_step
            if writer is not None:
                # writer.add_scalar('train/loss', float(loss.item()), global_step)
                for k, v in engine.module.scalers.items():
                    writer.add_scalar(f'train/{k}', float(v), global_step)
                # Try to log LR
                try:
                    if hasattr(engine, 'optimizer') and engine.optimizer is not None:
                        pgs = getattr(engine.optimizer, 'param_groups', None)
                        if pgs:
                            # Log learning rates for both groups
                            for i, pg in enumerate(pgs):
                                writer.add_scalar(
                                    f'train/lr_group_{i}',
                                    float(pg.get('lr', 0.0)),
                                    global_step,
                                )
                except Exception:
                    pass
            if (
                engine.global_rank == 0
                and (global_step % log_interval == 0)
                and log_logits
            ):
                loss_val = loss.item()
                if tqdm is None:
                    print(f"epoch={epoch} step={global_step} loss={loss_val:.6f}")
                else:
                    iterable.set_postfix({"loss": f"{loss_val:.6f}"})

                # Log abnormal loss values
                if loss_val > 20 or loss_val < 0:
                    print(f"WARNING: Abnormal loss value: {loss_val}")

                # Logits visualization
                try:
                    if (
                        engine.module.last_logits is not None
                        and engine.module.last_labels is not None
                    ):
                        log_logits(
                            engine.module,
                            engine.module.last_logits,
                            engine.module.last_labels,
                            (
                                batch['text'][0]
                                if isinstance(batch, dict) and 'text' in batch
                                else ""
                            ),
                            logits_log_path,
                            step=global_step,
                        )

                except Exception as e:
                    print(f"[WARN] Failed to visualize logits: {e}")

            # Periodic checkpointing on all ranks
            if save_every > 0 and (global_step % save_every == 0) and global_step > 0:
                engine.save_checkpoint(str(ckpt_dir), client_state={'global_step': global_step, 'epoch': epoch})
            if global_step % val_interval == 0 and global_step > 0:
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
def sample_and_log_predictions(
    engine, loader: DataLoader, cfg: Dict[str, Any], global_step: int, writer=None
) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    loader_len = len(loader)
    # randomly pick 5 indices
    sample_indices = random.sample(range(loader_len), min(5, loader_len))
    samples = []
    for idx, batch in enumerate(loader):
        if idx not in sample_indices:
            continue
        # move tensors to device
        if isinstance(batch, dict):
            batch = {
                k: (v.to(engine.local_rank) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        elif isinstance(batch, (tuple, list)):
            batch = [
                b.to(engine.local_rank) if isinstance(b, torch.Tensor) else b
                for b in batch
            ]
        pose = batch['pose']
        part_lens = batch['part_lens']
        adjacency = batch['adjacency_matrix']
        texts = batch.get('text')
        pose_len = batch.get('pose_len')
        if texts is None:
            continue
        device = next(engine.module.visual.parameters()).device
        pose = pose.to(device)
        pose_len = pose_len.to(device) if pose_len is not None else None
        adjacency = {k: v.to(device) for k, v in adjacency.items()}
        tokens, token_mask, _ = engine.module.visual(
            pose,
            part_lens=part_lens,
            pose_len=pose_len,
            adjacency=adjacency,
        )
        res = engine.module.llm.generate(
            tokens,
            token_mask,
            max_new_tokens=64,
            do_sample=True,
            temperature=1.0,
            top_k=10,
        )
        for i in range(len(texts)):
            pred = res[i]
            gt = texts[i]
            samples.append((gt, pred))
    if not samples:
        return
    log_lines = [f"=== Sample predictions at step {global_step} ==="]
    for i, (gt, pred) in enumerate(samples):
        log_lines.append(f"[Sample {i}]")
        log_lines.append(f"GT: {gt}")
        log_lines.append(f"PR: {pred}")
        log_lines.append("")
    log_text = "\n".join(log_lines)
    # print(log_text)
    logging.info(log_text)
    if writer is not None:
        writer.add_text('val/samples', log_text, global_step)

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
