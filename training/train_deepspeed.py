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
from training.utils import set_seed, log_logits, compute_cosine_similarity
from metrics import bleu_report, rouge_report
from training.contrastive_modules import ProjectionHead, SequenceSummarizer


class VLLMTrainer(nn.Module):
    """Couple the visual encoder with the language model for training."""

    def __init__(self, cfg: Dict[str, Any], verbose: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        llm_cfg = cfg.get('llm', {})
        self.llm_cfg = llm_cfg
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

        # --- New Modules for Hierarchical Contrastive Learning ---
        self.sequence_summarizer = SequenceSummarizer(embed_dim=self.llm.hidden_size)
        self.projection_dim = int(llm_cfg.get('contrastive_projection_dim', 256))
        self.visual_proj_head = ProjectionHead(
            self.llm.hidden_size, self.projection_dim
        )
        self.text_proj_head = ProjectionHead(self.llm.hidden_size, self.projection_dim)
        # ---------------------------------------------------------

        self.current_epoch = 0
        self.tau = llm_cfg.get('contrastive_tau', 0.07)
        self.contrastive_alpha = float(llm_cfg.get('contrastive_alpha', 0.0))
        self.diversity_alpha = float(llm_cfg.get('diversity_alpha', 0.0))
        # For visualization
        self.last_logits = None
        self.last_labels = None
        self.scalers = {}

    def get_parameter_groups(self) -> List[Dict]:
        """Separate parameters for differential learning rates."""
        train_cfg = self.cfg.get("train", {})
        visual_lr = float(train_cfg.get("visual_lr", 5e-5))
        llm_lr = float(train_cfg.get("llm_lr", 1e-6))

        # Visual parameters include the encoder, summarizer, and projection heads
        visual_params = (
            list(self.visual.parameters())
            + list(self.sequence_summarizer.parameters())
            + list(self.visual_proj_head.parameters())
            + list(self.text_proj_head.parameters())
        )

        if self.freeze_lm:
            print(
                f"LLM is frozen. Only training visual modules with lr={visual_lr}"
            )
            return [
                {
                    "params": visual_params,
                    "lr": visual_lr,
                    "initial_lr": visual_lr,
                }
            ]

        print(f"Training with differential LRs: visual_lr={visual_lr}, llm_lr={llm_lr}")
        return [
            {
                "params": visual_params,
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
        """
        Computes a decorrelation loss to penalize similarity between chunk tokens.
        This encourages the model to produce diverse, non-collapsed representations.
        """
        # tokens: [B, N, P, E], token_mask: [B, N, P]

        # Apply mask to select only valid tokens
        valid_tokens = tokens[token_mask]  # [num_valid_tokens, E]

        num_valid_tokens = valid_tokens.shape[0]

        if num_valid_tokens < 2:
            return torch.tensor(0.0, device=tokens.device, requires_grad=True)

        # Normalize tokens to compute cosine similarity
        tokens_norm = F.normalize(valid_tokens, p=2, dim=1)

        # Compute the similarity matrix
        # sim_matrix will be [num_valid_tokens, num_valid_tokens]
        sim_matrix = torch.matmul(tokens_norm, tokens_norm.t())

        # The loss is the mean of the squared off-diagonal elements.
        # We want to push the similarity towards 0, making the features orthogonal.
        # Squaring penalizes large similarities more heavily and ensures the loss
        # is non-negative. This is inspired by redundancy reduction objectives
        # in models like Barlow Twins.
        n = num_valid_tokens
        # Create a mask for the upper triangle, excluding the diagonal
        off_diag_mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()

        # Get the off-diagonal elements
        off_diag_elements = sim_matrix[off_diag_mask]

        loss = off_diag_elements.pow(2).mean()

        return loss

    def get_contrastive_loss(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """
        Computes InfoNCE loss with in-batch negatives, projection heads,
        and a sequence summarizer.
        """
        B, N, P, E = tokens.shape
        device = tokens.device
        dtype = tokens.dtype  # Get the correct dtype from the input tensor

        # --- 1. Summarize Visual and Text Sequences ---
        # a) Prepare batch of visual token sequences (variable length)
        visual_seqs = [
            tokens[i][token_mask[i]] for i in range(B)
        ]  # List of [L_v, E]

        # b) Prepare batch of text token sequences (variable length)
        text_ids_list = self.llm.get_texts_ids(texts)
        text_seqs = [
            self.llm.get_id_embeddings(ids) for ids in text_ids_list
        ]  # List of [L_t, E]

        def _summarize_batch(sequences: List[torch.Tensor]) -> torch.Tensor:
            # Helper to pad, summarize, and handle empty sequences
            non_empty_seqs = []
            non_empty_indices = []
            for i, seq in enumerate(sequences):
                if seq.numel() > 0:
                    non_empty_seqs.append(seq)
                    non_empty_indices.append(i)

            if not non_empty_seqs:
                return torch.zeros(B, E, device=device, dtype=dtype)

            # Pad sequences to the max length in the non-empty batch
            padded_seqs = nn.utils.rnn.pad_sequence(non_empty_seqs, batch_first=True)
            # Create a boolean mask (True for valid tokens)
            mask = padded_seqs.sum(dim=-1) != 0

            summaries = self.sequence_summarizer(padded_seqs, mask=mask)

            # Place summaries back into a tensor of shape [B, E]
            full_batch_summaries = torch.zeros(B, E, device=device, dtype=dtype)
            full_batch_summaries[non_empty_indices] = summaries
            return full_batch_summaries

        visual_summaries = _summarize_batch(visual_seqs)  # [B, E]
        text_summaries = _summarize_batch(text_seqs)  # [B, E]

        # --- 2. Project, Gather, and Compute Loss ---
        projected_visual = self.visual_proj_head(visual_summaries)  # [B, E_proj]
        projected_text = self.text_proj_head(text_summaries)  # [B, E_proj]

        if dist.is_initialized():
            world_size = dist.get_world_size()
            visual_list = [torch.zeros_like(projected_visual) for _ in range(world_size)]
            text_list = [torch.zeros_like(projected_text) for _ in range(world_size)]

            dist.all_gather(visual_list, projected_visual)
            dist.all_gather(text_list, projected_text)

            world_visual = torch.cat(visual_list, dim=0)
            world_text = torch.cat(text_list, dim=0)
        else:
            world_visual = projected_visual
            world_text = projected_text

        world_visual = F.normalize(world_visual, p=2, dim=-1)
        world_text = F.normalize(world_text, p=2, dim=-1)

        logits_per_visual = (world_visual @ world_text.t()) / self.tau

        labels = torch.arange(logits_per_visual.shape[0], device=device)
        loss_v2t = F.cross_entropy(logits_per_visual, labels)
        loss_t2v = F.cross_entropy(logits_per_visual.t(), labels)

        loss = (loss_v2t + loss_t2v) / 2.0
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
            else torch.tensor(0.0, device=device)
        )

        diversity_loss = (
            self.get_diverse_loss(tokens, token_mask)
            if self.diversity_alpha > 0
            else torch.tensor(0.0, device=device)
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
        self.scalers['contrastive_loss'] = contrastive_loss.item()
        self.scalers['diversity_loss'] = diversity_loss.item()

        loss = loss + self.contrastive_alpha * contrastive_loss + self.diversity_alpha * diversity_loss
        self.scalers['total_loss'] = loss.item()
        return loss

    @torch.no_grad()
    def generate(self, batch: Dict[str, Any]) -> List[str]:
        self.visual.eval()
        self.llm.eval()
        pose = batch['pose']
        part_lens = batch['part_lens']
        adjacency = batch['adjacency_matrix']
        texts = batch.get('text')
        pose_len = batch.get('pose_len')
        last_chunk_valid_len = batch.get('last_chunk_valid_len')

        device = next(self.visual.parameters()).device
        pose = pose.to(device)
        pose_len = pose_len.to(device) if pose_len is not None else None
        last_chunk_valid_len = (
            last_chunk_valid_len.to(device)
            if last_chunk_valid_len is not None
            else None
        )
        adjacency = {k: v.to(device) for k, v in adjacency.items()}

        tokens, token_mask, _ = self.visual(
            pose,
            part_lens=part_lens,
            pose_len=pose_len,
            last_chunk_valid_len=last_chunk_valid_len,
            adjacency=adjacency,
        )

        predictions = self.llm.generate(
            tokens,
            token_mask,
            max_new_tokens=self.llm_cfg.get('max_new_tokens', 64),
            do_sample=self.llm_cfg.get('do_sample', True),
            temperature=self.llm_cfg.get('temperature', 0.3),
            top_k=self.llm_cfg.get('top_k', 15),
        )

        return predictions


def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a

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

def get_sync_run_id() -> str:
    run_id = os.environ.get('RUN_ID')
    if not run_id:
        # Rank0 creates a run id; broadcast to all
        run_id = (
            datetime.now().strftime('%Y%m%d_%H%M%S')
            if (not dist.is_initialized() or dist.get_rank() == 0)
            else None
        )
        obj = [run_id]
        if dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]  # type: ignore
    return run_id

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

    local_rank = args.local_rank
    if not args.deepspeed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if dist.is_initialized():
        local_rank = dist.get_rank()

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
            pass
        else:
            # default to 5% if unspecified or invalid
            params['warmup_num_steps'] = max(1, int(round(0.05 * total_num_steps)))
        if params.get('total_num_steps') == 'auto':
            params['total_num_steps'] = total_num_steps
        elif 'total_num_steps' not in params:
            params['total_num_steps'] = total_num_steps

    net = VLLMTrainer(cfg)
    if local_rank == 0:
        print(
            f"Model built. Total params: {sum(p.numel() for p in net.parameters()):,}"
        )

    net = cast_model(net, get_cast_type(ds_config))
    if local_rank == 0:
        print(f"Model cast to {next(net.parameters()).dtype}.")

    param_groups = net.get_parameter_groups()
    target_lrs = [float(pg.get("lr", 0.0)) for pg in param_groups]

    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=net, model_parameters=param_groups, config=ds_config
    )
    _sync_param_group_lrs(engine, target_lrs)
    local_rank = engine.local_rank
    device = engine.device
    print(
        f"DeepSpeed engine initialized. Local rank: {engine.local_rank}, Global rank: {engine.global_rank}"
    )
    net.to(device)

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
        run_id = get_sync_run_id()
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
                run_dir = run_dir.parent / f"{run_dir_name}_resume_{get_sync_run_id()}"
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
        try:
            import yaml

            with open(run_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        except Exception:
            pass
        log_path = run_dir / 'val_samples.log'
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path, encoding='utf-8')])

    epochs = int(train_cfg.get('epochs', args.epochs))
    log_interval = int(train_cfg.get('log_interval', 1000))
    log_logits = bool(train_cfg.get('log_logits', True))
    val_interval = int(train_cfg.get('val_interval', 500))
    save_every = int(train_cfg.get('save_every', 0))  # 0 disables periodic save
    ckpt_tag = str(train_cfg.get('ckpt_tag', None)).strip()
    resume_for_new_stage = bool(train_cfg.get('resume_for_new_stage', False))

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

        ckpt_dir = run_dir / 'checkpoints'  # do not overwrite old checkpoints

        if not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Manual LR scheduling for differential learning rates
    def update_learning_rate(step, warmup_steps, total_steps, base_lrs):
        """Cosine annealing with linear warmup for each param group."""
        import math
        lrs = []
        for base_lr in base_lrs:
            if step < warmup_steps:
                # Linear warmup
                lr = base_lr * (step + 1) / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs

    warmup_steps = max(1, int(0.05 * total_num_steps))  # 5% warmup
    base_lrs = target_lrs  # [visual_lr, llm_lr]

    min_val_loss = float('inf')

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
            # Update learning rates manually
            current_lrs = update_learning_rate(global_step, warmup_steps, total_num_steps, base_lrs)
            if hasattr(engine, 'optimizer') and engine.optimizer is not None:
                for i, pg in enumerate(engine.optimizer.param_groups):
                    if i < len(current_lrs):
                        pg['lr'] = current_lrs[i]

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
                engine.save_checkpoint(
                    str(ckpt_dir),
                    client_state={'global_step': global_step, 'epoch': epoch},
                )
            if global_step % val_interval == 0 and global_step > 0:
                eval_stat, score_dict = evaluate(engine, val_loader)
                val_loss = float(eval_stat)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    # Save best checkpoint
                    engine.save_checkpoint(
                        str(ckpt_dir),
                        tag='best',
                        client_state={'global_step': global_step, 'epoch': epoch},
                    )
                    print(
                        f"  [info] New best model saved at step {global_step} with val_loss={val_loss:.6f}"
                    )
                if engine.global_rank == 0:
                    if writer is not None:
                        writer.add_scalar('val/loss', float(eval_stat), global_step)
                        for metric, score in score_dict.items():
                            writer.add_scalar(
                                f'val/{metric}', float(score), global_step
                            )
                    print(f"[eval] step={global_step} val_loss={eval_stat:.6f}")

                    # Sample a few predictions for inspection
                    try:
                        sample_and_log_predictions(engine, val_loader, cfg, global_step, writer)
                    except Exception as e:
                        print(f"[warn] sample_and_log_predictions failed: {e}")

    eval_stat, score_dict = evaluate(engine, val_loader)
    if engine.global_rank == 0:
        if writer is not None:
            writer.add_scalar('val/final_loss', float(eval_stat), global_step)
            writer.flush()
            writer.close()
        print(f"[final eval] val_loss={eval_stat:.6f}")
    engine.save_checkpoint(str(ckpt_dir), client_state={'global_step': global_step, 'epoch': epochs})


@torch.no_grad()
def evaluate(engine, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
    engine.eval()
    local_sum = 0.0
    local_count = 0
    local_bleu1 = 0.0
    local_rouge1 = 0.0
    local_bleu4 = 0.0
    local_rougeL = 0.0
    for batch in loader:
        # move tensors to device
        if isinstance(batch, dict):
            batch = {k: (v.to(engine.local_rank) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            batch = [b.to(engine.local_rank) if isinstance(b, torch.Tensor) else b for b in batch]
        loss = engine(batch)
        predictions = engine.module.generate(batch)
        gt_texts = batch.get('text', [''] * len(predictions))
        # Compute BLEU and ROUGE for logging
        b1s = []
        r1s = []
        b4s = []
        rls = []
        for gt, pred in zip(gt_texts, predictions):
            b1 = bleu_report.bleu_score(gt, pred, max_n=1)
            b4 = bleu_report.bleu_score(gt, pred, max_n=4)
            rl = rouge_report.rouge_l(gt, pred)
            r1 = rouge_report.rouge_n(gt, pred, n=1)
            b1s.append(b1)
            b4s.append(b4)
            r1s.append(r1)
            rls.append(rl)
        avg_b1 = sum(b1s) / len(b1s) if b1s else 0.0
        avg_b4 = sum(b4s) / len(b4s) if b4s else 0.0
        avg_r1 = sum(r1s) / len(r1s) if r1s else 0.0
        avg_rl = sum(rls) / len(rls) if rls else 0.0
        local_bleu1 += avg_b1
        local_bleu4 += avg_b4
        local_rouge1 += avg_r1
        local_rougeL += avg_rl

        local_sum += float(loss.item())
        local_count += 1
    # Aggregate across ranks to avoid desync
    if dist.is_initialized():
        buf = torch.tensor(
            [
                local_sum,
                local_count,
                local_bleu1,
                local_bleu4,
                local_rouge1,
                local_rougeL,
            ],
            device=engine.device,
            dtype=torch.float64,
        )
        dist.all_reduce(buf, op=ReduceOp.SUM)
        total_sum = buf[0].item()
        total_count = max(1.0, buf[1].item())
        total_bleu1 = buf[2].item()
        total_bleu4 = buf[3].item()
        total_rouge1 = buf[4].item()
        total_rougeL = buf[5].item()
        if dist.get_rank() == 0:
            print(
                f"  [eval] val_loss={total_sum / total_count:.6f} BLEU-1={total_bleu1 / total_count:.4f} BLEU-4={total_bleu4 / total_count:.4f} ROUGE-1={total_rouge1 / total_count:.4f} ROUGE-L={total_rougeL / total_count:.4f}"
            )
        dic = {
            'BLEU-1': total_bleu1 / total_count,
            'BLEU-4': total_bleu4 / total_count,
            'ROUGE-1': total_rouge1 / total_count,
            'ROUGE-L': total_rougeL / total_count,
        }
        return float(total_sum / total_count), dic
    else:
        dic = {
            'BLEU-1': local_bleu1 / max(1, local_count),
            'BLEU-4': local_bleu4 / max(1, local_count),
            'ROUGE-1': local_rouge1 / max(1, local_count),
            'ROUGE-L': local_rougeL / max(1, local_count),
        }
        return float(local_sum / max(1, local_count)), dic


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
        texts = batch.get('text', [''] * batch['pose'].size(0))
        res = engine.module.generate(batch)
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
