from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    import deepspeed  # type: ignore
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint  # type: ignore
except Exception as e:
    raise RuntimeError("DeepSpeed not installed. Install via: pip install deepspeed") from e

from model.config import load_config
from training.train_deepspeed import VLLMTrainer, cast_model, get_cast_type  # reuse trainer and helpers


def _build_test_loader(cfg: Dict[str, Any]) -> Any:
    # Prefer dataset.my_dataset with batch_size=1
    from dataset.my_dataset import create_dataloader
    from dataset.transform import NormalizeProcessor

    data_cfg = cfg.get('dataset', {})
    assert data_cfg and data_cfg.get('data_dir'), "dataset.data_dir must be provided in config to run test mode."
    transform = NormalizeProcessor()
    loader = create_dataloader(
        data_cfg,
        split='test',
        transform=transform,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get('data', {}).get('num_workers', 0)),
        pin_memory=True,
        verbose=True,
    )
    return loader


@torch.no_grad()
def _normalize_single_npy(npy_path: str) -> Tuple[Dict[str, np.ndarray], int]:
    from dataset.transform import NormalizeProcessor

    arr = np.load(npy_path)  # [T, 134, 3]
    proc = NormalizeProcessor()
    parts = proc(arr)
    T = next(iter(parts.values())).shape[0]
    return parts, T


def _milestone_indices(N: int) -> List[int]:
    # Return chunk indices for ~1/3, 1/2, 2/3, 1. Ensure valid and unique sorted.
    idxs = set()
    thirds = max(1, N // 3)
    halves = max(1, N // 2)
    two_thirds = max(1, (2 * N) // 3)
    idxs.update([thirds - 1, halves - 1, two_thirds - 1, N - 1])
    idxs = [i for i in idxs if 0 <= i < N]
    idxs.sort()
    return idxs


@torch.no_grad()
def stream_predict_for_sample(
    module: VLLMTrainer,
    parts: Dict[str, torch.Tensor] | Dict[str, np.ndarray],
    pose_len: int | torch.Tensor | None,
    text_gt: str | None,
    window: int,
    stride: int,
    drop_last: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    bf16: bool,
    timing: str = 'total',  # 'total' or 'llm'
) -> Dict[str, Any]:
    device = next(module.parameters()).device
    # Move parts to device
    parts_dev: Dict[str, torch.Tensor] = {}
    for k, v in parts.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v).to(device)
        else:
            t = v.to(device)
        parts_dev[k] = t
    pose_len_t = None
    if pose_len is not None:
        pose_len_t = torch.as_tensor(pose_len, device=device).view(1)

    # Encode once
    cast_dtype = torch.bfloat16 if bf16 else None
    autocast_kwargs = dict(enabled=bf16, device_type='cuda', dtype=torch.bfloat16)
    with torch.autocast(**autocast_kwargs):
        z_seq = module.embedder.encode_chunks(parts_dev, pose_len_t, window, stride, drop_last)  # [1,N,D]
        prefix_seq = module.adapter(z_seq)  # [1,N,E]

    N = prefix_seq.shape[1]
    idxs = _milestone_indices(N)
    preds: List[str] = []
    times_ms: List[float] = []

    for idx in idxs:
        # Reset cache; step prefixes up to idx
        module.llm.reset_prefix_cache()
        # Measure time
        if timing == 'total':
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        # Step prefix (KV-cache build)
        with torch.autocast(**autocast_kwargs):
            for ci in range(idx + 1):
                module.llm.step_prefix(prefix_seq[:, ci])
        # LLM generate
        if timing == 'llm':
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        with torch.autocast(**autocast_kwargs):
            out = module.llm.generate_from_prefix(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            )
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        preds.append(out[0] if out else "")
        times_ms.append((t1 - t0) * 1000.0)

    return {
        'gt': text_gt or "",
        'predict': preds,
        'time': times_ms,
        'num_chunks': int(N),
        'milestones': idxs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to runs/{ts}/checkpoints or runs/{ts}')
    parser.add_argument('--ds_config', type=str, default='configs/ds_config_bf16.json')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'single'])
    parser.add_argument('--npy', type=str, default='', help='Path to single .npy (T,134,3) for mode=single')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path for predictions')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--drop_last', action='store_true', default=None)
    parser.add_argument('--bf16', action='store_true', default=True)
    parser.add_argument('--timing', type=str, default='total', choices=['total', 'llm'])
    args = parser.parse_args()

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    cfg = load_config(args.config)
    # Force streaming enabled for inference
    s_cfg = cfg.setdefault('streaming', {})
    s_cfg['enabled'] = True

    # Load DS config and set micro-batch=1 for inference
    with open(args.ds_config, 'r', encoding='utf-8') as f:
        ds_config = json.load(f)
    ds_config['train_micro_batch_size_per_gpu'] = 1
    ds_config['train_batch_size'] = 1
    ds_config['gradient_accumulation_steps'] = 1

    # Override decoding from args or use config
    dec = cfg.get('decoding', {})
    temperature = float(args.temperature) if args.temperature is not None else float(dec.get('temperature', 1.0))
    top_k = int(args.top_k) if args.top_k is not None else int(dec.get('top_k', 0))
    max_new_tokens = int(args.max_new_tokens) if args.max_new_tokens is not None else int(dec.get('max_new_tokens', 48))

    # Override streaming params if provided
    window = int(args.window) if args.window is not None else int(s_cfg.get('window', 16))
    stride = int(args.stride) if args.stride is not None else int(s_cfg.get('stride', 8))
    drop_last = bool(args.drop_last) if args.drop_last is not None else bool(s_cfg.get('drop_last', True))

    # Build model and load ZeRO checkpoint by merging shards (works for different DP sizes)
    net = VLLMTrainer(cfg)
    net = cast_model(net, get_cast_type(ds_config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    ckpt_dir = Path(args.checkpoint)
    if ckpt_dir.name != 'checkpoints':
        ckpt_dir = ckpt_dir / 'checkpoints'
    assert ckpt_dir.exists(), f"Checkpoint dir not found: {ckpt_dir}"
    try:
        load_state_dict_from_zero_checkpoint(net, str(ckpt_dir))
        print(f"Loaded (merged) ZeRO checkpoint from: {ckpt_dir}")
    except Exception as e:
        # Fallback: try to find a single-state dict
        print(f"[warn] zero_to_fp32 merge failed: {e}. Trying to load a single state_dict...")
        # Common filenames
        cand = None
        for name in [
            'pytorch_model.bin',
            'model_fp32.pt',
            'mp_rank_00_model_states.pt',
        ]:
            p = ckpt_dir / name
            if p.exists():
                cand = p
                break
        if cand is None:
            raise RuntimeError(f"No valid state dict file found in {ckpt_dir}")
        sd = torch.load(cand, map_location='cpu')
        if isinstance(sd, dict) and 'module' in sd:
            sd = sd['module']
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"Loaded fallback state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    results: Dict[str, Any] = {}

    if args.mode == 'single':
        assert args.npy and Path(args.npy).exists(), "For mode=single, --npy must be provided"
        parts_np, T = _normalize_single_npy(args.npy)
        # Convert parts to torch
        parts = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in parts_np.items()}  # [1,T,V,C]
        pose_len = torch.tensor([T], device=device)
        sample = stream_predict_for_sample(
            net,
            parts,
            pose_len,
            text_gt="",
            window=window,
            stride=stride,
            drop_last=drop_last,
            max_new_tokens=max_new_tokens,
            do_sample=bool(dec.get('do_sample', False) if args.temperature is None else (args.top_k or 0) > 0 or args.temperature != 1.0),
            temperature=temperature,
            top_k=top_k,
            bf16=bool(args.bf16),
            timing=args.timing,
        )
        key = Path(args.npy).stem
        results[key] = sample
    else:
        # test mode: iterate dataset
        loader = _build_test_loader(cfg)
        for bidx, batch in enumerate(loader):
            # Each batch_size=1
            parts = {k: v.to(device) for k, v in batch['pose'].items()}
            pose_len = batch.get('pose_len', None)
            pose_len = pose_len.to(device) if pose_len is not None else None
            text_list = batch.get('text', None)
            text = text_list[0] if text_list is not None and len(text_list) > 0 else ""
            sample = stream_predict_for_sample(
                net,
                parts,
                pose_len,
                text_gt=text,
                window=window,
                stride=stride,
                drop_last=drop_last,
                max_new_tokens=max_new_tokens,
                do_sample=bool(dec.get('do_sample', False) if args.temperature is None else (args.top_k or 0) > 0 or args.temperature != 1.0),
                temperature=temperature,
                top_k=top_k,
                bf16=bool(args.bf16),
                timing=args.timing,
            )
            key = f"test_batch{bidx}_index0"
            results[key] = sample

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved inference results to: {out_path}")


if __name__ == '__main__':
    main()
