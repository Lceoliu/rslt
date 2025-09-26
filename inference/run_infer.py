from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    import deepspeed  # type: ignore
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint  # type: ignore
except Exception as exc:
    raise RuntimeError("DeepSpeed not installed. Install via: pip install deepspeed") from exc

from model.config import load_config
from training.train_deepspeed import VLLMTrainer, cast_model, get_cast_type
from training.utils import set_seed


def _build_test_loader(cfg: Dict[str, Any], split: str = "test") -> Any:
    from dataset.my_dataset import create_dataloader
    from dataset.transform import NormalizeProcessor

    data_cfg = cfg.get("dataset", {})
    assert data_cfg and data_cfg.get("data_dir"), "dataset.data_dir must be provided in config."
    data_cfg['min_reserved_ratio'] = 1.0
    conf_threshold = cfg.get("data", {}).get("conf_threshold", 0.1)
    transform = NormalizeProcessor(conf_threshold=conf_threshold)
    loader = create_dataloader(
        data_cfg,
        split=split,
        transform=transform,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=True,
        verbose=True,
    )
    return loader


@torch.no_grad()
def predict_for_sample(
    module: VLLMTrainer,
    batch: Dict[str, Any],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
) -> Dict[str, Any]:
    device = next(module.parameters()).device
    pose = batch["pose"].to(device)
    pose_len = batch.get("pose_len")
    if pose_len is not None:
        pose_len = pose_len.to(device)
    adjacency = {k: v.to(device) for k, v in batch["adjacency_matrix"].items()}
    part_lens = batch["part_lens"]

    tokens, token_mask, _ = module.visual(
        pose,
        part_lens=part_lens,
        pose_len=pose_len,
        adjacency=adjacency,
    )
    predictions = module.llm.generate(
        tokens,
        token_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
    )
    gt_text = batch.get("text", [""])[0] if batch.get("text") else ""
    return {
        "prediction": predictions[0],
        "ground_truth": gt_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(args.seed))

    ds_config_path = Path(cfg.get("train", {}).get("deepspeed_config", cfg.get("deepspeed_config", "configs/ds_config.json")))
    if not ds_config_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {ds_config_path}")
    ds_config = json.loads(ds_config_path.read_text(encoding="utf-8"))
    ds_config["train_batch_size"] = 1
    ds_config["gradient_accumulation_steps"] = 1

    dec_cfg = cfg.get("decoding", {})
    temperature = float(args.temperature) if args.temperature is not None else float(dec_cfg.get("temperature", 1.0))
    top_k = int(args.top_k) if args.top_k is not None else int(dec_cfg.get("top_k", 0))
    max_new_tokens = int(args.max_new_tokens) if args.max_new_tokens is not None else int(dec_cfg.get("max_new_tokens", 48))
    do_sample = bool(args.do_sample or dec_cfg.get("do_sample", False))

    net = VLLMTrainer(cfg, verbose=True)
    net = cast_model(net, get_cast_type(ds_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    ckpt_dir = Path(args.checkpoint)
    try:
        load_state_dict_from_zero_checkpoint(net, str(ckpt_dir))
        print(f"Loaded ZeRO checkpoint from {ckpt_dir}")
    except Exception as exc:
        print(f"[warn] zero_to_fp32 merge failed: {exc}. Trying raw state dict...")
        cand = None
        for name in ["pytorch_model.bin", "model_fp32.pt", "mp_rank_00_model_states.pt"]:
            candidate = ckpt_dir / name
            if candidate.exists():
                cand = candidate
                break
        if cand is None:
            raise RuntimeError(f"No checkpoint file found in {ckpt_dir}")
        state = torch.load(cand, map_location="cpu")
        if isinstance(state, dict) and "module" in state:
            state = state["module"]
        net.load_state_dict(state, strict=False)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_test_loader(cfg, split=args.split)
    results: Dict[str, Any] = {}
    for idx, batch in enumerate(loader):
        result = predict_for_sample(
            net,
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        key = f"{args.split}_batch{idx}"
        results[key] = result
        (out_dir / "partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "full.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved inference results to {out_dir}")


if __name__ == "__main__":
    main()
