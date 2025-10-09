import argparse
import json
from pathlib import Path
import pdb
from typing import Any, Dict, List
import torch.nn.functional as F
from tqdm import tqdm

import torch
import numpy as np

try:
    import deepspeed  # type: ignore
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint  # type: ignore
except Exception as exc:
    raise RuntimeError("DeepSpeed not installed. Install via: pip install deepspeed") from exc

from model.config import load_config
from training.train_deepspeed import VLLMTrainer, cast_model, get_cast_type
from training.utils import set_seed, compute_cosine_similarity, plot_similarity


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
        verbose=False,
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
    gt_text: str = None,
    save_path: str = None,
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
    if save_path is not None:
        S_all, flat_to_cti, Xn = compute_cosine_similarity(tokens, token_mask)
        plot_similarity(S_all, token_mask, save_path)

    loss = 0.0
    log_text = None
    if gt_text is not None:
        output, labels = module.llm(
            tokens,
            token_mask,
            [gt_text],
        )
        logits = output.logits
        if logits is not None and labels is not None:
            # pdb.set_trace()
            # Get the first sample in the batch
            sample_logits = logits[0]
            sample_labels = labels[0]

            # Get probabilities
            probs = F.softmax(sample_logits, dim=-1)

            # Get predicted token IDs
            predicted_ids = torch.argmax(probs, dim=-1)

            # Find where the actual text labels start (ignore -100)
            text_start_idx = (sample_labels == module.llm._special_ids['bot']).nonzero(
                as_tuple=True
            )[0]
            if text_start_idx.numel() > 0:
                start = text_start_idx[0]

                # Get tokenizer from the model
                tokenizer = module.llm.tokenizer

                log_lines = ["\n--- Logits Visualization (Corrected) ---"]
                log_lines.append(f"Sample: '{batch['text'][0]}'")
                log_lines.append(
                    "Pos(i)| Input Token(i) | Pred Token(i+1)| GT Token(i+1)  |  GT Prob  | Correct?"
                )
                log_lines.append(
                    "------------------------------------------------------------------------------------"
                )

                # Visualize up to 15 tokens
                for i in range(start, min(start + 15, len(sample_labels) - 1)):
                    # The model at position 'i' predicts the token for position 'i+1'
                    input_id = sample_labels[i].item()
                    gt_id = sample_labels[i + 1].item()

                    # Skip prefix padding
                    if input_id == -100:
                        continue

                    pred_id = predicted_ids[i].item()
                    gt_prob = probs[i, gt_id].item()

                    input_token = tokenizer.decode([input_id])
                    gt_token = tokenizer.decode([gt_id])
                    pred_token = tokenizer.decode([pred_id])

                    is_correct = "✅" if pred_id == gt_id else "❌"

                    log_lines.append(
                        f"{i:<6} | {input_token:>14} | {pred_token:>15} | {gt_token:>14} | {gt_prob:^9.2%} | {is_correct}"
                    )

                log_lines.append("--- End Visualization ---\n")
                log_text = "\n".join(log_lines)

        if hasattr(output, 'loss'):
            loss = output.loss
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
        "loss": float(loss) if loss else None,
        "logits_visualization": log_text if gt_text is not None else None,
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
    parser.add_argument(
        "--save_similarity",
        action="store_true",
        help="Whether to save similarity plots",
    )
    parser.add_argument(
        "--sim_count", type=int, default=15, help="Number of similarity plots to save"
    )
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

    net = VLLMTrainer(cfg, verbose=False)
    net = cast_model(net, get_cast_type(ds_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    ckpt_dir = Path(args.checkpoint)

    # Try to find checkpoint in subdirectories (e.g., global_step34332/)
    if not any(ckpt_dir.glob("*.pt")) and not any(ckpt_dir.glob("*.bin")):
        # Look for subdirectories with global_step pattern
        subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("global_step")]
        if subdirs:
            # Use the latest checkpoint
            ckpt_dir = max(subdirs, key=lambda x: int(x.name.replace("global_step", "")))
            print(f"Found checkpoint in subdirectory: {ckpt_dir}")

    # Try loading model_states.pt directly (single rank DeepSpeed checkpoint)
    model_states_file = ckpt_dir / "mp_rank_00_model_states.pt"
    if model_states_file.exists():
        print(f"Loading model states from {model_states_file}")
        state = torch.load(model_states_file, map_location="cpu")

        # DeepSpeed saves with 'module' key
        if isinstance(state, dict):
            if "module" in state:
                state = state["module"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        net.load_state_dict(state, strict=False)
        print(f"✓ Loaded checkpoint from {model_states_file}")
    else:
        # Fallback: try ZeRO checkpoint merge or other formats
        try:
            load_state_dict_from_zero_checkpoint(net, str(ckpt_dir))
            print(f"✓ Loaded ZeRO checkpoint from {ckpt_dir}")
        except Exception as exc:
            print(f"[warn] ZeRO checkpoint load failed: {exc}. Trying other formats...")
            cand = None
            for name in ["pytorch_model.bin", "model_fp32.pt", "model.pt"]:
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
            print(f"✓ Loaded checkpoint from {cand}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_test_loader(cfg, split=args.split)
    results: Dict[str, Any] = {}
    randomly_selected_indices = set()
    if args.save_similarity:
        randomly_selected_indices = set(
            np.random.choice(len(loader), size=args.sim_count, replace=False)
        )
        print(
            f"Randomly selected indices for similarity plots: {sorted(randomly_selected_indices)}"
        )
    for idx, batch in enumerate(tqdm(loader, desc="Inference")):
        gt_text = batch.get("text", [None])[0]
        if args.save_similarity and idx in randomly_selected_indices:
            save_path = out_dir / f"similarity_{idx:04d}.png"
        else:
            save_path = None
        result = predict_for_sample(
            net,
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            gt_text=gt_text,
            save_path=save_path,
        )
        key = f"{args.split}_batch{idx}"
        if result.get("logits_visualization") is not None:
            with open(out_dir / "partial.txt", "a", encoding="utf-8") as f:
                f.write(f"=== Sample {key} ===\n")
                f.write(result["logits_visualization"])
        result.pop("logits_visualization", None)
        results[key] = result
        (out_dir / "partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "full.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved inference results to {out_dir}")


if __name__ == "__main__":
    main()
