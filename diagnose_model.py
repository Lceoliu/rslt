#!/usr/bin/env python
"""
Comprehensive model diagnosis script.

Runs all diagnostic metrics and generates a detailed report.

Usage:
    python diagnose_model.py --checkpoint runs/xxx/checkpoints --config configs/train_default.yaml
    python diagnose_model.py --checkpoint runs/xxx/checkpoints --config configs/train_default.yaml --output diagnosis_report.txt
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch

try:
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
except Exception as exc:
    raise RuntimeError("DeepSpeed not installed. Install via: pip install deepspeed") from exc

from model.config import load_config
from training.train_deepspeed import VLLMTrainer, cast_model, get_cast_type
from training.data import build_dataloaders

from metrics import (
    compute_retrieval_metrics,
    test_visual_importance,
    analyze_prediction_errors,
    check_data_quality,
    compute_oracle_bleu,
    analyze_special_tokens,
)


def load_model_from_checkpoint(checkpoint_dir: Path, config_path: Path):
    """Load model from checkpoint using DeepSpeed's method."""
    print(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))

    # Load DeepSpeed config
    ds_config_path = Path(cfg.get("train", {}).get("deepspeed_config", cfg.get("deepspeed_config", "configs/ds_config.json")))
    if not ds_config_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {ds_config_path}")

    ds_config = json.loads(ds_config_path.read_text(encoding="utf-8"))
    ds_config["train_batch_size"] = 1
    ds_config["gradient_accumulation_steps"] = 1

    print(f"Building model...")
    model = VLLMTrainer(cfg, verbose=False)
    model = cast_model(model, get_cast_type(ds_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Try loading ZeRO checkpoint first
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint path does not exist: {ckpt_dir}")

    try:
        load_state_dict_from_zero_checkpoint(model, str(ckpt_dir))
        print(f"✓ Loaded ZeRO checkpoint from {ckpt_dir}")
    except Exception as exc:
        print(f"[warn] ZeRO checkpoint load failed: {exc}")
        print("Trying raw state dict...")

        # Try different checkpoint file names
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
        model.load_state_dict(state, strict=False)
        print(f"✓ Loaded checkpoint from {cand}")

    return model, cfg


def print_section(title: str, log_fn=print):
    """Print a formatted section header."""
    log_fn("\n" + "=" * 80)
    log_fn(f"  {title}")
    log_fn("=" * 80)


def format_dict(d: dict, indent: int = 2) -> str:
    """Format dictionary for printing."""
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{' ' * indent}{key}:")
            lines.append(format_dict(value, indent + 2))
        elif isinstance(value, list):
            if len(value) <= 3:
                lines.append(f"{' ' * indent}{key}: {value}")
            else:
                lines.append(f"{' ' * indent}{key}: [{len(value)} items]")
        elif isinstance(value, float):
            lines.append(f"{' ' * indent}{key}: {value:.4f}")
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Diagnose model performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report file (default: print to stdout)')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Maximum samples for each metric')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    # Prepare output - collect all lines
    output_lines = []

    def log(msg: str):
        """Log to both console and output buffer."""
        print(msg)
        output_lines.append(msg)

    try:
        # Header
        log("=" * 80)
        log("MODEL DIAGNOSIS REPORT")
        log("=" * 80)
        log(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Checkpoint: {args.checkpoint}")
        log(f"Config: {args.config}")
        log(f"Device: {args.device}")
        log("=" * 80)

        # Load model
        print_section("1. Loading Model", log)
        model, cfg = load_model_from_checkpoint(
            Path(args.checkpoint),
            Path(args.config)
        )
        model = model.to(args.device)
        model.eval()
        log("✓ Model loaded successfully")

        # Build dataloaders
        print_section("2. Loading Data", log)
        train_loader, val_loader, _ = build_dataloaders(cfg)
        log(f"✓ Train samples: {len(train_loader.dataset)}")
        log(f"✓ Val samples: {len(val_loader.dataset)}")

        # === PHASE 1: Quick Checks ===
        print_section("PHASE 1: Quick Data Quality Checks", log)

        # Data quality
        log("\n[1.1] Dataset Quality Metrics")
        log("-" * 40)
        data_quality = check_data_quality(val_loader.dataset, max_samples=args.max_samples)
        log(format_dict(data_quality))

        # Oracle BLEU
        log("\n[1.2] Oracle BLEU (Self-Reference Test)")
        log("-" * 40)
        oracle_results = compute_oracle_bleu(val_loader)
        log(format_dict(oracle_results))
        if 'oracle_bleu' in oracle_results:
            if oracle_results['oracle_bleu'] < 99:
                log("⚠ WARNING: Oracle BLEU should be ~100. Data processing may have bugs!")

        # === PHASE 2: Model Behavior Tests ===
        print_section("PHASE 2: Model Behavior Analysis", log)

        # Visual ablation
        log("\n[2.1] Visual Information Usage Test")
        log("-" * 40)
        log("Testing if LLM actually uses visual input...")
        ablation_results = test_visual_importance(model, val_loader, max_samples=args.max_samples // 2)
        log(format_dict(ablation_results))

        if 'delta_normal_random' in ablation_results:
            delta = ablation_results['delta_normal_random']
            if delta < 0.5:
                log("❌ CRITICAL: LLM barely uses visual input! (Δ < 0.5)")
            elif delta < 2.0:
                log("⚠ WARNING: LLM partially uses visual input (Δ < 2.0)")
            else:
                log("✓ GOOD: LLM strongly relies on visual input (Δ >= 2.0)")

        # Special tokens
        log("\n[2.2] Special Token Analysis")
        log("-" * 40)
        token_results = analyze_special_tokens(model)
        log(format_dict(token_results))

        # === PHASE 3: Feature Quality ===
        print_section("PHASE 3: Visual Feature Quality", log)

        log("\n[3.1] Visual-Text Retrieval Accuracy")
        log("-" * 40)
        log("Testing if visual features can retrieve correct text...")
        retrieval_results = compute_retrieval_metrics(model, val_loader, max_samples=args.max_samples)
        log(format_dict(retrieval_results))

        if 'retrieval_r@1' in retrieval_results:
            r1 = retrieval_results['retrieval_r@1']
            if r1 < 10:
                log("❌ CRITICAL: Visual features have no semantic meaning (R@1 < 10%)")
            elif r1 < 30:
                log("⚠ WARNING: Visual features are weak (R@1 < 30%)")
            else:
                log("✓ GOOD: Visual features have good quality (R@1 >= 30%)")

        # === PHASE 4: Error Analysis ===
        print_section("PHASE 4: Prediction Error Analysis", log)

        log("\n[4.1] Error Pattern Classification")
        log("-" * 40)
        log("Analyzing prediction errors...")
        error_results = analyze_prediction_errors(
            model, val_loader,
            max_samples=args.max_samples,
            generation_kwargs={
                'max_new_tokens': 64,
                'do_sample': True,
                'temperature': 1.0,
                'top_k': 10,
            }
        )

        # Print statistics
        if 'total_samples' in error_results:
            total = error_results['total_samples']
            log(f"\nTotal samples analyzed: {total}\n")

            categories = ['total_fail', 'keyword_only', 'partial_correct', 'good']
            for cat in categories:
                count_key = f'{cat}_count'
                ratio_key = f'{cat}_ratio'
                if count_key in error_results:
                    count = error_results[count_key]
                    ratio = error_results[ratio_key]
                    log(f"{cat:20s}: {count:4d} ({ratio:5.1f}%)")

            # Print examples
            log("\nExample Errors:")
            for cat in categories:
                examples_key = f'{cat}_examples'
                if examples_key in error_results and error_results[examples_key]:
                    log(f"\n--- {cat.upper()} Examples ---")
                    for i, ex in enumerate(error_results[examples_key][:2], 1):
                        log(f"  Example {i}:")
                        log(f"    GT:   {ex['ground_truth']}")
                        log(f"    Pred: {ex['prediction']}")
                        log(f"    BLEU: {ex['bleu1']:.1f} / {ex['bleu4']:.1f}")

        # === SUMMARY ===
        print_section("DIAGNOSTIC SUMMARY", log)

        issues = []
        recommendations = []

        # Check for critical issues
        if 'delta_normal_random' in ablation_results and ablation_results['delta_normal_random'] < 0.5:
            issues.append("LLM is not using visual information")
            recommendations.append("Check label masking in LLM_wrapper.py")
            recommendations.append("Increase visual encoder training")

        if 'retrieval_r@1' in retrieval_results and retrieval_results['retrieval_r@1'] < 10:
            issues.append("Visual features have no semantic meaning")
            recommendations.append("Improve GCN architecture or add more temporal modeling")
            recommendations.append("Check if contrastive learning is working")

        if 'oracle_bleu' in oracle_results and oracle_results['oracle_bleu'] < 99:
            issues.append("Data processing may have bugs")
            recommendations.append("Check dataset collation and text preprocessing")

        if 'total_fail_ratio' in error_results and error_results['total_fail_ratio'] > 30:
            issues.append("High rate of complete failures")
            recommendations.append("Visual encoder is not extracting useful features")

        if 'keyword_only_ratio' in error_results and error_results['keyword_only_ratio'] > 50:
            issues.append("Model only predicts keywords, not fluent text")
            recommendations.append("LLM may need more training")
            recommendations.append("Consider using a larger LLM")

        if issues:
            log("\n⚠ CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                log(f"  {i}. {issue}")

            log("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                log(f"  {i}. {rec}")
        else:
            log("\n✓ No critical issues detected. Model appears healthy!")

        log("\n" + "=" * 80)
        log("END OF REPORT")
        log("=" * 80)

        # Write to file if specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\n✓ Report saved to: {args.output}")

    except Exception as e:
        error_msg = f"\n❌ Error during diagnosis: {e}"
        print(error_msg)
        if args.output:
            output_lines.append(error_msg)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
        raise


if __name__ == '__main__':
    main()
