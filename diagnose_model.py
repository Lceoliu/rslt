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
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(Path(__file__).parent.as_posix())

import torch
import torch.distributed as dist

from model.config import load_config
from training.train_deepspeed import VLLMTrainer
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
    """Load model from checkpoint."""
    print(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))

    print(f"Building model...")
    model = VLLMTrainer(cfg, verbose=False)

    # Find checkpoint file
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")

    # Look for pytorch_model.bin or similar
    model_files = list(ckpt_path.glob("*.bin")) + list(ckpt_path.glob("*.pt"))
    if not model_files:
        # Try to find in subdirectories (DeepSpeed format)
        model_files = list(ckpt_path.glob("**/pytorch_model.bin"))

    if not model_files:
        raise ValueError(f"No model checkpoint found in {ckpt_path}")

    # Load the first found checkpoint
    checkpoint_file = model_files[0]
    print(f"Loading checkpoint from: {checkpoint_file}")

    state_dict = torch.load(checkpoint_file, map_location='cpu')

    # Handle different checkpoint formats
    if 'module' in state_dict:
        state_dict = state_dict['module']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    try:
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load checkpoint strictly: {e}")
        print("Attempting partial load...")
        model.load_state_dict(state_dict, strict=False)

    return model, cfg


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


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

    # Redirect output if specified
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
        sys.stdout = output_file

    try:
        # Header
        print("=" * 80)
        print("MODEL DIAGNOSIS REPORT")
        print("=" * 80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Config: {args.config}")
        print(f"Device: {args.device}")
        print("=" * 80)

        # Load model
        print_section("1. Loading Model")
        model, cfg = load_model_from_checkpoint(
            Path(args.checkpoint),
            Path(args.config)
        )
        model = model.to(args.device)
        model.eval()
        print("✓ Model loaded successfully")

        # Build dataloaders
        print_section("2. Loading Data")
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
        print(f"✓ Train samples: {len(train_loader.dataset)}")
        print(f"✓ Val samples: {len(val_loader.dataset)}")
        if test_loader:
            print(f"✓ Test samples: {len(test_loader.dataset)}")

        # === PHASE 1: Quick Checks ===
        print_section("PHASE 1: Quick Data Quality Checks")

        # Data quality
        print("\n[1.1] Dataset Quality Metrics")
        print("-" * 40)
        data_quality = check_data_quality(val_loader.dataset, max_samples=args.max_samples)
        print(format_dict(data_quality))

        # Oracle BLEU
        print("\n[1.2] Oracle BLEU (Self-Reference Test)")
        print("-" * 40)
        oracle_results = compute_oracle_bleu(val_loader)
        print(format_dict(oracle_results))
        if 'oracle_bleu' in oracle_results:
            if oracle_results['oracle_bleu'] < 99:
                print("⚠ WARNING: Oracle BLEU should be ~100. Data processing may have bugs!")

        # === PHASE 2: Model Behavior Tests ===
        print_section("PHASE 2: Model Behavior Analysis")

        # Visual ablation
        print("\n[2.1] Visual Information Usage Test")
        print("-" * 40)
        print("Testing if LLM actually uses visual input...")
        ablation_results = test_visual_importance(model, val_loader, max_samples=args.max_samples // 2)
        print(format_dict(ablation_results))

        if 'delta_normal_random' in ablation_results:
            delta = ablation_results['delta_normal_random']
            if delta < 0.5:
                print("❌ CRITICAL: LLM barely uses visual input! (Δ < 0.5)")
            elif delta < 2.0:
                print("⚠ WARNING: LLM partially uses visual input (Δ < 2.0)")
            else:
                print("✓ GOOD: LLM strongly relies on visual input (Δ >= 2.0)")

        # Special tokens
        print("\n[2.2] Special Token Analysis")
        print("-" * 40)
        token_results = analyze_special_tokens(model)
        print(format_dict(token_results))

        # === PHASE 3: Feature Quality ===
        print_section("PHASE 3: Visual Feature Quality")

        print("\n[3.1] Visual-Text Retrieval Accuracy")
        print("-" * 40)
        print("Testing if visual features can retrieve correct text...")
        retrieval_results = compute_retrieval_metrics(model, val_loader, max_samples=args.max_samples)
        print(format_dict(retrieval_results))

        if 'retrieval_r@1' in retrieval_results:
            r1 = retrieval_results['retrieval_r@1']
            if r1 < 10:
                print("❌ CRITICAL: Visual features have no semantic meaning (R@1 < 10%)")
            elif r1 < 30:
                print("⚠ WARNING: Visual features are weak (R@1 < 30%)")
            else:
                print("✓ GOOD: Visual features have good quality (R@1 >= 30%)")

        # === PHASE 4: Error Analysis ===
        print_section("PHASE 4: Prediction Error Analysis")

        print("\n[4.1] Error Pattern Classification")
        print("-" * 40)
        print("Analyzing prediction errors...")
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
            print(f"\nTotal samples analyzed: {total}\n")

            categories = ['total_fail', 'keyword_only', 'partial_correct', 'good']
            for cat in categories:
                count_key = f'{cat}_count'
                ratio_key = f'{cat}_ratio'
                if count_key in error_results:
                    count = error_results[count_key]
                    ratio = error_results[ratio_key]
                    print(f"{cat:20s}: {count:4d} ({ratio:5.1f}%)")

            # Print examples
            print("\nExample Errors:")
            for cat in categories:
                examples_key = f'{cat}_examples'
                if examples_key in error_results and error_results[examples_key]:
                    print(f"\n--- {cat.upper()} Examples ---")
                    for i, ex in enumerate(error_results[examples_key][:2], 1):
                        print(f"  Example {i}:")
                        print(f"    GT:   {ex['ground_truth']}")
                        print(f"    Pred: {ex['prediction']}")
                        print(f"    BLEU: {ex['bleu1']:.1f} / {ex['bleu4']:.1f}")

        # === SUMMARY ===
        print_section("DIAGNOSTIC SUMMARY")

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
            print("\n⚠ CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("\n✓ No critical issues detected. Model appears healthy!")

        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)

    finally:
        if output_file:
            output_file.close()
            sys.stdout = sys.__stdout__
            print(f"Report saved to: {args.output}")


if __name__ == '__main__':
    main()
