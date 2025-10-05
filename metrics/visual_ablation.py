"""Visual ablation test to check if LLM uses visual information."""

import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader


@torch.no_grad()
def test_visual_importance(
    model: torch.nn.Module,
    data_loader: DataLoader,
    max_samples: int = 100,
) -> Dict[str, float]:
    """
    Test whether LLM actually uses visual input.

    Compares LLM loss with:
    1. Normal visual features
    2. Random visual features (same distribution)
    3. Zero visual features

    If losses are similar, LLM is not using visual information!

    Args:
        model: VLLMTrainer model
        data_loader: Validation data loader
        max_samples: Maximum number of batches to test

    Returns:
        Dictionary with losses for each condition
    """
    model.eval()
    device = next(model.parameters()).device

    losses = {
        'normal': [],
        'random': [],
        'zero': [],
    }

    batch_count = 0

    for batch in data_loader:
        if batch_count >= max_samples:
            break

        # Move batch to device
        pose = batch['pose'].to(device)
        part_lens = batch['part_lens']
        adjacency = {k: v.to(device) for k, v in batch['adjacency_matrix'].items()}
        pose_len = batch.get('pose_len')
        if pose_len is not None:
            pose_len = pose_len.to(device)
        last_chunk_valid_len = batch.get('last_chunk_valid_len')
        if last_chunk_valid_len is not None:
            last_chunk_valid_len = last_chunk_valid_len.to(device)
        texts = batch['text']

        # Extract visual tokens
        tokens, token_mask, _ = model.visual(
            pose,
            part_lens=part_lens,
            pose_len=pose_len,
            last_chunk_valid_len=last_chunk_valid_len,
            adjacency=adjacency,
        )
        # tokens: [B, N, P, E]

        # Test 1: Normal visual
        output, labels = model.llm(tokens, token_mask, texts)
        if hasattr(output, 'loss') and output.loss is not None:
            losses['normal'].append(output.loss.item())

        # Test 2: Random visual (same shape and distribution)
        random_tokens = torch.randn_like(tokens)
        # Match the std of original tokens
        random_tokens = random_tokens * tokens.std() + tokens.mean()
        output, labels = model.llm(random_tokens, token_mask, texts)
        if hasattr(output, 'loss') and output.loss is not None:
            losses['random'].append(output.loss.item())

        # Test 3: Zero visual
        zero_tokens = torch.zeros_like(tokens)
        output, labels = model.llm(zero_tokens, token_mask, texts)
        if hasattr(output, 'loss') and output.loss is not None:
            losses['zero'].append(output.loss.item())

        batch_count += 1

    # Compute statistics
    results = {}
    for key, values in losses.items():
        if len(values) > 0:
            results[f'loss_{key}'] = np.mean(values)
            results[f'loss_{key}_std'] = np.std(values)
        else:
            results[f'loss_{key}'] = float('nan')
            results[f'loss_{key}_std'] = float('nan')

    # Compute deltas
    if 'loss_normal' in results and 'loss_random' in results:
        results['delta_normal_random'] = results['loss_normal'] - results['loss_random']
    if 'loss_normal' in results and 'loss_zero' in results:
        results['delta_normal_zero'] = results['loss_normal'] - results['loss_zero']

    results['num_batches'] = batch_count

    return results
