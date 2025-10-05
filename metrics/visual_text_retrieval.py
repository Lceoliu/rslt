"""Visual-Text retrieval metrics to test visual encoder quality."""

import torch
import torch.nn.functional as F
from typing import Dict, Any
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_retrieval_metrics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    max_samples: int = 500,
) -> Dict[str, float]:
    """
    Test visual-to-text retrieval accuracy.

    If visual features are good, they should be able to retrieve
    the correct text based on cosine similarity.

    Args:
        model: VLLMTrainer model
        data_loader: Validation data loader
        max_samples: Maximum number of samples to evaluate

    Returns:
        Dictionary with R@1, R@5, R@10 metrics
    """
    model.eval()
    device = next(model.parameters()).device

    visual_features = []
    text_embeddings = []
    ground_truths = []

    sample_count = 0

    for batch in data_loader:
        if sample_count >= max_samples:
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

        # 1. Extract visual tokens
        tokens, token_mask, _ = model.visual(
            pose,
            part_lens=part_lens,
            pose_len=pose_len,
            last_chunk_valid_len=last_chunk_valid_len,
            adjacency=adjacency,
        )
        # tokens: [B, N, P, E], token_mask: [B, N, P]

        # Average pool over valid tokens
        valid_count = token_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
        pooled = (tokens * token_mask.unsqueeze(-1)).sum(dim=(1, 2)) / valid_count.squeeze(-1)
        visual_features.append(pooled.cpu())

        # 2. Get text embeddings (from LLM embedding layer)
        text_ids = model.llm.get_texts_ids(texts)
        text_emb = []
        for ids in text_ids:
            if len(ids) > 0:
                emb = model.llm.get_id_embeddings(ids).mean(dim=0)
            else:
                emb = torch.zeros(model.llm.hidden_size, device=device)
            text_emb.append(emb)
        text_embeddings.append(torch.stack(text_emb).cpu())
        ground_truths.extend(texts)

        sample_count += len(texts)

    if len(visual_features) == 0:
        return {
            'retrieval_r@1': 0.0,
            'retrieval_r@5': 0.0,
            'retrieval_r@10': 0.0,
            'num_samples': 0,
        }

    # Concatenate all features
    V = torch.cat(visual_features, dim=0)[:max_samples]  # [N, E]
    T = torch.cat(text_embeddings, dim=0)[:max_samples]  # [N, E]
    N = V.size(0)

    # Normalize
    V = F.normalize(V, dim=-1)
    T = F.normalize(T, dim=-1)

    # Compute similarity matrix
    sim = V @ T.t()  # [N, N]

    # Top-k retrieval
    k_max = min(10, N)
    _, top_indices = sim.topk(k_max, dim=-1)

    # Compute R@k
    correct_indices = torch.arange(N).unsqueeze(1)  # [N, 1]

    r_at_1 = (top_indices[:, :1] == correct_indices).any(dim=1).float().mean().item()
    r_at_5 = (top_indices[:, :min(5, k_max)] == correct_indices).any(dim=1).float().mean().item()
    r_at_10 = (top_indices[:, :k_max] == correct_indices).any(dim=1).float().mean().item()

    return {
        'retrieval_r@1': r_at_1 * 100,  # Convert to percentage
        'retrieval_r@5': r_at_5 * 100,
        'retrieval_r@10': r_at_10 * 100,
        'num_samples': N,
    }
