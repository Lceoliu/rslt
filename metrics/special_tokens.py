"""Analysis of special token embeddings."""

import torch
import torch.nn.functional as F
from typing import Dict
import numpy as np


@torch.no_grad()
def analyze_special_tokens(model: torch.nn.Module) -> Dict[str, any]:
    """
    Analyze learned special token embeddings.

    Checks if special tokens (<BOC>, <EOC>, <BOT>, <EOT>) have
    learned meaningful and distinct representations.

    Args:
        model: VLLMTrainer model

    Returns:
        Dictionary with similarity matrix and statistics
    """
    if not hasattr(model.llm, "_special_ids"):
        return {
            'error': 'Model does not expose special tokens; prompt-based prefix is used instead.',
        }
    # Get embedding layer
    embedding_layer = model.llm.model.get_input_embeddings()

    # Get special token IDs
    special_ids = model.llm._special_ids
    token_names = ['boc', 'eoc', 'bot', 'eot']

    # Check if all tokens exist
    missing_tokens = [name for name in token_names if name not in special_ids]
    if missing_tokens:
        return {
            'error': f'Missing special tokens: {missing_tokens}',
            'available_tokens': list(special_ids.keys()),
        }

    # Extract embeddings
    embeddings = []
    for name in token_names:
        token_id = special_ids[name]
        emb = embedding_layer.weight[token_id]
        embeddings.append(emb.cpu())

    # Stack into matrix
    emb_matrix = torch.stack(embeddings)  # [4, E]

    # Normalize
    emb_norm = F.normalize(emb_matrix, dim=-1)

    # Compute cosine similarity matrix
    sim_matrix = emb_norm @ emb_norm.t()  # [4, 4]

    # Convert to numpy for easier handling
    sim_np = sim_matrix.numpy()

    # Build results
    results = {
        'token_names': token_names,
        'similarity_matrix': sim_np.tolist(),
    }

    # Compute statistics
    # Get off-diagonal elements (pairwise similarities)
    off_diag_mask = ~np.eye(4, dtype=bool)
    off_diag_sims = sim_np[off_diag_mask]

    results['avg_pairwise_similarity'] = float(np.mean(off_diag_sims))
    results['min_pairwise_similarity'] = float(np.min(off_diag_sims))
    results['max_pairwise_similarity'] = float(np.max(off_diag_sims))

    # Check specific pairs
    boc_idx, eoc_idx, bot_idx, eot_idx = 0, 1, 2, 3

    results['boc_eoc_similarity'] = float(sim_np[boc_idx, eoc_idx])
    results['bot_eot_similarity'] = float(sim_np[bot_idx, eot_idx])
    results['chunk_text_similarity'] = float(np.mean([
        sim_np[boc_idx, bot_idx],
        sim_np[boc_idx, eot_idx],
        sim_np[eoc_idx, bot_idx],
        sim_np[eoc_idx, eot_idx],
    ]))

    # Interpretation
    # Good: BOC≈EOC, BOT≈EOT, but chunk tokens != text tokens
    chunk_pair_sim = (results['boc_eoc_similarity'] + results['bot_eot_similarity']) / 2
    cross_sim = results['chunk_text_similarity']

    if chunk_pair_sim > 0.9 and cross_sim > 0.9:
        results['interpretation'] = 'Poor: All tokens are too similar (>0.9). Not learning distinctions.'
    elif chunk_pair_sim > 0.7 and cross_sim < 0.6:
        results['interpretation'] = 'Good: Chunk boundaries (BOC/EOC) and text boundaries (BOT/EOT) are distinct.'
    elif chunk_pair_sim < 0.5:
        results['interpretation'] = 'Warning: Even within-pair similarities are low. May need more training.'
    else:
        results['interpretation'] = 'Moderate: Some structure learned but not very distinct.'

    return results
