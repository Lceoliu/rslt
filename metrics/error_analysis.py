"""Error pattern analysis for generated predictions."""

import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
try:
    from sacrebleu.metrics import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


@torch.no_grad()
def analyze_prediction_errors(
    model: torch.nn.Module,
    data_loader: DataLoader,
    max_samples: int = 200,
    generation_kwargs: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Analyze error patterns in model predictions.

    Categorizes errors into:
    1. Total fail (BLEU-1 = 0) - completely wrong
    2. Keyword only (BLEU-1 > 0 but BLEU-4 = 0) - has words but no phrases
    3. Partial correct (BLEU-4 > 0 but < 20) - some phrases correct
    4. Good (BLEU-4 >= 20) - mostly correct

    Args:
        model: VLLMTrainer model
        data_loader: Validation data loader
        max_samples: Maximum number of samples to analyze
        generation_kwargs: Generation parameters

    Returns:
        Dictionary with error statistics and examples
    """
    if not BLEU_AVAILABLE:
        return {
            'error': 'sacrebleu not installed',
            'total_fail': 0,
            'keyword_only': 0,
            'partial_correct': 0,
            'good': 0,
        }

    model.eval()
    device = next(model.parameters()).device

    if generation_kwargs is None:
        generation_kwargs = {
            'max_new_tokens': 64,
            'do_sample': True,
            'temperature': 1.0,
            'top_k': 10,
        }

    categories = {
        'total_fail': [],       # BLEU-1 = 0
        'keyword_only': [],     # BLEU-1 > 0 but BLEU-4 = 0
        'partial_correct': [],  # BLEU-4 > 0 but < 20
        'good': [],             # BLEU-4 >= 20
    }

    bleu_scorer = BLEU(effective_order=True)

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

        # Extract visual tokens
        tokens, token_mask, _ = model.visual(
            pose,
            part_lens=part_lens,
            pose_len=pose_len,
            last_chunk_valid_len=last_chunk_valid_len,
            adjacency=adjacency,
        )

        # Generate predictions
        try:
            predictions = model.llm.generate(tokens, token_mask, **generation_kwargs)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue

        # Analyze each prediction
        for pred, gt in zip(predictions, texts):
            # Compute BLEU-1 and BLEU-4
            bleu1_score = bleu_scorer.sentence_score(pred, [gt], max_ngram_order=1).score
            bleu4_score = bleu_scorer.sentence_score(pred, [gt], max_ngram_order=4).score

            # Categorize
            if bleu1_score == 0:
                categories['total_fail'].append((pred, gt, bleu1_score, bleu4_score))
            elif bleu4_score == 0:
                categories['keyword_only'].append((pred, gt, bleu1_score, bleu4_score))
            elif bleu4_score < 20:
                categories['partial_correct'].append((pred, gt, bleu1_score, bleu4_score))
            else:
                categories['good'].append((pred, gt, bleu1_score, bleu4_score))

            sample_count += 1
            if sample_count >= max_samples:
                break

    # Compute statistics
    total = sum(len(v) for v in categories.values())
    results = {
        'total_samples': total,
    }

    for category, samples in categories.items():
        count = len(samples)
        ratio = count / total if total > 0 else 0.0
        results[f'{category}_count'] = count
        results[f'{category}_ratio'] = ratio * 100

        # Store up to 3 examples
        examples = []
        for i, (pred, gt, b1, b4) in enumerate(samples[:3]):
            examples.append({
                'prediction': pred,
                'ground_truth': gt,
                'bleu1': b1,
                'bleu4': b4,
            })
        results[f'{category}_examples'] = examples

    return results
