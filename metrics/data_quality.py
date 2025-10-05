"""Data quality checks to ensure dataset is valid."""

import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset
try:
    from sacrebleu.metrics import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


def check_data_quality(dataset: Dataset, max_samples: int = 1000) -> Dict[str, any]:
    """
    Check dataset quality metrics.

    Args:
        dataset: Training or validation dataset
        max_samples: Maximum number of samples to check

    Returns:
        Dictionary with quality metrics
    """
    texts = []
    pose_lengths = []
    chunk_counts = []

    num_samples = min(len(dataset), max_samples)

    for i in range(num_samples):
        try:
            sample = dataset[i]
            text = sample.get('text', '')
            texts.append(text)

            # Get pose info
            if 'pose' in sample:
                pose = sample['pose']
                if isinstance(pose, dict):
                    # Dict format {part: tensor}
                    for part_name, part_tensor in pose.items():
                        if hasattr(part_tensor, 'shape'):
                            # Assume shape [N, T, K, C] or similar
                            chunk_counts.append(part_tensor.shape[0])
                            break  # Only count once per sample
                else:
                    # Tensor format
                    if hasattr(pose, 'shape'):
                        chunk_counts.append(pose.shape[0])

            # Get frame count
            if 'frame_cnt' in sample:
                pose_lengths.append(sample['frame_cnt'])
            elif 'original_frame_cnt' in sample:
                pose_lengths.append(sample['original_frame_cnt'])

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Compute statistics
    results = {}

    # Text statistics
    if texts:
        unique_texts = list(set(texts))
        results['total_samples'] = num_samples
        results['unique_texts'] = len(unique_texts)
        results['unique_ratio'] = len(unique_texts) / len(texts) * 100

        # Text lengths (word count)
        text_lengths = [len(t.split()) for t in texts]
        results['avg_text_length'] = np.mean(text_lengths)
        results['min_text_length'] = np.min(text_lengths)
        results['max_text_length'] = np.max(text_lengths)
        results['std_text_length'] = np.std(text_lengths)

        # Vocabulary size
        vocab = set()
        for t in texts:
            vocab.update(t.split())
        results['vocabulary_size'] = len(vocab)

    # Pose statistics
    if pose_lengths:
        results['avg_pose_length'] = np.mean(pose_lengths)
        results['min_pose_length'] = np.min(pose_lengths)
        results['max_pose_length'] = np.max(pose_lengths)
        results['std_pose_length'] = np.std(pose_lengths)

    if chunk_counts:
        results['avg_chunk_count'] = np.mean(chunk_counts)
        results['min_chunk_count'] = np.min(chunk_counts)
        results['max_chunk_count'] = np.max(chunk_counts)
        results['std_chunk_count'] = np.std(chunk_counts)

    return results


def compute_oracle_bleu(data_loader: DataLoader) -> Dict[str, float]:
    """
    Compute oracle BLEU by using ground truth as prediction.

    Should get score close to 100. If not, there's a bug in data processing.

    Args:
        data_loader: Data loader

    Returns:
        Dictionary with BLEU scores
    """
    if not BLEU_AVAILABLE:
        return {'error': 'sacrebleu not installed'}

    bleu_scorer = BLEU(effective_order=True)

    hypotheses = []
    references = []

    for batch in data_loader:
        texts = batch.get('text', [])
        for text in texts:
            hypotheses.append(text)
            references.append([text])  # Self-reference

    if len(hypotheses) == 0:
        return {'error': 'No samples found'}

    # Compute BLEU
    bleu_result = bleu_scorer.corpus_score(hypotheses, [[ref[0] for ref in references]])

    return {
        'oracle_bleu': bleu_result.score,
        'oracle_bleu1': bleu_result.precisions[0],
        'oracle_bleu2': bleu_result.precisions[1] if len(bleu_result.precisions) > 1 else 0.0,
        'oracle_bleu3': bleu_result.precisions[2] if len(bleu_result.precisions) > 2 else 0.0,
        'oracle_bleu4': bleu_result.precisions[3] if len(bleu_result.precisions) > 3 else 0.0,
        'num_samples': len(hypotheses),
    }
