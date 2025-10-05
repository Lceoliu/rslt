"""Evaluation metrics utilities for the rslt project.

Currently includes:
- BLEU report script (bleu_report.py)
- ROUGE report script (rouge_report.py)
- Diagnostic metrics for model analysis
"""

from .visual_text_retrieval import compute_retrieval_metrics
from .visual_ablation import test_visual_importance
from .error_analysis import analyze_prediction_errors
from .data_quality import check_data_quality, compute_oracle_bleu
from .special_tokens import analyze_special_tokens

__all__ = [
    'compute_retrieval_metrics',
    'test_visual_importance',
    'analyze_prediction_errors',
    'check_data_quality',
    'compute_oracle_bleu',
    'analyze_special_tokens',
]
