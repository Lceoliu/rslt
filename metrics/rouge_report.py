from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _char_tokenize(s: str) -> List[str]:
    return list(str(s).strip())


def _ngram_counts(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    L = len(tokens)
    if n <= 0:
        return counts
    for i in range(max(L - n + 1, 0)):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def rouge_n(ref: str, hyp: str, n: int = 1) -> Tuple[float, float, float]:
    """ROUGE-N at character n-gram level.

    Returns (precision, recall, f1).
    """
    ref_toks = _char_tokenize(ref)
    hyp_toks = _char_tokenize(hyp)

    ref_counts = _ngram_counts(ref_toks, n)
    hyp_counts = _ngram_counts(hyp_toks, n)

    overlap = 0
    for ng, c in hyp_counts.items():
        overlap += min(c, ref_counts.get(ng, 0))

    ref_total = sum(ref_counts.values())
    hyp_total = sum(hyp_counts.values())

    precision = overlap / hyp_total if hyp_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = _f1(precision, recall)
    return precision, recall, f1


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Classic DP LCS length in O(len(a)*len(b))."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Use 2-row DP to reduce memory
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev  # swap
    return prev[m]


def rouge_l(ref: str, hyp: str) -> Tuple[float, float, float]:
    """ROUGE-L based on character-level LCS.

    Returns (precision, recall, f1).
    """
    ref_toks = _char_tokenize(ref)
    hyp_toks = _char_tokenize(hyp)

    lcs = _lcs_length(ref_toks, hyp_toks)
    hyp_len = len(hyp_toks)
    ref_len = len(ref_toks)

    precision = lcs / hyp_len if hyp_len > 0 else 0.0
    recall = lcs / ref_len if ref_len > 0 else 0.0
    f1 = _f1(precision, recall)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Inference results JSON (from run_infer.py)',
    )
    parser.add_argument(
        '--output', type=str, default='', help='Optional JSON output for ROUGE summary'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Output full per-sample scores (F1 only, concise)',
    )
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    per_sample: Dict[str, Dict[str, float]] = {}

    for key, item in data.items():
        ref = str(item.get('ground_truth', ''))
        pred = str(item.get('prediction', ''))

        _, _, r1_f1 = rouge_n(ref, pred, n=1)
        _, _, r2_f1 = rouge_n(ref, pred, n=2)
        _, _, rl_f1 = rouge_l(ref, pred)
        per_sample[key] = {
            'ROUGE-1': r1_f1,
            'ROUGE-2': r2_f1,
            'ROUGE-L': rl_f1,
        }

    num = len(per_sample) if per_sample else 1
    mean_r1 = sum(v['ROUGE-1'] for v in per_sample.values()) / num
    mean_r2 = sum(v['ROUGE-2'] for v in per_sample.values()) / num
    mean_rl = sum(v['ROUGE-L'] for v in per_sample.values()) / num

    summary: Dict[str, Any] = {
        'ROUGE-1': mean_r1,
        'ROUGE-2': mean_r2,
        'ROUGE-L': mean_rl,
        'num_samples': len(per_sample),
    }

    if args.full:
        summary['scores'] = per_sample

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved ROUGE summary to: {outp}")
    else:
        # If no output path, print summary to stdout
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
