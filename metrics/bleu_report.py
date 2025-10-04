from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import math


def _char_tokenize(s: str) -> List[str]:
    return list(s.strip())


def _ngram_counts(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    L = len(tokens)
    for i in range(L - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def bleu_score(ref: str, hyp: str, max_n: int = 4, smooth_eps: float = 1e-9) -> float:
    ref_toks = _char_tokenize(ref)
    hyp_toks = _char_tokenize(hyp)

    if len(hyp_toks) == 0:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_counts = _ngram_counts(ref_toks, n)
        hyp_counts = _ngram_counts(hyp_toks, n)

        total = sum(hyp_counts.values())
        if total == 0:
            # hyp 长度不足以形成 n-gram：该阶 precision 记为 0（或极小值）
            precisions.append(0.0)
            continue

        match = 0
        for ng, c in hyp_counts.items():
            match += min(c, ref_counts.get(ng, 0))

        if match == 0:
            # 只在这一阶完全零匹配时做极小平滑，避免 log(0)
            precisions.append(smooth_eps)
        else:
            precisions.append(match / total)

    # 几何平均（等权）
    # 若出现 0，会因上面的 eps 变成很小但不致于报错
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    # Brevity Penalty
    ref_len = len(ref_toks)
    hyp_len = len(hyp_toks)
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / max(hyp_len, 1))

    return bp * geo_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Inference results JSON (from run_infer.py)')
    parser.add_argument('--output', type=str, default='', help='Optional JSON output for BLEU summary')
    parser.add_argument(
        '--full', action='store_true', help='Output full per-sample scores'
    )
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    scores = {}

    for key, item in data.items():
        ref = item.get('ground_truth', '')
        pred = item.get('prediction', [])
        b1 = bleu_score(ref, pred, max_n=1)
        b2 = bleu_score(ref, pred, max_n=2)
        b3 = bleu_score(ref, pred, max_n=3)
        b4 = bleu_score(ref, pred, max_n=4)
        scores[key] = (b1, b2, b3, b4)

    summary = {}
    mean_b1 = sum(b1 for b1, _, _, _ in scores.values()) / len(scores)
    mean_b2 = sum(b2 for _, b2, _, _ in scores.values()) / len(scores)
    mean_b3 = sum(b3 for _, _, b3, _ in scores.values()) / len(scores)
    mean_b4 = sum(b4 for _, _, _, b4 in scores.values()) / len(scores)
    summary['BLEU-1'] = mean_b1
    summary['BLEU-2'] = mean_b2
    summary['BLEU-3'] = mean_b3
    summary['BLEU-4'] = mean_b4
    summary['num_samples'] = len(scores)

    if args.full:
        summary['scores'] = {}
        for key, (b1, b2, b3, b4) in scores.items():
            summary['scores'][key] = {
                'BLEU-1': b1,
                'BLEU-2': b2,
                'BLEU-3': b3,
                'BLEU-4': b4,
            }

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved BLEU summary to: {outp}")


if __name__ == '__main__':
    main()
