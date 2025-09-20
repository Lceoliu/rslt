from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _char_tokenize(s: str) -> List[str]:
    return list(s.strip())


def _ngram_counts(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    L = len(tokens)
    for i in range(L - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def bleu_score(ref: str, hyp: str, max_n: int = 4, smooth: float = 1.0) -> float:
    ref_toks = _char_tokenize(ref)
    hyp_toks = _char_tokenize(hyp)
    if len(hyp_toks) == 0:
        return 0.0
    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_counts = _ngram_counts(ref_toks, n)
        hyp_counts = _ngram_counts(hyp_toks, n)
        match = 0
        total = 0
        for ng, c in hyp_counts.items():
            match += min(c, ref_counts.get(ng, 0))
            total += c
        # add-one smoothing
        p = (match + smooth) / (total + smooth)
        precisions.append(p)
    import math

    # geometric mean
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    # brevity penalty
    ref_len = len(ref_toks)
    hyp_len = len(hyp_toks)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(ref_len) / float(hyp_len))
    return bp * geo_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Inference results JSON (from run_infer.py)')
    parser.add_argument('--output', type=str, default='', help='Optional JSON output for BLEU summary')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    # Collect per milestone BLEU
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}

    for key, item in data.items():
        ref = item.get('gt', '')
        preds = item.get('predict', [])
        ms = item.get('milestones', list(range(len(preds))))
        for i, pred in enumerate(preds):
            m = int(ms[i]) if i < len(ms) else i
            b = bleu_score(ref, pred, max_n=4, smooth=1.0)
            sums[m] = sums.get(m, 0.0) + b
            counts[m] = counts.get(m, 0) + 1

    # Map milestones to human-friendly ratios roughly
    # We'll sort milestones and output ratio ~ (idx+1)/N is not known globally, so leave as milestone index
    summary = []
    for m in sorted(sums.keys()):
        avg = sums[m] / max(1, counts[m])
        summary.append({'milestone_chunk_index': m, 'bleu': avg, 'count': counts[m]})

    print("BLEU summary (by milestone index):")
    for row in summary:
        print(row)

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary}, f, ensure_ascii=False, indent=2)
        print(f"Saved BLEU summary to: {outp}")


if __name__ == '__main__':
    main()

