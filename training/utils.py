import torch
import numpy as np
import random, os

import torch.nn.functional as F


@torch.no_grad()
def log_logits(module, logits, labels, gt_text, log_path, step=None):
    if logits is None or labels is None:
        return
    try:
        # Get the first sample in the batch
        sample_logits = logits[0]
        sample_labels = labels[0]

        probs = F.softmax(sample_logits, dim=-1)
        predicted_ids = torch.argmax(probs, dim=-1)

        # Find BOT token position safely
        bot_mask = sample_labels == module.llm._special_ids['bot']
        bot_positions = bot_mask.nonzero(as_tuple=True)[0]

        if bot_positions.numel() > 0:
            start = int(bot_positions[0].item())

            # Get tokenizer from the model
            tokenizer = module.llm.tokenizer

            log_lines = ["\n--- Logits Visualization (Corrected) ---"]
            log_lines.append(f"Step: {step} | Sample: '{gt_text}'")
            log_lines.append(
                "Pos(i)| Input Token(i) | Pred Token(i+1)| GT Token(i+1)  |  GT Prob  | Correct?"
            )
            log_lines.append(
                "------------------------------------------------------------------------------------"
            )

            # Visualize up to 15 tokens
            for i in range(start, min(start + 15, len(sample_labels) - 1)):
                # The model at position 'i' predicts the token for position 'i+1'
                input_id = sample_labels[i].item()
                gt_id = sample_labels[i + 1].item()

                # Skip prefix padding
                if input_id == -100:
                    continue
                if gt_id < 0:
                    continue

                pred_id = predicted_ids[i].item()
                gt_prob = probs[i, gt_id].item()

                input_token = tokenizer.decode([input_id])
                gt_token = tokenizer.decode([gt_id])
                pred_token = tokenizer.decode([pred_id])

                is_correct = 'OK' if pred_id == gt_id else 'NG'

                log_lines.append(
                    f"{i:<6} | {input_token:>14} | {pred_token:>15} | {gt_token:>14} | {gt_prob:^9.2%} | {is_correct}"
                )

            log_lines.append("--- End Visualization ---\n")
            log_text = "\n".join(log_lines)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_text)
    except Exception as e:
        print(f"[ERROR in log_logits] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise e


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def compute_cosine_similarity(
    tokens: torch.Tensor,  # [N_chunks, T, D]
    token_mask: torch.Tensor,  # [N_chunks, T]  (True/1 = valid)
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
    """
    返回:
      S_all: [T_valid, T_valid]  有效token的余弦相似度
      flat_to_cti: list[(c, t)]  全局下标 -> (chunk_idx, token_idx)
      Xn: [T_valid, D]           归一化后的有效token向量
    """
    assert tokens.dim() == 3, "tokens should be [N_chunks, tokens_per_chunk, token_dim]"
    assert token_mask.shape[:2] == tokens.shape[:2], "mask shape mismatch"

    N, T, D = tokens.shape
    # 归一化 token
    X = tokens  # [N, T, D]
    X = X / (X.norm(dim=-1, keepdim=True) + eps)

    # 只取有效 token
    valid = token_mask.bool()
    Xn = X[valid]  # [T_valid, D]

    # 余弦相似度（归一化后内积即为cosine）
    S_all = Xn @ Xn.t()  # [T_valid, T_valid]

    # 全局下标映射：将每个有效位置对应回 (chunk_idx, token_idx)
    flat_to_cti: List[Tuple[int, int]] = []
    valid_np = valid.cpu().numpy()
    for c in range(N):
        for t in range(T):
            if valid_np[c, t]:
                flat_to_cti.append((c, t))

    return S_all, flat_to_cti, Xn


def _chunk_boundaries_from_mask(token_mask: torch.Tensor) -> List[int]:
    """
    计算在flatten有效token后的分块边界（累积长度）。
    用于在热图上画出chunk分隔线。
    """
    N, T = token_mask.shape
    per_chunk_counts = token_mask.bool().sum(dim=1).tolist()
    boundaries = []
    acc = 0
    for cnt in per_chunk_counts:
        acc += int(cnt)
        boundaries.append(acc)
    return boundaries  # 长度N，每个值是该chunk结束的全局位置(不含)


def plot_similarity(
    S_all: torch.Tensor,
    token_mask: torch.Tensor,
    save_path: str,
    figsize=(8, 8),
    title: str = "Cosine similarity across chunks/tokens",
    vmin: float = -1.0,
    vmax: float = 1.0,
    dpi: int = 200,
):
    """
    保存余弦相似度热力图到指定路径。

    参数：
      S_all: [T_valid, T_valid]  余弦相似度矩阵
      token_mask: [N_chunks, tokens_per_chunk]  有效mask
      save_path: str 保存路径 (自动创建目录)
    """
    S = S_all.detach().cpu().float()

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 绘图
    plt.figure(figsize=figsize)
    im = plt.imshow(S, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(title)
    plt.xlabel("token (flattened, valid only)")
    plt.ylabel("token (flattened, valid only)")

    # 画chunk边界
    N, T = token_mask.shape
    counts = token_mask.bool().sum(dim=1).tolist()
    acc = 0
    for c, n_valid in enumerate(counts[:-1]):
        acc += n_valid
        plt.axvline(x=acc - 0.5, linewidth=1, linestyle="--", color="gray")
        plt.axhline(y=acc - 0.5, linewidth=1, linestyle="--", color="gray")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # 保存
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def topk_neighbors(
    Xn: torch.Tensor,  # [T_valid, D]  归一化后的有效token向量
    k: int = 5,
    exclude_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回每个token的top-k近邻索引与相似度，基于cosine（Xn已归一化）。
    输出:
      nn_idx: [T_valid, k]
      nn_sim: [T_valid, k]
    """
    # 计算完整相似度（和 compute_cosine_similarity 中一致）
    S = Xn @ Xn.t()  # [T_valid, T_valid]

    if exclude_self:
        # 把对角置为 -inf 以避免自身被选为近邻
        diag_idx = torch.arange(S.size(0), device=S.device)
        S[diag_idx, diag_idx] = float("-inf")

    nn_sim, nn_idx = torch.topk(S, k=k, dim=-1, largest=True, sorted=True)
    return nn_idx, nn_sim
