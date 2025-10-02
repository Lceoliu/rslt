import torch
import numpy as np
import random

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
