import torch
import torch.nn.functional as F


@torch.no_grad()
def log_logits(module, logits, labels, gt_text, log_path, step=None):
    if logits is None or labels is None:
        return

    sample_logits = logits[0]
    sample_labels = labels[0]

    probs = F.softmax(sample_logits, dim=-1)
    predicted_ids = torch.argmax(probs, dim=-1)

    valid_positions = torch.nonzero(sample_labels >= 0, as_tuple=False).squeeze(-1)
    if valid_positions.numel() == 0:
        return

    tokenizer = module.llm.tokenizer
    log_lines = ["\n--- Logits Visualization ---"]
    if step is not None:
        log_lines.append(f"Step: {step} | Sample: '{gt_text}'")
    else:
        log_lines.append(f"Sample: '{gt_text}'")
    log_lines.append("Pos | Prev Token | Pred Token | GT Token | GT Prob | Correct?")
    log_lines.append("--------------------------------------------------------------")

    for pos in valid_positions[:15]:
        pos = int(pos.item())
        gt_id = int(sample_labels[pos].item())
        pred_id = int(predicted_ids[pos].item())
        gt_prob = float(probs[pos, gt_id].item())

        prev_label = sample_labels[pos - 1] if pos > 0 else torch.tensor(-100)
        if prev_label >= 0:
            prev_token = tokenizer.decode([int(prev_label.item())])
        else:
            prev_token = "[PROMPT]"

        gt_token = tokenizer.decode([gt_id]) if gt_id >= 0 else "<PAD>"
        pred_token = tokenizer.decode([pred_id])
        is_correct = "OK" if pred_id == gt_id else "NG"

        log_lines.append(
            f"{pos:<3d}| {prev_token:>12} | {pred_token:>11} | {gt_token:>9} | {gt_prob:6.2%} | {is_correct}"
        )

    log_lines.append("--- End Visualization ---\n")
    log_text = "\n".join(log_lines)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_text)
