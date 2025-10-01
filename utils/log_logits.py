import torch
import torch.nn.functional as F


@torch.no_grad()
def log_logits(module, logits, labels, gt_text, log_path, step=None):

    if logits is not None and labels is not None:
        # Get the first sample in the batch
        sample_logits = logits[0]
        sample_labels = labels[0]

        probs = F.softmax(sample_logits, dim=-1)
        predicted_ids = torch.argmax(probs, dim=-1)

        text_start_idx = (sample_labels == module.llm._special_ids['bot']).nonzero(
            as_tuple=True
        )[0]
        if text_start_idx.numel() > 0:
            start = text_start_idx[0]

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

                pred_id = predicted_ids[i].item()
                gt_prob = probs[i, gt_id].item()

                input_token = tokenizer.decode([input_id])
                gt_token = tokenizer.decode([gt_id])
                pred_token = tokenizer.decode([pred_id])

                is_correct = "✅" if pred_id == gt_id else "❌"

                log_lines.append(
                    f"{i:<6} | {input_token:>14} | {pred_token:>15} | {gt_token:>14} | {gt_prob:^9.2%} | {is_correct}"
                )

            log_lines.append("--- End Visualization ---\\n")
            log_text = "\n".join(log_lines)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_text)
