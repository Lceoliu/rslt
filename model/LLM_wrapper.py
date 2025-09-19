from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class LLMWithVisualPrefix(nn.Module):
    """HuggingFace CausalLM wrapper that accepts visual prefix embeddings.

    Given prefix embeddings [B, P, E] and texts (list of strings), builds
    inputs_embeds = concat(prefix, token_embeds(text)) and computes CE loss.
    """

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        max_text_len: int = 128,
        gradient_checkpointing: bool = False,
        freeze_lm: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers not installed. Please install: pip install transformers"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        # Ensure pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        if freeze_lm:
            for p in self.model.parameters():
                p.requires_grad = False

        self.hidden_size = getattr(self.model.config, 'hidden_size', None)
        if self.hidden_size is None:
            # Some models use 'n_embd' or similar
            self.hidden_size = getattr(self.model.config, 'n_embd', None)
        if self.hidden_size is None:
            raise RuntimeError("Unable to infer LLM hidden size from model config")

        self.max_text_len = int(max_text_len)

    def forward(self, prefix_embeds: torch.Tensor, texts: List[str]) -> torch.Tensor:
        # prefix_embeds: [B, P, E]
        device = prefix_embeds.device
        tok = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            add_special_tokens=True,
        )
        input_ids = tok['input_ids'].to(device)  # [B, T]
        attn = tok['attention_mask'].to(device)  # [B, T]

        # Token embeddings
        tok_emb = self.model.get_input_embeddings()(input_ids)  # [B, T, E]
        inputs_embeds = torch.cat([prefix_embeds, tok_emb], dim=1)  # [B, P+T, E]

        # Attention mask: prefix as 1
        B, P, _ = prefix_embeds.shape
        prefix_mask = torch.ones((B, P), dtype=attn.dtype, device=device)
        attn_full = torch.cat([prefix_mask, attn], dim=1)  # [B, P+T]

        # Labels: ignore prefix (-100); ignore padding
        ignore = torch.full_like(input_ids, fill_value=-100)
        labels = input_ids.clone()
        labels = torch.where(attn.bool(), labels, ignore)
        labels_full = torch.cat([torch.full((B, P), -100, dtype=labels.dtype, device=device), labels], dim=1)

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_full, labels=labels_full)
        return out.loss

