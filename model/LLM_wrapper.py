from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class LLMWithVisualPrefix(nn.Module):
    """HuggingFace CausalLM wrapper that accepts visual prefix embeddings.

    Supports two modes:
    - Non-streaming: prefix [B, P, E] or [B, E] -> concat <BOT> + text -> CE loss.
    - Streaming: prefix_seq [B, N, E] -> iteratively update KV-cache with prefix and
      compute CE on (<BOT> + text) per chunk, aggregate losses.
    """

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        max_text_len: int = 128,
        gradient_checkpointing: bool = False,
        freeze_lm: bool = False,
        bot_token: str = "<BOT>",
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

        # Add BOT token
        self.bot_token = bot_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.bot_token]})

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        # Resize embeddings after adding special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        if freeze_lm:
            for p in self.model.parameters():
                p.requires_grad = False

        self.hidden_size = getattr(self.model.config, 'hidden_size', None)
        if self.hidden_size is None:
            self.hidden_size = getattr(self.model.config, 'n_embd', None)
        if self.hidden_size is None:
            raise RuntimeError("Unable to infer LLM hidden size from model config")

        self.max_text_len = int(max_text_len)
        self.prefix_past = None  # type: ignore
        self.bot_token_id = self.tokenizer.convert_tokens_to_ids(self.bot_token)

    # -------- Non-streaming convenience --------
    def forward(self, prefix_embeds: torch.Tensor, texts: List[str]) -> torch.Tensor:
        # Accept [B, E] or [B, P, E]
        if prefix_embeds.dim() == 2:
            prefix_embeds = prefix_embeds.unsqueeze(1)
        return self._loss_once(prefix_embeds, texts)

    def _loss_once(self, prefix_embeds: torch.Tensor, texts: List[str]) -> torch.Tensor:
        device = prefix_embeds.device
        B, P, E = prefix_embeds.shape
        model_dtype = self.model.get_input_embeddings().weight.dtype
        prefix_embeds = prefix_embeds.to(model_dtype)
        # Prepare <BOT> + text
        tok = self.tokenizer(
            texts,
            return_tensors='pt', padding=True, truncation=True,
            max_length=self.max_text_len, add_special_tokens=True,
        )
        input_ids = tok['input_ids'].to(device)  # [B, T]
        attn = tok['attention_mask'].to(device)  # [B, T]

        # BOT embedding
        bot_ids = torch.full((B, 1), self.bot_token_id, dtype=input_ids.dtype, device=device)
        bot_emb = self.model.get_input_embeddings()(bot_ids)  # [B,1,E]
        bot_emb = bot_emb.to(model_dtype)
        # Text embeddings
        tok_emb = self.model.get_input_embeddings()(input_ids)  # [B, T, E]
        tok_emb = tok_emb.to(model_dtype)
        inputs_embeds = torch.cat([prefix_embeds, bot_emb, tok_emb], dim=1)  # [B, P+1+T, E]

        # Attention mask
        prefix_mask = torch.ones((B, P + 1), dtype=attn.dtype, device=device)
        attn_full = torch.cat([prefix_mask, attn], dim=1)

        # Labels: ignore prefix and BOT (-100); ignore padding in text
        ignore = torch.full_like(input_ids, fill_value=-100)
        labels_text = torch.where(attn.bool(), input_ids, ignore)
        labels_full = torch.cat([
            torch.full((B, P + 1), -100, dtype=labels_text.dtype, device=device),
            labels_text
        ], dim=1)

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_full, labels=labels_full)
        return out.loss

    # -------- Streaming KV-cache API --------
    def reset_prefix_cache(self) -> None:
        self.prefix_past = None

    def step_prefix(self, prefix_step: torch.Tensor) -> None:
        # prefix_step: [B, E] or [B, 1, E]
        if prefix_step.dim() == 2:
            prefix_step = prefix_step.unsqueeze(1)
        model_dtype = self.model.get_input_embeddings().weight.dtype
        prefix_step = prefix_step.to(model_dtype)
        B, S, E = prefix_step.shape
        device = prefix_step.device
        attn = torch.ones((B, S), dtype=torch.long, device=device)
        out = self.model(inputs_embeds=prefix_step, attention_mask=attn, use_cache=True, past_key_values=self.prefix_past)
        self.prefix_past = out.past_key_values

    def loss_with_text(self, texts: List[str]) -> torch.Tensor:
        # Compute CE given current prefix_past; do not update cache
        device = next(self.model.parameters()).device
        tok = self.tokenizer(
            texts,
            return_tensors='pt', padding=True, truncation=True,
            max_length=self.max_text_len, add_special_tokens=True,
        )
        input_ids = tok['input_ids'].to(device)  # [B, T]
        attn = tok['attention_mask'].to(device)  # [B, T]

        # BOT + text embeddings
        B = input_ids.size(0)
        bot_ids = torch.full((B, 1), self.bot_token_id, dtype=input_ids.dtype, device=device)
        bot_emb = self.model.get_input_embeddings()(bot_ids)  # [B,1,E]
        tok_emb = self.model.get_input_embeddings()(input_ids)  # [B,T,E]
        inputs_embeds = torch.cat([bot_emb, tok_emb], dim=1)  # [B, 1+T, E]

        # Labels: ignore BOT & padding in text
        ignore = torch.full_like(input_ids, fill_value=-100)
        labels_text = torch.where(attn.bool(), input_ids, ignore)
        labels_full = torch.cat([
            torch.full((B, 1), -100, dtype=labels_text.dtype, device=device),
            labels_text
        ], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels_full,
            use_cache=False,
            past_key_values=self.prefix_past,
        )
        return out.loss

    def forward_stream(self, prefix_seq: torch.Tensor, texts: List[str], reduction: str = 'mean') -> torch.Tensor:
        # prefix_seq: [B, N, E]
        assert prefix_seq.dim() == 3, "prefix_seq must be [B, N, E]"
        self.reset_prefix_cache()
        B, N, E = prefix_seq.shape
        losses = []
        for i in range(N):
            self.step_prefix(prefix_seq[:, i])
            loss_i = self.loss_with_text(texts)
            losses.append(loss_i)
        if reduction == 'last':
            return losses[-1]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def generate_from_prefix(
        self,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> List[str]:
        """Generate text starting from current prefix_past using <BOT> as the first input.

        Returns: list of strings length B.
        """
        device = next(self.model.parameters()).device
        B = len(self.prefix_past[0][0]) if (self.prefix_past is not None) else 1  # fallback
        bot_ids = torch.full((B, 1), self.bot_token_id, dtype=torch.long, device=device)
        gen = self.model.generate(
            input_ids=bot_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            past_key_values=self.prefix_past,
        )
        # gen includes the BOT token at position 0; strip it
        gen_no_bot = gen[:, 1:]
        texts = self.tokenizer.batch_decode(gen_no_bot, skip_special_tokens=True)
        return texts
