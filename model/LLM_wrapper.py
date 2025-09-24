from typing import List, Optional

import torch
import torch.nn as nn


class LLMWithVisualPrefix(nn.Module):
    """HuggingFace CausalLM wrapper that accepts visual prefix embeddings.

    Sequence formatting (special tokens):
    - Visual scope: <BOV> ... <EOV>
    - Per-chunk:     <BOC> prefix(P tokens) <EOC>
    - Text scope:    <BOT> text ... <EOT>

    Modes:
    - Non-streaming: prefix [B, P, E] or [B, E] ->
        [<BOV>, <BOC>, prefix, <EOC>, <EOV>, <BOT>, text, <EOT>] -> CE loss.
    - Streaming: prefix_seq [B, N, P, E] or [B, N, E] -> iteratively updates KV-cache:
        First step inserts <BOV>, each chunk inserts <BOC> chunk <EOC>, and the last
        chunk additionally inserts <EOV>. For each step, compute CE on (<BOT>, text, <EOT>).

    目的：将视觉前缀嵌入作为文本生成的上下文信息，并在最外层包裹<BOV>/<EOV>，每个chunk包裹<BOC>/<EOC>，文本以<BOT>开始，以<EOT>结束。
    """

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        max_text_len: int = 128,
        gradient_checkpointing: bool = False,
        freeze_lm: bool = False,
        bot_token: str = "<BOT>",  # begin of translation
        bov_token: str = "<BOV>",  # begin of visual scope
        eov_token: str = "<EOV>",  # end of visual scope
        boc_token: str = "<BOC>",  # begin of chunk
        eoc_token: str = "<EOC>",  # end of chunk
        eot_token: str = "<EOT>",  # end of text
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
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

        # Store special tokens
        self.bot_token = bot_token
        self.bov_token = bov_token
        self.eov_token = eov_token
        self.boc_token = boc_token
        self.eoc_token = eoc_token
        self.eot_token = eot_token
        
        # Add all special tokens at once to avoid overwriting
        all_special_tokens = [
            self.bot_token, self.bov_token, self.eov_token, 
            self.boc_token, self.eoc_token, self.eot_token
        ]
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": all_special_tokens
        })

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
        self.bov_token_id = self.tokenizer.convert_tokens_to_ids(self.bov_token)
        self.eov_token_id = self.tokenizer.convert_tokens_to_ids(self.eov_token)
        self.boc_token_id = self.tokenizer.convert_tokens_to_ids(self.boc_token)
        self.eoc_token_id = self.tokenizer.convert_tokens_to_ids(self.eoc_token)
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.eot_token)

        # Internal flags for streaming boundaries
        self._bov_started: bool = False
        self._eov_finished: bool = False

    # -------- Non-streaming --------
    def forward(self, prefix_embeds: torch.Tensor, texts: List[str]) -> torch.Tensor:
        # for one chunk
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
            max_length=self.max_text_len, add_special_tokens=False,
        )
        input_ids = tok['input_ids'].to(device)  # [B, T], T = max_text_len
        attn = tok['attention_mask'].to(device)  # [B, T], 1 for valid tokens

        # BOT embedding
        bot_ids = torch.full((B, 1), self.bot_token_id, dtype=input_ids.dtype, device=device)
        bot_emb = self.model.get_input_embeddings()(bot_ids)  # [B,1,E]
        bot_emb = bot_emb.to(model_dtype)
        # BOV/EOV embeddings for visual scope
        bov_ids = torch.full((B, 1), self.bov_token_id, dtype=input_ids.dtype, device=device)
        eov_ids = torch.full((B, 1), self.eov_token_id, dtype=input_ids.dtype, device=device)
        bov_emb = self.model.get_input_embeddings()(bov_ids).to(model_dtype)
        eov_emb = self.model.get_input_embeddings()(eov_ids).to(model_dtype)
        boc_ids = torch.full(
            (B, 1), self.boc_token_id, dtype=input_ids.dtype, device=device
        )
        boc_emb = self.model.get_input_embeddings()(boc_ids)  # [B,1,E]
        boc_emb = boc_emb.to(model_dtype)
        eoc_ids = torch.full(
            (B, 1), self.eoc_token_id, dtype=input_ids.dtype, device=device
        )
        eoc_emb = self.model.get_input_embeddings()(eoc_ids)  # [B,1,E]
        eoc_emb = eoc_emb.to(model_dtype)
        # Text embeddings
        tok_emb = self.model.get_input_embeddings()(input_ids)  # [B, T, E]
        tok_emb = tok_emb.to(model_dtype)
        # EOT embedding appended after text
        eot_ids = torch.full((B, 1), self.eot_token_id, dtype=input_ids.dtype, device=device)
        eot_emb = self.model.get_input_embeddings()(eot_ids).to(model_dtype)
        inputs_embeds = torch.cat(
            [bov_emb, boc_emb, prefix_embeds, eoc_emb, eov_emb, bot_emb, tok_emb, eot_emb], dim=1
        )  # [B, P + (markers) + T + 2, E]

        # Attention mask
        prefix_mask = torch.ones((B, P + 5), dtype=attn.dtype, device=device)  # BOV,BOC,P...,EOC,EOV,BOT
        attn_text = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], dim=1)  # text + EOT
        attn_full = torch.cat([prefix_mask, attn_text], dim=1)

        # Labels: ignore prefix and BOT (-100); ignore padding in text
        ignore = torch.full_like(input_ids, fill_value=-100)
        labels_text = torch.where(attn.bool(), input_ids, ignore)
        # Append EOT label (always to be predicted)
        labels_text_eot = torch.cat([
            labels_text,
            torch.full((B, 1), self.eot_token_id, dtype=labels_text.dtype, device=device)
        ], dim=1)
        labels_full = torch.cat(
            [
                torch.full((B, P + 5), -100, dtype=labels_text.dtype, device=device),  # BOV,BOC,P...,EOC,EOV,BOT
                labels_text_eot,
            ],
            dim=1,
        )

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_full, labels=labels_full)
        return out.loss

    # -------- Streaming KV-cache API --------
    def reset_prefix_cache(self) -> None:
        self.prefix_past = None
        self._bov_started = False
        self._eov_finished = False

    def step_prefix(self, prefix_step: torch.Tensor, is_last: bool = False) -> None:
        """Append one chunk of prefix to the KV cache.

        Accepts either [B, E] (P=1) or [B, P, E]. The visual scope is wrapped by
        <BOV> at the very beginning (first call), and <EOV> at the end if
        is_last=True. Each chunk is wrapped with <BOC> and <EOC> before feeding
        into the LLM cache.
        """
        model_dtype = self.model.get_input_embeddings().weight.dtype
        if prefix_step.dim() == 2:
            # [B, E] -> [B, 1, E]
            prefix_step = prefix_step.unsqueeze(1)
        elif prefix_step.dim() != 3:
            raise AssertionError("prefix_step must be [B, E] or [B, P, E]")

        prefix_step = prefix_step.to(model_dtype)
        B, P, E = prefix_step.shape
        device = prefix_step.device

        emb_layer = self.model.get_input_embeddings()
        # Optional BOV at the start (only once)
        pieces = []
        if not self._bov_started:
            bov_ids = torch.full((B, 1), self.bov_token_id, dtype=torch.long, device=device)
            bov_emb = emb_layer(bov_ids).to(model_dtype)
            pieces.append(bov_emb)
            self._bov_started = True

        # Prepare BOC/EOC embeddings
        boc_ids = torch.full((B, 1), self.boc_token_id, dtype=torch.long, device=device)
        eoc_ids = torch.full((B, 1), self.eoc_token_id, dtype=torch.long, device=device)
        boc_emb = emb_layer(boc_ids).to(model_dtype)  # [B,1,E]
        eoc_emb = emb_layer(eoc_ids).to(model_dtype)  # [B,1,E]
        pieces.extend([boc_emb, prefix_step, eoc_emb])
        # Optional EOV at the very end
        if is_last and (not self._eov_finished):
            eov_ids = torch.full((B, 1), self.eov_token_id, dtype=torch.long, device=device)
            eov_emb = emb_layer(eov_ids).to(model_dtype)
            pieces.append(eov_emb)
            self._eov_finished = True

        step_embeds = torch.cat(pieces, dim=1)
        attn = torch.ones((B, step_embeds.shape[1]), dtype=torch.long, device=device)
        out = self.model(
            inputs_embeds=step_embeds,
            attention_mask=attn,
            use_cache=True,
            past_key_values=self.prefix_past,
        )
        self.prefix_past = out.past_key_values

    def end_visual(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None) -> None:
        """Explicitly append <EOV> to the cache if not already finished.

        If batch_size/device are not provided, they are inferred from the cache.
        Safe to call multiple times.
        """
        if self._eov_finished:
            return
        if self.prefix_past is None:
            # Nothing to end
            return
        # Infer B and device
        if batch_size is None or device is None:
            try:
                k0 = self.prefix_past[0][0]
                batch_size = k0.size(0)
                device = k0.device
            except Exception:
                return
        assert batch_size is not None and device is not None
        model_dtype = self.model.get_input_embeddings().weight.dtype
        emb_layer = self.model.get_input_embeddings()
        eov_ids = torch.full((batch_size, 1), self.eov_token_id, dtype=torch.long, device=device)
        eov_emb = emb_layer(eov_ids).to(model_dtype)
        attn = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        out = self.model(
            inputs_embeds=eov_emb,
            attention_mask=attn,
            use_cache=True,
            past_key_values=self.prefix_past,
        )
        self.prefix_past = out.past_key_values
        self._eov_finished = True

    def loss_with_text(self, texts: List[str]) -> torch.Tensor:
        # Compute CE given current prefix_past; do not update cache
        device = next(self.model.parameters()).device
        tok = self.tokenizer(
            texts,
            return_tensors='pt', padding=True, truncation=True,
            max_length=self.max_text_len, add_special_tokens=False,
        )
        input_ids = tok['input_ids'].to(device)  # [B, T]
        attn = tok['attention_mask'].to(device)  # [B, T]

        # BOT + text + EOT embeddings
        B = input_ids.size(0)
        bot_ids = torch.full((B, 1), self.bot_token_id, dtype=input_ids.dtype, device=device)
        bot_emb = self.model.get_input_embeddings()(bot_ids)  # [B,1,E]
        tok_emb = self.model.get_input_embeddings()(input_ids)  # [B,T,E]
        eot_ids = torch.full((B, 1), self.eot_token_id, dtype=input_ids.dtype, device=device)
        eot_emb = self.model.get_input_embeddings()(eot_ids)
        inputs_embeds = torch.cat([bot_emb, tok_emb, eot_emb], dim=1)  # [B, 1+T+1, E]

        # Labels: ignore BOT & padding in text
        ignore = torch.full_like(input_ids, fill_value=-100)
        labels_text = torch.where(attn.bool(), input_ids, ignore)
        labels_full = torch.cat([
            torch.full((B, 1), -100, dtype=labels_text.dtype, device=device),  # BOT
            labels_text,
            torch.full((B, 1), self.eot_token_id, dtype=labels_text.dtype, device=device),  # EOT label
        ], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels_full,
            use_cache=False,
            past_key_values=self.prefix_past,
        )
        return out.loss

    def forward_stream(
        self,
        prefix_seq: torch.Tensor,
        texts: List[str],
        reduction: str = 'mean',
        skip_prob: float = 0.0,
        keep_first: bool = True,
        always_keep_last: bool = True,
    ) -> torch.Tensor:
        # prefix_seq: [B, N, P, E] or [B, N, E] (P=1)
        assert prefix_seq.dim() in (3, 4), "prefix_seq must be [B, N, E] or [B, N, P, E]"
        self.reset_prefix_cache()
        if prefix_seq.dim() == 3:
            B, N, E = prefix_seq.shape
            P = 1
        else:
            B, N, P, E = prefix_seq.shape
        device = prefix_seq.device
        losses = []
        for i in range(N):
            # Each step is either [B, E] (when P=1 and dim=3 was given) or [B, P, E]
            step = prefix_seq[:, i]  # shape [B, E] or [B, P, E]
            self.step_prefix(step, is_last=(i == N - 1))
            # Randomly skip CE computation for some chunks, but always include policy guards
            take = True
            if skip_prob > 0.0:
                r = torch.rand((), device=device)
                take = bool(r.item() >= float(skip_prob))
            if keep_first and i == 0:
                take = True
            if always_keep_last and i == (N - 1):
                take = True
            if take:
                loss_i = self.loss_with_text(texts)
                losses.append(loss_i)
        if reduction == 'last':
            # With guards, losses is guaranteed non-empty; last corresponds to last kept
            return losses[-1]
        # Mean over kept losses (guaranteed non-empty by guards)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def generate_from_prefix(
        self,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> List[str]:
        """Greedy decode starting from current prefix_past using <BOT> as the first input.

        Returns: list of strings length B.
        """
        device = next(self.model.parameters()).device
        if self.prefix_past is None:
            # No prefix; fallback to just BOT
            B = 1
        else:
            try:
                B = self.prefix_past[0][0].size(0)
            except Exception:
                B = 1

        # Ensure <EOV> is appended before text decoding if visual scope is open
        if (self.prefix_past is not None) and (not self._eov_finished):
            self.end_visual(batch_size=B, device=device)

        # Prepare local past; if needed, append <EOV> without mutating self.prefix_past
        past = self.prefix_past
        if (past is not None) and (not self._eov_finished):
            model_dtype = self.model.get_input_embeddings().weight.dtype
            emb_layer = self.model.get_input_embeddings()
            eov_ids = torch.full((B, 1), self.eov_token_id, dtype=torch.long, device=device)
            eov_emb = emb_layer(eov_ids).to(model_dtype)
            attn = torch.ones((B, 1), dtype=torch.long, device=device)
            out = self.model(inputs_embeds=eov_emb, attention_mask=attn, use_cache=True, past_key_values=past)
            past = out.past_key_values

        # Start with BOT token
        input_ids = torch.full((B, 1), self.bot_token_id, dtype=torch.long, device=device)
        outputs = self.model(input_ids=input_ids, use_cache=True, past_key_values=past)
        past = outputs.past_key_values
        sequences = []  # collect generated token ids per step
        prev_tokens = None

        for step in range(max_new_tokens):
            if prev_tokens is None:
                # use last token of current outputs (BOT)
                logits = outputs.logits[:, -1, :]
            else:
                out = self.model(input_ids=prev_tokens, use_cache=True, past_key_values=past)
                past = out.past_key_values
                logits = out.logits[:, -1, :]

            if do_sample:
                # basic temperature sampling with optional top_k
                logits = logits / max(1e-6, float(temperature))
                if top_k and top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    probs = torch.softmax(topk_vals, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                    next_tokens = topk_idx.gather(-1, next_idx)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            sequences.append(next_tokens)
            prev_tokens = next_tokens

            # Early stop if any sample has produced eos or <EOT>
            eos = self.tokenizer.eos_token_id
            if eos is not None:
                if torch.any(prev_tokens.squeeze(-1) == eos):
                    break
            if self.eot_token_id is not None:
                if torch.any(prev_tokens.squeeze(-1) == self.eot_token_id):
                    break

        if sequences:
            gen_ids = torch.cat(sequences, dim=1)  # [B, L]
        else:
            gen_ids = torch.zeros((B, 0), dtype=torch.long, device=device)
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return texts
