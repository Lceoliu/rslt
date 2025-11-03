from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn


class LLMWithVisualPrefix(nn.Module):
    """Wrap a Hugging Face CausalLM to accept visual token embeddings with a prompt prefix."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        trust_remote_code: bool = True,
        max_text_len: int = 128,
        gradient_checkpointing: bool = False,
        freeze_lm: bool = False,
        verbose: bool = False,
        compute_special_token_loss: bool = False,
        prompt_text: str = "\u8bf7\u5c06\u63a5\u4e0b\u6765\u7684\u624b\u8bed\u5185\u5bb9\u7ffb\u8bd1\u6210\u6587\u5b57\uff1a",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Install transformers to use LLMWithVisualPrefix."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            print("Using eos token as padding token!")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.verbose = verbose
        self.prompt_text = prompt_text
        self._prompt_ids: Optional[torch.Tensor] = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.verbose:
            print(
                f"Loaded LLM: {model_name_or_path}, tokenizer size {len(self.tokenizer)}"
            )
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if freeze_lm:
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self.model, "get_input_embeddings"):
                self.model.get_input_embeddings().weight.requires_grad = False

        config = self.model.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "n_embd", None)
        if hidden_size is None:
            raise RuntimeError("Unable to infer hidden size from LLM config.")
        self.hidden_size = int(hidden_size)

        self.max_text_len = int(max_text_len)
        self.compute_special_token_loss = compute_special_token_loss  # kept for config compatibility

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
        self,
        chunk_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        texts: Sequence[str],
    ):
        """
        Compute autoregressive loss given chunk embeddings and texts.

        chunk_tokens: [B, N, T, E]
        token_mask: [B, N, T] bool
        texts: List[str] with length B
        """
        if chunk_tokens.dim() != 4:
            raise ValueError("chunk_tokens must be [B, N, T, E].")
        if token_mask.shape != chunk_tokens.shape[:3]:
            raise ValueError("token_mask must match chunk_tokens batch/length dims.")
        if len(texts) != chunk_tokens.size(0):
            raise ValueError("texts length must equal batch size.")

        embed_layer = self.model.get_input_embeddings()
        model_dtype = embed_layer.weight.dtype
        chunk_tokens = chunk_tokens.to(model_dtype)
        token_mask = token_mask.to(torch.bool)

        batch_size, num_chunks, tokens_per_chunk, embed_dim = chunk_tokens.shape
        flat_tokens = chunk_tokens.reshape(batch_size, num_chunks * tokens_per_chunk, embed_dim)
        flat_mask = token_mask.reshape(batch_size, num_chunks * tokens_per_chunk)

        prompt_embeds = self._get_prompt_embeddings(device=chunk_tokens.device)
        eos_id = self.tokenizer.eos_token_id
        eos_embed = embed_layer(torch.tensor([eos_id], device=chunk_tokens.device))

        seq_embeds: List[torch.Tensor] = []
        seq_labels: List[torch.Tensor] = []

        for visual_embed, visual_mask, text in zip(flat_tokens, flat_mask, texts):
            valid_visual = visual_embed[visual_mask]
            prefix_embeds = torch.cat([prompt_embeds, valid_visual], dim=0)
            prefix_labels = torch.full(
                (prefix_embeds.size(0),),
                fill_value=-100,
                dtype=torch.long,
                device=chunk_tokens.device,
            )

            text_ids = self._tokenize_text(text).to(chunk_tokens.device)
            text_embeds = embed_layer(text_ids.unsqueeze(0)).squeeze(0)

            seq = torch.cat([prefix_embeds, text_embeds, eos_embed], dim=0)
            text_labels = torch.cat(
                [text_ids, torch.tensor([eos_id], device=chunk_tokens.device)]
            )
            labels = torch.cat([prefix_labels, text_labels], dim=0)
            if labels.numel() > 0:
                labels[0] = -100

            seq_embeds.append(seq)
            seq_labels.append(labels)

        inputs_embeds, attention_mask, labels = self._pad_sequences(seq_embeds, seq_labels)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs, labels

    @torch.no_grad()
    def generate(
        self,
        chunk_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 10,
    ) -> List[str]:
        if chunk_tokens.dim() != 4:
            raise ValueError("chunk_tokens must be [B, N, T, E].")
        if token_mask.shape != chunk_tokens.shape[:3]:
            raise ValueError("token_mask must match chunk_tokens batch/length dims.")

        embed_layer = self.model.get_input_embeddings()
        model_dtype = embed_layer.weight.dtype
        chunk_tokens = chunk_tokens.to(model_dtype)
        token_mask = token_mask.to(torch.bool)

        batch_size, num_chunks, tokens_per_chunk, embed_dim = chunk_tokens.shape
        flat_tokens = chunk_tokens.reshape(batch_size, num_chunks * tokens_per_chunk, embed_dim)
        flat_mask = token_mask.reshape(batch_size, num_chunks * tokens_per_chunk)

        prompt_embeds = self._get_prompt_embeddings(device=chunk_tokens.device)
        eos_id = self.tokenizer.eos_token_id

        prefix_embeds_list: List[torch.Tensor] = []
        for visual_embed, visual_mask in zip(flat_tokens, flat_mask):
            valid_visual = visual_embed[visual_mask]
            prefix = torch.cat([prompt_embeds, valid_visual], dim=0)
            prefix_embeds_list.append(prefix)

        inputs_embeds, attention_mask, lengths = self._pad_prefixes(prefix_embeds_list)

        sequences = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.5,
        )

        results: List[str] = []
        for seq, prefix_len in zip(sequences, lengths.tolist()):
            gen_tokens = seq[prefix_len:]
            if eos_id is not None and (gen_tokens == eos_id).any():
                stop = torch.nonzero(gen_tokens == eos_id, as_tuple=False)[0].item()
                gen_tokens = gen_tokens[:stop]
            text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
            results.append(text)
        return results

    def _get_prompt_ids(self) -> torch.Tensor:
        if self._prompt_ids is None:
            encoded = self.tokenizer(
                self.prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"][0]
            if prompt_ids.numel() == 0:
                prompt_ids = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            self._prompt_ids = prompt_ids
        return self._prompt_ids

    def _get_prompt_embeddings(self, device: torch.device) -> torch.Tensor:
        prompt_ids = self._get_prompt_ids().to(device)
        embed_layer = self.model.get_input_embeddings()
        return embed_layer(prompt_ids.unsqueeze(0)).squeeze(0)

    def _pad_prefixes(
        self,
        embeds: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad variable-length prefix embeddings to a batch tensor.
        """
        max_len = max(seq.size(0) for seq in embeds) if embeds else 1
        batch = len(embeds)
        device = embeds[0].device if embeds else torch.device("cpu")
        dtype = embeds[0].dtype if embeds else torch.float32

        padded_embeds = torch.zeros(
            batch,
            max_len,
            self.hidden_size,
            dtype=dtype,
            device=device,
        )
        attention = torch.zeros(batch, max_len, dtype=torch.long, device=device)
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        for idx, seq in enumerate(embeds):
            length = seq.size(0)
            padded_embeds[idx, :length] = seq
            attention[idx, :length] = 1
            lengths[idx] = length
        return padded_embeds, attention, lengths

    def _pad_sequences(
        self,
        embeds: Sequence[torch.Tensor],
        labels: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(seq.size(0) for seq in embeds)
        batch = len(embeds)
        device = embeds[0].device
        model_dtype = embeds[0].dtype

        padded_embeds = torch.zeros(
            batch,
            max_len,
            self.hidden_size,
            dtype=model_dtype,
            device=device,
        )
        padded_labels = torch.full(
            (batch, max_len),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros(
            batch,
            max_len,
            dtype=torch.long,
            device=device,
        )
        for idx, (seq, lab) in enumerate(zip(embeds, labels)):
            length = seq.size(0)
            padded_embeds[idx, :length] = seq
            padded_labels[idx, :length] = lab
            attention[idx, :length] = 1
        return padded_embeds, attention, padded_labels

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """
        Input: text string
        Output: token ids tensor [L]
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return encoded["input_ids"][0].to(self.device)

    def get_texts_ids(self, texts: Sequence[str]) -> Sequence[torch.Tensor]:
        return [self._tokenize_text(text) for text in texts]

    def get_id_embeddings(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        embed_layer = self.model.get_input_embeddings()
        return embed_layer(ids)

    def sample_negative_embeddings(
        self,
        num_samples: int,
        positive_ids: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = self.model.config.vocab_size
        device = self.device
        dtype = self.model.get_input_embeddings().weight.dtype

        random_ids = torch.randint(0, vocab_size, (num_samples,), device=device)
        mask = (random_ids.unsqueeze(1) == positive_ids.unsqueeze(0)).any(dim=1)
        negative_ids = random_ids[~mask]

        if negative_ids.numel() < num_samples:
            negative_ids = torch.randint(0, vocab_size, (num_samples,), device=device)

        embeds = self.get_id_embeddings(negative_ids)
        return embeds.to(dtype)
