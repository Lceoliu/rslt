from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import pdb


class LLMWithVisualPrefix(nn.Module):
    """Wrap a Hugging Face CausalLM to accept visual chunk embeddings."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        trust_remote_code: bool = True,
        max_text_len: int = 128,
        gradient_checkpointing: bool = False,
        freeze_lm: bool = False,
        boc_token: str = "<BOC>",
        eoc_token: str = "<EOC>",
        bot_token: str = "<BOT>",
        eot_token: str = "<EOT>",
        verbose: bool = False,
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
        original_tokenizer_size = len(self.tokenizer)
        if self.tokenizer.pad_token_id is None:
            print("Using eos token as padding token!")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.verbose = verbose
        self.special_tokens = {
            "boc": boc_token,
            "eoc": eoc_token,
            "bot": bot_token,
            "eot": eot_token,
        }
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.special_tokens.values())}
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        if self.verbose:
            print(
                f"Loaded LLM: {model_name_or_path}, tokenizer size {original_tokenizer_size} -> {len(self.tokenizer)}"
            )
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if freeze_lm:
            for param in self.model.parameters():
                param.requires_grad = False

        config = self.model.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "n_embd", None)
        if hidden_size is None:
            raise RuntimeError("Unable to infer hidden size from LLM config.")
        self.hidden_size = int(hidden_size)

        self.max_text_len = int(max_text_len)
        self._special_ids = {
            name: self.tokenizer.convert_tokens_to_ids(token)
            for name, token in self.special_tokens.items()
        }

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
        self,
        chunk_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        texts: Sequence[str],
    ):
        """Compute autoregressive loss given chunk embeddings and texts."""

        if chunk_tokens.dim() != 4:
            raise ValueError("chunk_tokens must be [B, N, P, E].")
        if token_mask.shape != chunk_tokens.shape[:3]:
            raise ValueError("token_mask must match chunk_tokens batch/length dims.")
        if len(texts) != chunk_tokens.size(0):
            raise ValueError("texts length must equal batch size.")

        if self.verbose:
            print(f"Texts: {texts}")

        model_dtype = self.model.get_input_embeddings().weight.dtype
        chunk_tokens = chunk_tokens.to(model_dtype)
        token_mask = token_mask.to(torch.bool)

        special_embeds = self._get_special_embeddings(device=chunk_tokens.device)
        embed_layer = self.model.get_input_embeddings()

        seq_embeds: List[torch.Tensor] = []
        seq_labels: List[torch.Tensor] = []
        for chunk_embed, mask, text in zip(chunk_tokens, token_mask, texts):
            # chunk_embed: [N, P, E], mask: [N, P], N为chunk数量，P为Token_Per_chunk数量，E为embedding维度
            prefix_embeds, prefix_ids = self._build_prefix(
                chunk_embed,
                mask,
                embed_layer,
                special_embeds,
            )
            # pdb.set_trace()
            text_ids = self._tokenize_text(text)
            text_embeds = embed_layer(text_ids.unsqueeze(0)).squeeze(0)
            eot_embed = special_embeds["eot"]

            seq = torch.cat([prefix_embeds, text_embeds, eot_embed], dim=0)
            labels = torch.cat(
                [
                    self._prefix_labels_from_ids(prefix_ids),
                    text_ids.to(prefix_ids.device),
                    torch.tensor([self._special_ids["eot"]], device=prefix_ids.device),
                ]
            )
            if labels.numel() > 0:
                labels[0] = -100  # first token has no target
            seq_embeds.append(seq)
            seq_labels.append(labels)

        inputs_embeds, attention_mask, labels = self._pad_sequences(seq_embeds, seq_labels)
        # pdb.set_trace()
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
            raise ValueError("chunk_tokens must be [B, N, P, E].")
        if token_mask.shape != chunk_tokens.shape[:3]:
            raise ValueError("token_mask must match chunk_tokens batch/length dims.")

        model_dtype = self.model.get_input_embeddings().weight.dtype
        chunk_tokens = chunk_tokens.to(model_dtype)
        token_mask = token_mask.to(torch.bool)
        # pdb.set_trace()

        special_embeds = self._get_special_embeddings(device=chunk_tokens.device)
        embed_layer = self.model.get_input_embeddings()

        prefix_embeds_list: List[torch.Tensor] = []
        prefix_id_list: List[torch.Tensor] = []
        for chunk_embed, mask in zip(chunk_tokens, token_mask):
            prefix_embeds, prefix_ids = self._build_prefix(
                chunk_embed,
                mask,
                embed_layer,
                special_embeds,
            )
            prefix_embeds_list.append(prefix_embeds)
            prefix_id_list.append(prefix_ids)
        if self.verbose:
            print(f"prefix ids: {prefix_id_list}")

        inputs_embeds, attention_mask, lengths = self._pad_prefixes(prefix_embeds_list)

        sequences = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self._special_ids["eot"],
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.5,
        )

        results: List[str] = []
        for seq, prefix_len in zip(sequences, lengths.tolist()):
            gen_tokens = seq[:]
            eot_id = self._special_ids["eot"]
            if (gen_tokens == eot_id).any():
                stop = torch.nonzero(gen_tokens == eot_id, as_tuple=False)[0].item()
                gen_tokens = gen_tokens[:stop]
            text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
            results.append(text)
        return results

    def _build_prefix(
        self,
        chunk_embed: torch.Tensor,
        mask: torch.Tensor,
        embed_layer: nn.Embedding,
        special_embeds: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        chunk_embed: [N, P, E]
        mask: [N, P] bool
        Returns:
            prefix: [L, E], 包含 <BOC> chunk1 <EOC> <BOC> chunk2 <EOC> ... <BOT>
            prefix_ids: [L] (-1 for non-token positions)
        """
        parts: List[torch.Tensor] = []
        token_ids: List[int] = []
        for chunk_vecs, chunk_mask in zip(chunk_embed, mask):
            # chunk_vecs: [P, E], chunk_mask: [P]
            valid_idx = torch.nonzero(chunk_mask, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            parts.append(special_embeds["boc"])
            token_ids.append(self._special_ids["boc"])
            selected = chunk_vecs.index_select(0, valid_idx)
            parts.append(selected)
            token_ids.extend([-1] * selected.size(0))
            parts.append(special_embeds["eoc"])
            token_ids.append(self._special_ids["eoc"])
        parts.append(special_embeds["bot"])
        token_ids.append(self._special_ids["bot"])
        prefix = torch.cat(parts, dim=0)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=prefix.device)
        return prefix, token_ids_tensor

    def _prefix_labels_from_ids(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        labels = prefix_ids.clone()
        labels[labels == -1] = -100
        return labels
        # Ignore all prefix tokens by setting their labels to -100.
        # The model should only be trained to predict the text tokens.
        # return torch.full_like(prefix_ids, -100)

    def _pad_prefixes(
        self,
        embeds: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        把不同长度的embeddings转化到右padding的形式

        Input: embeds: List of [L_i, E] tensors

        Output: padded_embeds: [B, L_max, E]
                attention_mask: [B, L_max] (1 for valid, 0 for padding)
                lengths: [B] lengths of each sequence
        """
        max_len = max(seq.size(0) for seq in embeds)
        batch = len(embeds)
        device = embeds[0].device
        dtype = embeds[0].dtype

        padded_embeds = torch.zeros(
            batch,
            max_len,
            self.hidden_size,
            dtype=dtype,
            device=device,
        )  # [B, L_max, E]
        attention = torch.zeros(batch, max_len, dtype=torch.long, device=device)
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        for idx, seq in enumerate(embeds):
            length = seq.size(0)
            padded_embeds[idx, :length] = seq
            attention[idx, :length] = 1
            lengths[idx] = length
        return padded_embeds, attention, lengths

    def _pad_prefix_token_ids(
        self,
        token_id_list: Sequence[torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        pad_id = self.tokenizer.pad_token_id
        max_len = int(lengths.max().item())
        batch = len(token_id_list)
        ids = torch.full(
            (batch, max_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=lengths.device,
        )
        for idx, token_ids in enumerate(token_id_list):
            padded = torch.where(token_ids == -1, torch.full_like(token_ids, pad_id), token_ids)
            ids[idx, : token_ids.numel()] = padded
        return ids

    def _tokenize_text(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return encoded["input_ids"][0].to(self.device)

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

    def _get_special_embeddings(self, device: torch.device) -> Dict[str, torch.Tensor]:
        embed_layer = self.model.get_input_embeddings()
        ids = torch.tensor(
            [
                self._special_ids["boc"],
                self._special_ids["eoc"],
                self._special_ids["bot"],
                self._special_ids["eot"],
            ],
            device=device,
        )
        embeds = embed_layer(ids)
        embeds = embeds.to(embed_layer.weight.dtype)
        return {
            "boc": embeds[0:1],
            "eoc": embeds[1:2],
            "bot": embeds[2:3],
            "eot": embeds[3:4],
        }
