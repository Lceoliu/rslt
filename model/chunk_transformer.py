from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < b) & (tmp > a)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class ChunkTokenEncoder(nn.Module):
    """Encode per-chunk frame embeddings into a fixed number of tokens using a Q-Former style mechanism.

    This module first processes the input sequence with a self-attention encoder,
    then uses a set of learnable query tokens to distill the information into a
    fixed-size output via cross-attention, following modern Transformer best practices.

    Args:
        in_dim: Input feature dimensionality (concatenated parts).
        model_dim: Hidden size used by the transformer layers (must match LLM embedding dim).
        num_tokens: Number of output tokens per chunk (number of queries).
        max_temporal_len: Maximum expected number of frames in a chunk for positional embedding.
        num_layers: Transformer encoder layer count for self-attention on visual features.
        decoder_layers: Transformer decoder layer count for cross-attention.
        num_heads: Attention heads per layer.
        mlp_dim: Feed-forward width inside the transformer.
        dropout: Dropout rate applied inside transformer layers.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        model_dim: int,
        num_tokens: int,
        max_temporal_len: int = 128,  # Max frames for PE
        num_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 8,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        self.num_tokens = num_tokens
        self.model_dim = model_dim

        self.in_proj = nn.Linear(in_dim, model_dim) if in_dim != model_dim else nn.Identity()

        if mlp_dim is None:
            mlp_dim = max(4 * model_dim, 1024)

        # Positional and Slot Embeddings
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_temporal_len, model_dim))
        self.slot_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, model_dim))

        # Visual Encoder (Self-Attention)
        # Using norm_first=True for Pre-LN, which is more stable.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True, # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Q-Former part: Learnable queries and Cross-attention decoder
        self.query_tokens = nn.Parameter(torch.zeros(1, num_tokens, model_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True, # Pre-LN
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            # Use Xavier uniform for linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            # Use truncated normal for learnable embeddings/tokens
            if m.dim() > 1:
                _trunc_normal_(m, std=.02)

    def forward(
        self,
        x: torch.Tensor,
        *,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode frames to tokens using Q-Former style cross-attention.

        Args:
            x: Tensor of shape ``[B, T, F]``.
            frame_mask: Optional boolean mask ``[B, T]`` where ``True`` marks valid
                frames.

        Returns:
            Tensor of shape ``[B, num_tokens, model_dim]``.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, F] input, got {x.shape}")

        target_dtype = self.query_tokens.dtype
        target_device = self.query_tokens.device
        x = x.to(device=target_device, dtype=target_dtype)

        B, T, _ = x.shape
        key_padding_mask = None
        if frame_mask is not None:
            if frame_mask.shape != x.shape[:2]:
                raise ValueError("frame_mask shape must match the batch/time dims of x.")
            key_padding_mask = ~frame_mask.bool()

        x_proj = self.in_proj(x)

        # Add temporal positional embedding
        if T > self.temporal_pos_embed.shape[1]:
            raise ValueError(f"Input sequence length ({T}) exceeds max_temporal_len ({self.temporal_pos_embed.shape[1]})")
        x_proj = x_proj + self.temporal_pos_embed[:, :T, :]

        memory = self.encoder(x_proj, src_key_padding_mask=key_padding_mask)

        # queries with slot positional embedding
        query = self.query_tokens + self.slot_pos_embed
        query = query.expand(B, -1, -1)

        output_tokens = self.decoder(
            tgt=query, 
            memory=memory, 
            tgt_key_padding_mask=None, # Queries are not padded
            memory_key_padding_mask=key_padding_mask
        )

        # Zero out tokens from fully padded chunks
        if frame_mask is not None:
            chunk_alive = frame_mask.any(dim=1, keepdim=True).to(output_tokens.dtype)
            output_tokens = output_tokens * chunk_alive.unsqueeze(-1)

        return output_tokens
