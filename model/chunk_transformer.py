from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkTokenEncoder(nn.Module):
    """Encode per-chunk frame embeddings into a fixed number of tokens.

    Args:
        in_dim: Input feature dimensionality (concatenated parts).
        model_dim: Hidden size used by the transformer encoder.
        num_tokens: Number of output tokens per chunk.
        num_layers: Transformer encoder layer count.
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
        num_layers: int = 2,
        num_heads: int = 4,
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
            mlp_dim = max(4 * model_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        *,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode frames to tokens.

        Args:
            x: Tensor of shape ``[B, T, F]``.
            frame_mask: Optional boolean mask ``[B, T]`` where ``True`` marks valid
                frames.

        Returns:
            Tensor of shape ``[B, num_tokens, model_dim]``.
        """

        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, F] input, got {x.shape}")
        key_padding_mask = None
        if frame_mask is not None:
            if frame_mask.shape != x.shape[:2]:
                raise ValueError("frame_mask shape must match the batch/time dims of x.")
            key_padding_mask = ~frame_mask.bool()
        target_dtype = getattr(self.in_proj, "weight", None).dtype if isinstance(self.in_proj, nn.Linear) else x.dtype
        if target_dtype is None:
            target_dtype = x.dtype
        x = x.to(dtype=target_dtype)
        h = self.in_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        if self.num_tokens != h.size(1):
            h = h.transpose(1, 2)
            h = F.interpolate(
                h,
                size=self.num_tokens,
                mode="linear",
                align_corners=False,
            )
            h = h.transpose(1, 2)
        if frame_mask is not None:
            chunk_alive = frame_mask.any(dim=1, keepdim=True).to(h.dtype)
            h = h * chunk_alive.unsqueeze(-1)
        return h
