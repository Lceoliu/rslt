from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import pdb

from .chunk_transformer import ChunkTokenEncoder
from .parts_gcn import MultiPartGCNModel

__all__ = ["VisualEncoder"]


def _expand_chunk_mask(
    chunk_mask: Optional[torch.Tensor],
    *,
    batch: int,
    num_chunks: int,
    tokens_per_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    if chunk_mask is None:
        return torch.ones(batch, num_chunks, tokens_per_chunk, dtype=torch.bool, device=device)
    if chunk_mask.shape != (batch, num_chunks):
        raise ValueError("chunk_mask shape mismatch.")
    expanded = chunk_mask.view(batch, num_chunks, 1)
    return expanded.expand(-1, -1, tokens_per_chunk)


def _reshape_features(
    features: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if features.dim() != 4:
        raise ValueError("Expected [B*N, P, T, D] features.")
    return features.shape


class VisualEncoder(nn.Module):
    """Multi-part GCN + transformer encoder that outputs chunk tokens."""

    def __init__(
        self,
        *,
        parts: Optional[Sequence[str]] = None,
        drop_conf: bool = True,
        gcn_embed_dim: int = 256,
        gcn_proj_dim: int = 64,
        gcn_temporal_kernel: int = 5,
        gcn_adaptive: bool = True,
        gcn_dropout: float = 0.0,
        tokens_per_chunk: int = 4,
        llm_dim: int = 1024,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_mlp: Optional[int] = None,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.multipart = MultiPartGCNModel(
            parts=parts,
            drop_conf=drop_conf,
            embed_dim=gcn_embed_dim,
            proj_dim=gcn_proj_dim,
            temporal_kernel=gcn_temporal_kernel,
            adaptive=gcn_adaptive,
            dropout=gcn_dropout,
        )
        part_count = len(self.multipart.parts)
        self.tokens_per_chunk = tokens_per_chunk
        self.llm_dim = llm_dim
        self.transformer = ChunkTokenEncoder(
            in_dim=part_count * gcn_embed_dim,
            model_dim=llm_dim,
            num_tokens=tokens_per_chunk,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            mlp_dim=transformer_mlp,
            dropout=transformer_dropout,
        )

    @property
    def parts(self) -> Tuple[str, ...]:
        return self.multipart.parts

    def forward(
        self,
        pose: torch.Tensor,
        *,
        part_lens: Sequence[int],
        pose_len: Optional[torch.Tensor] = None,
        last_chunk_valid_len: Optional[torch.Tensor] = None,
        adjacency: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode chunked pose tensors.

        Args:
            pose: [B, N_chunk, chunk_len, sum_K, C]
            part_lens: Joint counts per part matching ``self.parts``.
            pose_len: Optional valid chunk counts per sample [B].
            last_chunk_valid_len: Optional valid frame counts for the last chunk ``[B]``.
            adjacency: Part adjacency matrices for the first call.

        Returns:
            chunk_tokens: [B, N_chunk, tokens_per_chunk, llm_dim]
            token_mask: [B, N_chunk, tokens_per_chunk] bool mask
            chunk_mask: [B, N_chunk] bool mask
        """

        batch_size, num_chunks, chunk_len, total_joints, channels = pose.shape
        # pdb.set_trace()
        # features: [B*N_chunk, Parts, chunk_len, gcn_embed_dim]
        # frame_mask: [B*N_chunk, chunk_len]， 代表每一帧是否有效
        # chunk_mask: [B, N_chunk]， 代表每一个chunk是否有效
        features, frame_mask, chunk_mask = self.multipart(
            pose,
            part_lens=part_lens,
            pose_len=pose_len,
            last_chunk_valid_len=last_chunk_valid_len,
            adjacency=adjacency,
        )
        bn, part_count, _, embed_dim = _reshape_features(features)
        seq = features.permute(0, 2, 1, 3).contiguous().reshape(bn, chunk_len, part_count * embed_dim)
        # [B*N_chunk, tokens_per_chunk, llm_dim]
        tokens = self.transformer(seq, frame_mask=frame_mask)
        tokens = tokens.view(batch_size, num_chunks, self.tokens_per_chunk, self.llm_dim)
        # [B, N_chunk, tokens_per_chunk]
        token_mask = _expand_chunk_mask(
            chunk_mask,
            batch=batch_size,
            num_chunks=num_chunks,
            tokens_per_chunk=self.tokens_per_chunk,
            device=pose.device,
        )
        if chunk_mask is None:
            chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool, device=pose.device)
        return tokens, token_mask, chunk_mask
