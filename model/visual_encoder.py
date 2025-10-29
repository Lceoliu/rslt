from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import pdb

from .chunk_transformer import ChunkTokenEncoder
from .parts_gcn import MultiPartGCNModel

__all__ = ["VisualEncoder"]

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

    def initialize_backbones(
        self,
        adjacency: Dict[str, torch.Tensor],
        *,
        in_channels: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if in_channels is None:
            in_channels = 2 if self.multipart.drop_conf else 3
        self.multipart.initialize_backbones(
            adjacency,
            in_channels=in_channels,
            device=device,
        )

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
        """

        batch_size, num_chunks, chunk_len, total_joints, channels = pose.shape
        # pdb.set_trace()
        # features: [B*N_chunk, Parts, chunk_len, gcn_embed_dim]
        # frame_mask: [B*N_chunk, chunk_len]， 代表每一帧是否有效
        features = []
        frame_mask = []
        for i in range(num_chunks):
            chunk = pose[:, i, :, :, :]
            valid_mask = torch.zeros(
                batch_size, chunk_len, dtype=torch.bool, device=pose.device
            )
            for b in range(batch_size):
                valid_chunk_count = pose_len[b] if pose_len is not None else num_chunks
                if i >= valid_chunk_count:
                    valid_len = 0
                elif (i == valid_chunk_count - 1) and (
                    last_chunk_valid_len is not None
                ):
                    valid_len = last_chunk_valid_len[b]
                else:
                    valid_len = chunk_len
                valid_mask[b, :valid_len] = 1
            feature = self.multipart(
                chunk,
                part_lens=part_lens,
                valid_mask=valid_mask,
                adjacency=adjacency,
            )
            features.append(feature)  # N_chunks * [B, P, chunk_len, gcn_embed_dim]
            if valid_mask is not None:
                frame_mask.append(valid_mask)  # N_chunks * [B, chunk_len]
            else:
                frame_mask.append(
                    torch.ones(
                        batch_size, chunk_len, dtype=torch.bool, device=pose.device
                    )
                )
        features = torch.stack(
            features, dim=1
        )  # [B, N_chunk, P, chunk_len, gcn_embed_dim]
        features = features.view(
            batch_size * num_chunks,
            features.size(2),
            features.size(3),
            features.size(4),
        )  # [B*N_chunk, P, chunk_len, gcn_embed_dim]
        frame_mask = torch.stack(frame_mask, dim=1)  # [B, N_chunk, chunk_len]
        frame_mask = frame_mask.view(
            batch_size * num_chunks,
            frame_mask.size(2),
        )  # [B*N_chunk, chunk_len]

        bn, part_count, _, embed_dim = _reshape_features(features)
        # seq: [B*N_chunk, chunk_len, Parts * gcn_embed_dim]
        seq = features.permute(0, 2, 1, 3).contiguous().reshape(bn, chunk_len, part_count * embed_dim)
        # tokens: [B*N_chunk, tokens_per_chunk, llm_dim]
        tokens = self.transformer(seq, frame_mask=frame_mask)
        tokens = tokens.view(batch_size, num_chunks, self.tokens_per_chunk, self.llm_dim)
        frame_mask = frame_mask.view(batch_size, num_chunks, chunk_len)
        # token_mask: [B, N_chunk, tokens_per_chunk]
        token_mask = frame_mask.any(dim=2, keepdim=True).expand(
            -1, -1, self.tokens_per_chunk
        )

        return tokens, token_mask
