from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import warnings

import torch
import torch.nn as nn
import pdb

from .parts_gcn import MultiPartGCNModel

__all__ = ["VisualEncoder"]


def _reshape_features(
    features: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if features.dim() != 4:
        raise ValueError("Expected [B*N, P, T, D] features.")
    return features.shape


class VisualEncoder(nn.Module):
    """Multi-part GCN encoder that downsamples temporal frames into LLM tokens."""

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
        sampling_stride: int = 2,
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
        if sampling_stride <= 0:
            raise ValueError("sampling_stride must be a positive integer.")
        self.sampling_stride = int(sampling_stride)
        self.tokens_per_chunk = int(tokens_per_chunk) if tokens_per_chunk > 0 else None
        self._tokens_warned = False
        self.llm_dim = llm_dim
        self.projection = nn.Linear(part_count * gcn_embed_dim, llm_dim)

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
        # seq: [B*N_chunk, chunk_len, Parts * gcn_embed_dim]
        seq = (
            features.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(bn, chunk_len, part_count * embed_dim)
        )
        if chunk_len % self.sampling_stride != 0:
            raise ValueError(
                f"chunk_len ({chunk_len}) must be divisible by sampling_stride ({self.sampling_stride})."
            )
        seq = seq[:, :: self.sampling_stride, :]
        if frame_mask is not None:
            frame_mask = frame_mask[:, :: self.sampling_stride]
            seq = seq * frame_mask.unsqueeze(-1).to(seq.dtype)
        tokens = self.projection(seq)
        tokens_per_chunk = tokens.size(1)
        if self.tokens_per_chunk is None:
            self.tokens_per_chunk = tokens_per_chunk
        elif self.tokens_per_chunk != tokens_per_chunk:
            if not self._tokens_warned:
                warnings.warn(
                    (
                        "Configured tokens_per_chunk differs from the computed value "
                        f"({self.tokens_per_chunk} vs {tokens_per_chunk}); "
                        "using the computed value for downstream modules."
                    ),
                    RuntimeWarning,
                )
                self._tokens_warned = True
            self.tokens_per_chunk = tokens_per_chunk
        tokens = tokens.view(
            batch_size, num_chunks, self.tokens_per_chunk, self.llm_dim
        )
        if frame_mask is not None:
            token_mask = frame_mask.view(
                batch_size, num_chunks, self.tokens_per_chunk
            )
        else:
            token_mask = torch.ones(
                batch_size,
                num_chunks,
                self.tokens_per_chunk,
                dtype=torch.bool,
                device=pose.device,
            )
        chunk_mask = (
            chunk_mask
            if chunk_mask is not None
            else torch.ones(
                batch_size, num_chunks, dtype=torch.bool, device=pose.device
            )
        )
        token_mask = token_mask.to(torch.bool) & chunk_mask.unsqueeze(-1)
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, token_mask, chunk_mask
