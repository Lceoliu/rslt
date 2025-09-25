"""Uni-GCN per-part backbone integration.

Wraps the Uni-GCN ST-GCN blocks to expose the same interface as the legacy
AAGCN/STGCN backbones used by :mod:`model.parts_gcn`.

"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from uni_GCN.stgcn_block import get_stgcn_chain


class UniGCNPartBackbone(nn.Module):
    """Per-part encoder built on Uni-GCN ST-GCN chains."""

    def __init__(
        self,
        in_channels: int,
        adjacency: torch.Tensor,
        *,
        proj_dim: int = 64,
        embed_dim: int = 256,
        adaptive: bool = True,
        temporal_kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        adjacency = adjacency.clone().detach().to(dtype=torch.float32)
        self.register_buffer("adjacency", adjacency, persistent=False)

        self.proj = nn.Linear(in_channels, proj_dim)

        spatial_kernel = (1, adjacency.size(0))
        temporal_kernel = (max(1, temporal_kernel_size), adjacency.size(0))

        spatial_chain, spatial_out_dim = get_stgcn_chain(
            proj_dim, "spatial", spatial_kernel, adjacency.clone(), adaptive=adaptive
        )
        temporal_chain, temporal_out_dim = get_stgcn_chain(
            spatial_out_dim, "temporal", temporal_kernel, adjacency.clone(), adaptive=adaptive
        )
        self.spatial_chain = spatial_chain
        self.temporal_chain = temporal_chain

        self.out_proj: nn.Module
        if temporal_out_dim != embed_dim:
            self.out_proj = nn.Linear(temporal_out_dim, embed_dim)
        else:
            self.out_proj = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embed_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_seq: bool = False,
    ) -> torch.Tensor:
        """Encode poses of shape [B, C, T, V].

        Args:
            x: Input pose tensor.
            mask: Optional binary mask [B, T] marking valid frames.
            return_seq: Whether to return per-frame features [B, T, D].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape [B,C,T,V], got {x.shape}")
        B, _, T, _ = x.shape
        dtype = self.proj.weight.dtype
        x = x.to(dtype)

        if mask is not None:
            mask = mask.to(dtype)
            mask_reshaped = mask.view(B, T, 1, 1)
        else:
            mask_reshaped = None

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        if mask_reshaped is not None:
            x = x * mask_reshaped
        x = self.proj(x)  # [B, T, V, proj_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, proj_dim, T, V]

        x = self.spatial_chain(x)
        x = self.temporal_chain(x)
        x = x.mean(dim=-1)  # [B, hidden, T]
        x = x.transpose(1, 2).contiguous()  # [B, T, hidden]
        x = self.out_proj(x)
        x = self.dropout(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        if return_seq:
            return x

        if mask is None:
            return x.mean(dim=1)

        weights = mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (x * weights).sum(dim=1) / denom
        return pooled




