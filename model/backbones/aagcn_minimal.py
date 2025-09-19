"""
Minimal AAGCN-style backbone (Torch-only) for pose embeddings.

Inputs:
- x: Tensor[N, C, T, V] where C is channels (e.g., x,y,conf),
  V is number of kept joints after discarding.
- A: adjacency as Tensor[K, V, V] or [V, V] (we expand to K=1).

Outputs:
- embedding: Tensor[N, embed_dim] via global average pooling.

Notes:
- No mmcv/mmengine dependency. Keep it simple and portable.
- Use NormalizeProcessor.gen_adjacency_matrix(normalize=True) to get A.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(A: torch.Tensor, add_self: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric normalize A. Accepts [V,V] or [K,V,V]."""
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if add_self:
        K, V, _ = A.shape
        I = torch.eye(V, device=A.device, dtype=A.dtype).unsqueeze(0).expand(K, -1, -1)
        A = A + I
    # D^{-1/2} A D^{-1/2}
    deg = A.sum(-1)  # [K, V]
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    A_norm = A * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    return A_norm


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, adaptive: bool = True):
        super().__init__()
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.K, self.V, _ = A.shape
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(self.K)]
        )
        if adaptive:
            self.PA = nn.Parameter(A.clone())
            self.register_buffer("A_base", A)  # for reference
        else:
            self.register_buffer("A_base", A)
            self.PA = None
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V]
        A = self.PA if self.PA is not None else self.A_base
        A = normalize_adjacency(A)
        y = 0
        for k in range(self.K):
            z = self.conv[k](x)  # [N, Cout, T, V]
            y = y + torch.einsum("nctv,vw->nctw", z, A[k])
        y = self.bn(y)
        return y


class TemporalConv(nn.Module):
    def __init__(self, channels: int, stride: int = 1, kernel_size: int = 9, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                              stride=(stride, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.drop(x)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A, adaptive=True)
        self.tcn = TemporalConv(out_channels, stride=stride)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gcn(x)
        y = self.tcn(y)
        y = y + self.residual(x)
        y = self.relu(y)
        return y


class AAGCNBackbone(nn.Module):
    """Compact AAGCN-like backbone producing a global embedding.

    Args:
        in_channels: input channel count (e.g., 2 for xy or 3 for xy+conf)
        A: adjacency tensor [V,V] or [K,V,V]
        embed_dim: output embedding dimension (default 256)
    """

    def __init__(self, in_channels: int, A: torch.Tensor, embed_dim: int = 256):
        super().__init__()
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.A = A

        self.normalize_dim = in_channels * A.shape[-1]
        self.data_bn = nn.BatchNorm1d(self.normalize_dim)

        self.st_blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, stride=1),
            STGCNBlock(64, 64, A, stride=1),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, embed_dim, A, stride=2),
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [N, C, T, V]; mask: [N, T] with 1 for valid frames
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)  # [N, VC, T]
        x = x.to(self.data_bn.weight.dtype)  # ensure same dtype
        assert (
            x.shape[1] == self.normalize_dim
        ), f"Expected {self.normalize_dim} channels, got {x.shape[1]}"
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for blk in self.st_blocks:
            x = blk(x)

        # Mask-aware spatiotemporal average pooling
        if mask is None:
            x = F.avg_pool2d(x, x.shape[-2:])  # [N, D, 1, 1]
            return x.flatten(1)
        else:
            m = mask.to(x.dtype).unsqueeze(1).unsqueeze(-1)  # [N,1,T_in,1]
            # Downsample/upsample mask to match temporal dim after strides
            To = x.size(2)
            if m.size(2) != To:
                m = F.interpolate(m, size=(To, 1), mode='nearest')
            x = (x * m).sum(dim=2) / (m.sum(dim=2).clamp_min(1e-6))  # [N, D, V]
            x = x.mean(dim=-1)  # [N, D]
            return x


def build_adjacency_from_numpy(A_np) -> torch.Tensor:
    """Helper to convert np.ndarray adjacency to torch.Tensor[float32]."""
    if hasattr(A_np, "astype"):
        import numpy as np  # local import

        if isinstance(A_np, dict):
            A_np = A_np.get("fullbody")
        A_np = A_np.astype("float32")
    return torch.from_numpy(A_np)
