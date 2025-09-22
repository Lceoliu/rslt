"""
Minimal ST-GCN backbone (Torch-only) for pose embeddings.
Reference idea: Yan et al., AAAI 2018.

Inputs: x [N, C, T, V], A [V, V] or [K, V, V]
Output: embedding [N, embed_dim]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(A: torch.Tensor, add_self: bool = True, eps: float = 1e-6) -> torch.Tensor:
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if add_self:
        K, V, _ = A.shape
        I = torch.eye(V, device=A.device, dtype=A.dtype).unsqueeze(0).expand(K, -1, -1)
        A = A + I
    deg = A.sum(-1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    A = A * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    return A


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.A = normalize_adjacency(A)
        self.K = self.A.shape[0]
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(self.K)]
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A.to(x.dtype)
        y = 0
        for k in range(self.K):
            z = self.conv[k](x)
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
        self.gcn = GraphConv(in_channels, out_channels, A)
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


class STGCNBackbone(nn.Module):
    def __init__(self, in_channels: int, A: torch.Tensor, embed_dim: int = 256):
        super().__init__()
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[-1])
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, stride=1),
            STGCNBlock(64, 64, A, stride=1),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, embed_dim, A, stride=2),
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_seq: bool = False) -> torch.Tensor:
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = x.to(self.data_bn.weight.dtype)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for blk in self.blocks:
            x = blk(x)
        To = x.size(2)
        if return_seq:
            if mask is not None:
                m = mask.to(x.dtype).unsqueeze(1).unsqueeze(-1)
                if m.size(2) != To:
                    m = F.interpolate(m, size=(To, 1), mode='nearest')
                x = x * m
            x = x.mean(dim=-1)  # [N, D, To]
            x = x.permute(0, 2, 1).contiguous()  # [N, To, D]
            return x
        else:
            if mask is None:
                x = F.avg_pool2d(x, x.shape[-2:])
                return x.flatten(1)
            else:
                m = mask.to(x.dtype).unsqueeze(1).unsqueeze(-1)  # [N,1,T_in,1]
                if m.size(2) != To:
                    m = F.interpolate(m, size=(To, 1), mode='nearest')
                x = (x * m).sum(dim=2) / (m.sum(dim=2).clamp_min(1e-6))  # [N, D, V]
                x = x.mean(dim=-1)  # [N, D]
                return x

