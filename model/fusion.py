from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConcatMLPFusion(nn.Module):
    def __init__(self, in_dims: List[int], out_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_dim = sum(in_dims)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(xs, dim=-1)
        net_dtype = self.net[0].weight.dtype  # query live dtype (bf16/fp32)
        x = x.to(net_dtype)
        return self.net(x)


class AttentionFusion(nn.Module):
    def __init__(self, in_dims: List[int], out_dim: int, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.projs = nn.ModuleList([nn.Linear(d, d_model) for d in in_dims])
        self.pos = nn.Parameter(torch.zeros(1, len(in_dims), d_model))
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # xs: list of [N, D_p], N = B*t'
        net_dtype = self.projs[0].weight.dtype  # query live dtype
        xs = [x.to(net_dtype) for x in xs]
        feats = [proj(x) for proj, x in zip(self.projs, xs)]  # [N, d_model] each
        x = torch.stack(feats, dim=1)  # [N, #num_parts, d_model]
        x = x + self.pos  # simple positional enc for parts
        y, _ = self.mha(x, x, x)
        y = y.mean(dim=1)  # pool over parts
        return self.out(y)
