from __future__ import annotations

import torch
import torch.nn as nn


class VisualAdapter(nn.Module):
    """Project visual embedding [B, D] to LLM hidden size as prefix tokens [B, P, E].

    Args:
        in_dim: visual embedding dim (D)
        llm_dim: LLM hidden size (E)
        num_prefix_tokens: number of prefix tokens (P), default 1
        hidden_dim: optional MLP hidden dim
        dropout: dropout prob
    """

    def __init__(
        self,
        in_dim: int,
        llm_dim: int,
        num_prefix_tokens: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.llm_dim = llm_dim
        self.num_prefix_tokens = int(num_prefix_tokens)
        if hidden_dim is None:
            hidden_dim = max(in_dim, llm_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, llm_dim * self.num_prefix_tokens),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        B, D = z.shape
        y = self.net(z)  # [B, P*E]
        y = y.view(B, self.num_prefix_tokens, self.llm_dim)
        return y

