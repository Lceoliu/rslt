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
        # Per-token projection used when the input already carries P tokens
        self.token_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Accept [B, D] or [B, N, D] and return prefix embeds.

        - If input is [B, D] -> output [B, P, E]
        - If input is [B, N, D] -> output [B, N, P, E] (P often = 1). If P==1, returns [B, N, E].
        """
        if z.dim() == 2:
            B, D = z.shape
            y = self.net(z)  # [B, P*E]
            y = y.view(B, self.num_prefix_tokens, self.llm_dim)
            return y
        elif z.dim() == 3:
            B, N, D = z.shape
            z2 = z.reshape(B * N, D)
            y = self.net(z2).view(B, N, self.num_prefix_tokens, self.llm_dim)
            if self.num_prefix_tokens == 1:
                return y.squeeze(2)  # [B, N, E]
            return y  # [B, N, P, E]
        elif z.dim() == 4:
            # Input already carries P tokens: [B, N, P, D] -> [B, N, P, E]
            B, N, P, D = z.shape
            z2 = z.reshape(B * N * P, D)
            y = self.token_net(z2).view(B, N, P, self.llm_dim)
            return y
        else:
            raise ValueError(f"Unexpected z.dim()={z.dim()} in VisualAdapter.forward")
