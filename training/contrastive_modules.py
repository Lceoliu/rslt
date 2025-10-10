"""
Modules for contrastive learning, including projection heads and summarizers.
"""
import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """A simple projection head for contrastive learning, inspired by CLIP."""

    def __init__(
        self, in_dim: int, out_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected  # Residual connection
        x = self.layer_norm(x)
        return x


class SequenceSummarizer(nn.Module):
    """
    Summarizes a sequence of tokens into a single vector using a Transformer.
    It prepends a learnable summary token to the sequence, passes it through a
    Transformer encoder, and returns the output embedding of the summary token.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.summary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # A simple fixed positional embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): Input sequence of shape [B, L, E].
            mask (torch.Tensor, optional): Boolean tensor mask of shape [B, L],
                                         where True indicates a valid token.
                                         Defaults to None.

        Returns:
            torch.Tensor: A summary vector for each sequence in the batch, shape [B, E].
        """
        B, L, E = tokens.shape

        # Prepend summary token
        summary_tokens = self.summary_token.expand(B, -1, -1)
        tokens = torch.cat((summary_tokens, tokens), dim=1)  # [B, L+1, E]

        # Add positional embedding
        if (L + 1) > self.positional_embedding.shape[1]:
            raise ValueError(
                f"Input sequence length {L+1} exceeds max_seq_len {self.positional_embedding.shape[1]}"
            )
        tokens = tokens + self.positional_embedding[:, : (L + 1), :]

        # Create key padding mask for the transformer
        padding_mask = None
        if mask is not None:
            # TransformerEncoder expects `True` for positions to be IGNORED.
            # Our input `mask` has `True` for positions to be KEPT. So we invert it.
            summary_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat((summary_mask, mask), dim=1) # [B, L+1]
            padding_mask = ~mask # Invert mask

        # Pass through transformer
        output = self.transformer_encoder(
            tokens, src_key_padding_mask=padding_mask
        )  # [B, L+1, E]

        # Return the embedding of the summary token
        summary_embedding = output[:, 0, :]  # [B, E]
        return summary_embedding
