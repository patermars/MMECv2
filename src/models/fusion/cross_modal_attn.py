import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attended, _ = self.attn(
            query=query, key=key_value, value=key_value,
            key_padding_mask=key_padding_mask
        )
        return self.norm(query + self.dropout(attended))
