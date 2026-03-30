import torch
import torch.nn as nn


class VolatilityHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, n_structured=5,
                 use_structured=True):
        super().__init__()
        self.use_structured = use_structured

        total_dim = input_dim
        if use_structured:
            self.structured_proj = nn.Linear(n_structured, hidden_dim // 2)
            total_dim += hidden_dim // 2

        self.head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor,
                structured: torch.Tensor = None) -> torch.Tensor:
        if self.use_structured and structured is not None:
            s = self.structured_proj(structured)
            features = torch.cat([features, s], dim=-1)

        return self.head(features).squeeze(-1)
