import torch
import torch.nn as nn


class EarlyFusionBaseline(nn.Module):
    def __init__(self, audio_dim=88, text_dim=768, hidden_dim=256, n_structured=5):
        super().__init__()
        self.structured_proj = nn.Linear(n_structured, hidden_dim // 4)
        self.encoder = nn.Sequential(
            nn.Linear(audio_dim + text_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, audio, text, structured, **kwargs):
        audio_pooled = audio.mean(1) if audio.dim() == 3 else audio
        text_pooled  = text.mean(1) if text.dim() == 3 else text
        struct_proj  = self.structured_proj(structured)
        x = torch.cat([audio_pooled, text_pooled, struct_proj], dim=-1)
        return self.encoder(x).squeeze(-1)
