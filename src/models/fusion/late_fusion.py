import torch
import torch.nn as nn


class LateFusionBaseline(nn.Module):
    def __init__(self, audio_dim=88, text_dim=768, hidden_dim=128, n_structured=5):
        super().__init__()

        self.audio_tower = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        self.text_tower = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        self.structured_tower = nn.Sequential(
            nn.Linear(n_structured, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.combine = nn.Linear(3, 1)

    def forward(self, audio, text, structured, **kwargs):
        audio_pooled = audio.mean(1) if audio.dim() == 3 else audio
        text_pooled  = text.mean(1) if text.dim() == 3 else text

        audio_pred = self.audio_tower(audio_pooled)
        text_pred  = self.text_tower(text_pooled)
        struct_pred = self.structured_tower(structured)

        combined = torch.cat([audio_pred, text_pred, struct_pred], dim=-1)
        return self.combine(combined).squeeze(-1)
