import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(self, acoustic_dim=29, text_dim=768, wav_dim=768,
                 hidden_dim=256, n_heads=4, dropout=0.2, n_structured=5,
                 n_targets=1):
        super().__init__()
        self.n_targets = n_targets
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.wav_proj = nn.Sequential(
            nn.Linear(wav_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.cross_attn_text_audio = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=dropout
        )
        self.cross_attn_text_wav = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, acoustic, text_emb, wav_emb, mask=None):
        a = self.acoustic_proj(acoustic)
        t = self.text_proj(text_emb)
        w = self.wav_proj(wav_emb)

        fused_ta, _ = self.cross_attn_text_audio(
            query=t, key=a, value=a, key_padding_mask=mask
        )
        fused_tw, _ = self.cross_attn_text_wav(
            query=t, key=w, value=w, key_padding_mask=mask
        )

        fused = fused_ta + fused_tw + t

        encoded = self.seq_encoder(fused, src_key_padding_mask=mask)

        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        out = self.head(pooled)
        return out.squeeze(-1) if self.n_targets == 1 else out


class AudioOnlyBaseline(nn.Module):
    def __init__(self, acoustic_dim=29, hidden_dim=128, dropout=0.2, n_targets=1):
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, acoustic, text_emb=None, wav_emb=None, mask=None):
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            pooled = (acoustic * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = acoustic.mean(dim=1)
        out = self.net(pooled)
        return out.squeeze(-1) if self.n_targets == 1 else out


class TextOnlyBaseline(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=128, dropout=0.2, n_targets=1):
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, acoustic=None, text_emb=None, wav_emb=None, mask=None):
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            pooled = (text_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = text_emb.mean(dim=1)
        out = self.net(pooled)
        return out.squeeze(-1) if self.n_targets == 1 else out


class EarlyFusionBaseline(nn.Module):
    def __init__(self, acoustic_dim=29, text_dim=768, wav_dim=768,
                 hidden_dim=256, dropout=0.2, n_targets=1):
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(
            nn.Linear(acoustic_dim + text_dim + wav_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, acoustic, text_emb, wav_emb, mask=None):
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            a_pool = (acoustic * valid).sum(1) / valid.sum(1).clamp(min=1)
            t_pool = (text_emb * valid).sum(1) / valid.sum(1).clamp(min=1)
            w_pool = (wav_emb * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            a_pool = acoustic.mean(1)
            t_pool = text_emb.mean(1)
            w_pool = wav_emb.mean(1)
        cat = torch.cat([a_pool, t_pool, w_pool], dim=-1)
        out = self.net(cat)
        return out.squeeze(-1) if self.n_targets == 1 else out


def build_model(cfg, model_type="cross_modal"):
    m = cfg["model"]
    a_dim = cfg["audio"]["n_acoustic_features"]
    t_dim = cfg["text"]["embedding_dim"]
    w_dim = cfg["wav2vec"]["embedding_dim"]

    # Determine number of output targets
    multi_task = cfg["label"].get("multi_task", False)
    mt_targets = cfg["label"].get("multi_task_targets", None)
    n_targets = len(mt_targets) if (multi_task and mt_targets) else 1

    if model_type == "cross_modal":
        return CrossModalFusion(a_dim, t_dim, w_dim, m["hidden_dim"],
                                m["n_heads"], m["dropout"], m["n_structured"],
                                n_targets=n_targets)
    elif model_type == "audio_only":
        return AudioOnlyBaseline(a_dim, m["hidden_dim"], m["dropout"],
                                 n_targets=n_targets)
    elif model_type == "text_only":
        return TextOnlyBaseline(t_dim, m["hidden_dim"], m["dropout"],
                                n_targets=n_targets)
    elif model_type == "early_fusion":
        return EarlyFusionBaseline(a_dim, t_dim, w_dim, m["hidden_dim"],
                                   m["dropout"], n_targets=n_targets)
    else:
        raise ValueError(f"Unknown model: {model_type}")
