import torch
import torch.nn as nn


class HierarchicalMultimodalEncoder(nn.Module):
    """
    Cross-modal attention encoder with reduced capacity for small datasets.
    Round 1 architecture: 128 hidden, 1 Transformer layer, 0.4 dropout.
    """
    def __init__(self, audio_dim=88, text_dim=768, hidden_dim=128,
                 n_heads=2, n_structured=5, dropout=0.4, n_layers=1):
        super().__init__()

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.utterance_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
        )
        self.section_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.structured_proj = nn.Sequential(
            nn.Linear(n_structured, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.call_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(hidden_dim, 1)

    def encode_section(self, audio_feats: torch.Tensor, text_feats: torch.Tensor,
                        padding_mask: torch.Tensor = None) -> torch.Tensor:
        a = self.audio_proj(audio_feats)
        t = self.text_proj(text_feats)

        fused, _ = self.utterance_cross_attn(
            query=t, key=a, value=a,
            key_padding_mask=padding_mask
        )

        section_enc = self.section_transformer(
            fused, src_key_padding_mask=padding_mask
        )

        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            section_vec = (section_enc * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            section_vec = section_enc.mean(dim=1)

        return section_vec

    def forward(
        self,
        remarks_audio:  torch.Tensor,
        remarks_text:   torch.Tensor,
        remarks_mask:   torch.Tensor,
        qa_audio:       torch.Tensor,
        qa_text:        torch.Tensor,
        qa_mask:        torch.Tensor,
        structured:     torch.Tensor,
    ) -> torch.Tensor:

        remarks_vec = self.encode_section(remarks_audio, remarks_text, remarks_mask)
        qa_vec      = self.encode_section(qa_audio,      qa_text,      qa_mask)
        struct_vec  = self.structured_proj(structured)

        call_vec   = self.call_fusion(
            torch.cat([remarks_vec, qa_vec, struct_vec], dim=-1)
        )
        return self.output_head(call_vec).squeeze(-1)


def collate_calls(batch: list[dict]) -> dict:
    import numpy as np

    def pad_section(key):
        seqs    = [np.asarray(b[key], dtype=np.float32) for b in batch]
        max_len = max(s.shape[0] for s in seqs)
        padded  = np.zeros((len(seqs), max_len, seqs[0].shape[1]), dtype=np.float32)
        mask    = np.ones((len(seqs), max_len), dtype=bool)
        for i, s in enumerate(seqs):
            padded[i, :s.shape[0]] = s
            mask[i, :s.shape[0]]   = False
        return torch.from_numpy(padded), torch.from_numpy(mask)

    remarks_audio, remarks_mask = pad_section("remarks_audio")
    remarks_text,  _            = pad_section("remarks_text")
    qa_audio,      qa_mask      = pad_section("qa_audio")
    qa_text,       _            = pad_section("qa_text")

    return {
        "remarks_audio": remarks_audio,
        "remarks_text":  remarks_text,
        "remarks_mask":  remarks_mask,
        "qa_audio":      qa_audio,
        "qa_text":       qa_text,
        "qa_mask":       qa_mask,
        "structured":    torch.from_numpy(np.array([b["structured"] for b in batch], dtype=np.float32)),
        "labels":        torch.from_numpy(np.array([b["label"]      for b in batch], dtype=np.float32)),
    }
