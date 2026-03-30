import torch
import torch.nn as nn


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = preds.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=preds.device)

        pred_diff   = preds.unsqueeze(0) - preds.unsqueeze(1)
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)

        sign = torch.sign(target_diff)
        loss = torch.clamp(self.margin - sign * pred_diff, min=0.0)

        mask = torch.triu(torch.ones(n, n, device=preds.device), diagonal=1).bool()
        return loss[mask].mean()


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.7, rank_weight: float = 0.3,
                 rank_margin: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.mse_loss = nn.MSELoss()
        self.rank_loss = PairwiseRankingLoss(margin=rank_margin)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        mse  = self.mse_loss(preds, targets)
        rank = self.rank_loss(preds, targets)
        total = self.mse_weight * mse + self.rank_weight * rank

        return {
            "total": total,
            "mse":   mse,
            "rank":  rank,
        }
