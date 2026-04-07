import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
from pathlib import Path
from ..evaluation.metrics import compute_metrics
from .losses import CombinedLoss


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: dict,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tc = config.get("training", {})
        self.max_epochs = tc.get("max_epochs", 100)
        self.patience = tc.get("patience", 10)
        self.gradient_clip = tc.get("gradient_clip", 1.0)
        self.finbert_unfreeze_epoch = tc.get("finbert_unfreeze_epoch", 5)
        self.warmup_pct = tc.get("warmup_pct", 0.05)

        self.criterion = CombinedLoss(
            mse_weight=tc.get("loss_mse_weight", 0.7),
            rank_weight=tc.get("loss_rank_weight", 0.3),
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc.get("lr", 2e-4),
            weight_decay=tc.get("weight_decay", 1e-2),
        )

        total_steps = self.max_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=tc.get("lr", 2e-4),
            total_steps=total_steps,
            pct_start=self.warmup_pct,
            anneal_strategy="cos",
        )

        self.best_spearman = -1.0
        self.patience_counter = 0
        self.best_state = None

        self.wandb_enabled = os.environ.get("WANDB_ENABLED", "false").lower() == "true"
        if self.wandb_enabled:
            import wandb
            wandb.init(project="mmec-v2", config=config)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in self.train_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            labels = batch.pop("labels")
            preds = self.model(**batch)

            loss_dict = self.criterion(preds, labels)
            loss = loss_dict["total"]

            if not torch.isfinite(loss):
                self.scheduler.step()
                continue

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())

        metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            labels = batch.pop("labels")
            preds = self.model(**batch)

            loss_dict = self.criterion(preds, labels)
            total_loss += loss_dict["total"].item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def train(self) -> dict:
        for epoch in range(self.max_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(f"Epoch {epoch+1}/{self.max_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Spearman: {val_metrics['spearman_rho']:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.4f}")

            if self.wandb_enabled:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })

            val_rho = val_metrics["spearman_rho"]
            if not np.isfinite(val_rho):
                print(f"Early stopping at epoch {epoch+1} — NaN val metrics")
                break

            if val_rho > self.best_spearman:
                self.best_spearman = val_rho
                self.best_state = {k: v.cpu().clone() for k, v in
                                   self.model.state_dict().items()}
                self.patience_counter = 0
                self.save_checkpoint("data/processed/checkpoints/best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return {"best_spearman": self.best_spearman}

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_spearman": self.best_spearman,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_spearman = ckpt.get("best_spearman", -1.0)
