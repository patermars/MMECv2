import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ..models.hierarchical_encoder import HierarchicalMultimodalEncoder, collate_calls
from ..models.fusion.early_fusion import EarlyFusionBaseline
from ..models.fusion.late_fusion import LateFusionBaseline
from ..evaluation.metrics import compute_metrics
from .trainer import Trainer


ABLATION_CONFIGS = {
    "EXP-00": {
        "text": False, "audio": False, "structured": True,
        "fusion": None, "notes": "Pure financial baseline"
    },
    "EXP-MAEC-INS": {
        "text": True, "audio": "maec_lld", "structured": False,
        "fusion": "early", "notes": "MAEC-inspired baseline"
    },
    "EXP-01": {
        "text": True, "audio": False, "structured": False,
        "fusion": None, "notes": "FinBERT text only"
    },
    "EXP-02": {
        "text": False, "audio": True, "structured": False,
        "fusion": None, "notes": "eGeMAPS audio only"
    },
    "EXP-03": {
        "text": False, "audio": False, "structured": False,
        "fusion": None, "notes": "LM dictionary only"
    },
    "EXP-04": {
        "text": True, "audio": False, "structured": True,
        "fusion": "early", "notes": "Text + structured"
    },
    "EXP-05": {
        "text": False, "audio": True, "structured": True,
        "fusion": "early", "notes": "Audio + structured"
    },
    "EXP-06": {
        "text": True, "audio": True, "structured": True,
        "fusion": "early", "notes": "All - early fusion"
    },
    "EXP-07": {
        "text": True, "audio": True, "structured": True,
        "fusion": "late", "notes": "All - late fusion"
    },
    "EXP-08": {
        "text": True, "audio": True, "structured": True,
        "fusion": "hierarchical", "notes": "FULL MODEL"
    },
    "EXP-09": {
        "text": True, "audio": True, "structured": True,
        "fusion": "best", "notes": "CEO utterances only",
        "filter": "ceo_only"
    },
    "EXP-10": {
        "text": True, "audio": True, "structured": True,
        "fusion": "best", "notes": "Q&A section only",
        "filter": "qa_only"
    },
    "EXP-11": {
        "text": True, "audio": True, "structured": True,
        "fusion": "best", "notes": "Guidance section only",
        "filter": "guidance_only"
    },
}


def build_model_for_experiment(exp_id: str, config: dict) -> torch.nn.Module:
    exp = ABLATION_CONFIGS[exp_id]
    mc = config.get("model", {})

    audio_dim = mc.get("audio_dim", 88)
    text_dim = mc.get("text_dim", 768)
    hidden_dim = mc.get("hidden_dim", 256)
    n_structured = mc.get("n_structured", 5)

    if not exp["audio"]:
        audio_dim = 0
    if not exp["text"]:
        text_dim = 0

    if exp["fusion"] == "hierarchical":
        return HierarchicalMultimodalEncoder(
            audio_dim=audio_dim or 88,
            text_dim=text_dim or 768,
            hidden_dim=hidden_dim,
            n_heads=mc.get("n_heads", 4),
            n_structured=n_structured,
        )
    elif exp["fusion"] == "late":
        return LateFusionBaseline(
            audio_dim=audio_dim or 88,
            text_dim=text_dim or 768,
            hidden_dim=hidden_dim // 2,
            n_structured=n_structured,
        )
    else:
        return EarlyFusionBaseline(
            audio_dim=audio_dim or 88,
            text_dim=text_dim or 768,
            hidden_dim=hidden_dim,
            n_structured=n_structured,
        )


class AblationRunner:
    def __init__(self, train_data, val_data, test_data, config: dict,
                 device: str = "cuda"):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.device = device
        self.results = {}

    def run_experiment(self, exp_id: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Running {exp_id}: {ABLATION_CONFIGS[exp_id]['notes']}")
        print(f"{'='*60}")

        model = build_model_for_experiment(exp_id, self.config)

        tc = self.config.get("training", {})
        batch_size = tc.get("batch_size", 16)

        collate_fn = collate_calls if ABLATION_CONFIGS[exp_id].get("fusion") == "hierarchical" else None

        train_loader = DataLoader(self.train_data, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size,
                                collate_fn=collate_fn,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=2, pin_memory=True)

        trainer = Trainer(model, train_loader, val_loader, self.config, self.device)
        train_result = trainer.train()

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                labels = batch.pop("labels")
                preds = model(**batch)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        test_metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
        test_metrics["exp_id"] = exp_id
        test_metrics["notes"] = ABLATION_CONFIGS[exp_id]["notes"]
        test_metrics["best_val_spearman"] = train_result["best_spearman"]

        self.results[exp_id] = test_metrics
        print(f"{exp_id} Test Spearman: {test_metrics['spearman_rho']:.4f} | "
              f"RMSE: {test_metrics['rmse']:.4f}")

        return test_metrics

    def run_all(self, experiment_ids: list = None) -> pd.DataFrame:
        if experiment_ids is None:
            experiment_ids = list(ABLATION_CONFIGS.keys())

        for exp_id in experiment_ids:
            self.run_experiment(exp_id)

        df = pd.DataFrame(self.results).T
        df = df.sort_values("spearman_rho", ascending=False)
        print(f"\n{'='*60}")
        print("ABLATION RESULTS")
        print(f"{'='*60}")
        print(df.to_string())

        return df
