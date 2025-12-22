#!/usr/bin/env python3
"""
trainer.py — Single entrypoint for:
  1) Training models (IsolationForest + Autoencoder + VAE + DeepSVDD) with time-based split
  2) Saving validation-normal score statistics and percentile thresholds (production policy)
  3) Fitting ensemble calibration/config ONCE on validation normals
  4) Config-driven evaluation and inference (never recompute thresholds on request batch)

Patched with:
  - Deep SVDD buffer registration (device safety)
  - AUPRC optimization support in Optuna
  - Native state_dict handling for SVDD center

Usage (recommended):

# 1) Train models + save artifacts + save score stats
python trainer.py train --data data/creditcard.csv --epochs 100 --threshold-percentile 99

# (Optional) Optuna HPO (Now defaults to AUPRC for better fraud detection)
python trainer.py train --data data/creditcard.csv --optimize --optimize-metric auprc \
  --optimize-ae --optimize-vae --optimize-svdd \
  --optimization-trials 30 --optimization-timeout 1800 --epochs 100 --threshold-percentile 99

# 2) Fit ensemble config on validation
python trainer.py fit-config --models-dir models/production \
  --val-data artifacts/X_val_scaled.npy --val-labels artifacts/y_val.npy \
  --save-config models/production/ensemble_config.json \
  --threshold-percentile 99 --optimize-weights --optimize-metric auprc

# 3) Evaluate on test
python trainer.py evaluate --models-dir models/production \
  --config models/production/ensemble_config.json \
  --data artifacts/X_test_scaled.npy --labels artifacts/y_test.npy

# 4) Predict on new batch
python trainer.py predict --models-dir models/production \
  --config models/production/ensemble_config.json \
  --data artifacts/X_some_batch_scaled.npy --out predictions.npy
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

# sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# optuna (optional)
try:
    import optuna
    from optuna import Trial
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trainer.log")],
)
logger = logging.getLogger(__name__)

# Apple Silicon support
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# =============================================================================
# Utilities
# =============================================================================

def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    return obj


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    try:
        return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))
    except Exception:
        return 0.0, 0.0


def zscore_fit(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores).astype(float)
    mu = float(np.mean(scores)) if len(scores) else 0.0
    sigma = float(np.std(scores)) if len(scores) else 1.0
    if sigma < 1e-12:
        sigma = 1.0
    return {"mu": mu, "sigma": sigma}


def zscore_apply(scores: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (np.asarray(scores).astype(float) - float(mu)) / float(sigma)


def percentile_threshold(scores_normal: np.ndarray, percentile: float) -> float:
    scores_normal = np.asarray(scores_normal).astype(float)
    if len(scores_normal) == 0:
        return 0.0
    return float(np.percentile(scores_normal, percentile))


def percentile_map(scores_normal: np.ndarray, percentiles: List[float]) -> Dict[str, float]:
    scores_normal = np.asarray(scores_normal).astype(float)
    out: Dict[str, float] = {}
    if len(scores_normal) == 0:
        for p in percentiles:
            out[str(p)] = 0.0
        return out
    for p in percentiles:
        out[str(p)] = float(np.percentile(scores_normal, p))
    return out


# =============================================================================
# Data Processing (time-based split)
# =============================================================================

class DataProcessor:
    def __init__(self, exclude_time: bool = True):
        self.exclude_time = exclude_time
        self.feature_names: Optional[List[str]] = None
        self.scaler: Optional[RobustScaler] = None

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Class" not in df.columns or "Time" not in df.columns:
            raise ValueError("Expected columns 'Time' and 'Class' in dataset.")
        logger.info(f"Loaded data: {df.shape} fraud_rate={df['Class'].mean():.4%}")
        return df

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        # Time-derived features
        df["Hour"] = (df["Time"] / 3600.0) % 24.0
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)

        # Amount features
        df["Amount_log"] = np.log1p(df["Amount"])
        df["Amount_sqrt"] = np.sqrt(df["Amount"].clip(lower=0))
        df["Is_High_Amount"] = (df["Amount"] > df["Amount"].quantile(0.90)).astype(int)
        df["Is_Zero_Amount"] = (df["Amount"] == 0).astype(int)

        # PCA aggregations
        pca_cols = [c for c in df.columns if c.startswith("V")]
        df["V_mean"] = df[pca_cols].mean(axis=1)
        df["V_std"] = df[pca_cols].std(axis=1)
        df["V_max"] = df[pca_cols].max(axis=1)
        df["V_min"] = df[pca_cols].min(axis=1)
        df["V_range"] = df["V_max"] - df["V_min"]

        df["Amount_Hour_interaction"] = df["Amount"] * df["Hour"]

        # Features
        drop_cols = ["Class"]
        if self.exclude_time:
            drop_cols.append("Time")
        self.feature_names = [c for c in df.columns if c not in drop_cols]
        return df

    def split_time_based(self, df: pd.DataFrame, train_size: float, val_size: float, test_size: float) -> Tuple[np.ndarray, ...]:
        s = float(train_size) + float(val_size) + float(test_size)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"train/val/test sizes must sum to 1.0 (got {s})")

        df = df.sort_values("Time").reset_index(drop=True)
        X = df[self.feature_names].values
        y = df["Class"].values.astype(int)

        n = len(df)
        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"Split time-based: train={len(y_train)} (fraud={y_train.sum()}) "
                    f"val={len(y_val)} (fraud={y_val.sum()}) test={len(y_test)} (fraud={y_test.sum()})")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform_scaler(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # fit on normal train only
        normal = X_train[y_train == 0]
        self.scaler = RobustScaler()
        self.scaler.fit(normal)
        return self.scaler.transform(X_train), self.scaler.transform(X_val), self.scaler.transform(X_test)

    def save_artifacts(self, artifacts_dir: Path, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test) -> None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifacts_dir / "X_train_scaled.npy", X_train_s)
        np.save(artifacts_dir / "X_val_scaled.npy", X_val_s)
        np.save(artifacts_dir / "X_test_scaled.npy", X_test_s)
        np.save(artifacts_dir / "y_train.npy", y_train)
        np.save(artifacts_dir / "y_val.npy", y_val)
        np.save(artifacts_dir / "y_test.npy", y_test)

        with open(artifacts_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        meta = {
            "created_at": datetime.now().isoformat(),
            "split_policy": "time_based",
            "exclude_time": self.exclude_time,
            "n_features": int(len(self.feature_names)),
            "feature_names": self.feature_names,
            "fraud_rate_train": float(np.mean(y_train)),
            "fraud_rate_val": float(np.mean(y_val)),
            "fraud_rate_test": float(np.mean(y_test)),
        }
        with open(artifacts_dir / "meta.json", "w") as f:
            json.dump(convert_to_json_serializable(meta), f, indent=2)

        logger.info(f"Saved artifacts to {artifacts_dir}")


# =============================================================================
# Datasets
# =============================================================================

class FraudDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# =============================================================================
# Models
# =============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32,
                 hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.1)

        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), act]
            if dropout_rate > 0:
                enc_layers += [nn.Dropout(dropout_rate)]
            prev = h
        enc_layers += [nn.Linear(prev, encoding_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = encoding_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), act]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16,
                 hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.1)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act]
            if dropout_rate > 0:
                layers += [nn.Dropout(dropout_rate)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), act]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-20, max=2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)
        return xhat, mu, logvar


class DeepSVDD(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 rep_dim: int = 16, activation: str = "relu", dropout_rate: float = 0.0):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.1)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act]
            if dropout_rate > 0:
                layers += [nn.Dropout(dropout_rate)]
            prev = h
        layers += [nn.Linear(prev, rep_dim)]
        self.net = nn.Sequential(*layers)

        # PATCH: Use register_buffer so center_c moves to GPU/MPS automatically with .to(device)
        self.register_buffer("center_c", torch.zeros(rep_dim))

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Training helpers
# =============================================================================

@torch.no_grad()
def init_svdd_center(model: DeepSVDD, loader: DataLoader, device: torch.device, eps: float = 1e-3) -> None:
    model.eval()
    reps = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        z = model(x.to(device))
        reps.append(z.detach().cpu())
    reps = torch.cat(reps, dim=0)
    c = reps.mean(dim=0)
    # avoid exactly-0 dims
    mask = (c.abs() < eps)
    c[mask] = eps * c[mask].sign().clamp(min=1)
    
    # PATCH: Update the buffer in place
    model.center_c.data = c.to(device)


class ModelTrainer:
    def __init__(self, model: nn.Module, model_type: str, device: torch.device,
                 lr: float = 1e-3, batch_size: int = 256,
                 epochs: int = 100, early_stopping: int = 20):
        self.model = model.to(device)
        self.model_type = model_type  # autoencoder | vae | deep_svdd
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)

        self.best_metric_value = -1.0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        # Note: center_c is now inside the model state_dict for SVDD

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        nb = 0

        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)

            self.optimizer.zero_grad()

            if self.model_type == "vae":
                xhat, mu, logvar = self.model(x)
                recon = F.mse_loss(xhat, x, reduction="sum")
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().clamp(max=1e10))
                loss = (recon + kld) / x.size(0)
            elif self.model_type == "deep_svdd":
                z = self.model(x)
                # PATCH: Access buffer directly
                loss = torch.mean(torch.sum((z - self.model.center_c) ** 2, dim=1))
            else:
                xhat = self.model(x)
                loss = F.mse_loss(xhat, x)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total += float(loss.item())
            nb += 1

            if self.device.type == "mps" and nb % 50 == 0:
                torch.mps.empty_cache()

        return total / max(nb, 1)

    @torch.no_grad()
    def score(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
            if self.model_type == "vae":
                xhat, _, _ = self.model(xb)
                sc = F.mse_loss(xhat, xb, reduction="none").mean(dim=1)
            elif self.model_type == "deep_svdd":
                z = self.model(xb)
                sc = torch.sum((z - self.model.center_c) ** 2, dim=1)
            else:
                xhat = self.model(xb)
                sc = F.mse_loss(xhat, xb, reduction="none").mean(dim=1)
            out.append(sc.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    def train(self, X_train_normal: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, optimize_metric: str = "auroc") -> None:
        train_loader = DataLoader(FraudDataset(X_train_normal), batch_size=self.batch_size, shuffle=True, num_workers=0)

        if self.model_type == "deep_svdd":
            init_svdd_center(self.model, train_loader, self.device)

        val_loader = DataLoader(FraudDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False, num_workers=0)

        patience = 0
        best_val = -1.0

        for epoch in range(self.epochs):
            tr_loss = self._train_epoch(train_loader)
            self.scheduler.step(tr_loss)

            # validate 
            all_scores = []
            all_labels = []
            self.model.eval()
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                if self.model_type == "vae":
                    xhat, _, _ = self.model(xb)
                    sc = F.mse_loss(xhat, xb, reduction="none").mean(dim=1)
                elif self.model_type == "deep_svdd":
                    z = self.model(xb)
                    sc = torch.sum((z - self.model.center_c) ** 2, dim=1)
                else:
                    xhat = self.model(xb)
                    sc = F.mse_loss(xhat, xb, reduction="none").mean(dim=1)
                all_scores.append(sc.detach().cpu().numpy())
                all_labels.append(yb.numpy())

            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels).astype(int)

            auroc, auprc = safe_auc(labels, scores)
            
            # PATCH: Choose metric based on argument
            current_metric = auprc if optimize_metric == "auprc" else auroc
            metric_name = optimize_metric.upper()

            if current_metric > best_val:
                best_val = current_metric
                self.best_metric_value = current_metric
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(f"[{self.model_type}] epoch {epoch+1}/{self.epochs} "
                            f"loss={tr_loss:.6f} {metric_name}={current_metric:.4f} (Best: {best_val:.4f})")

            if patience >= self.early_stopping:
                logger.info(f"[{self.model_type}] early stopping at epoch {epoch+1}")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


# =============================================================================
# Training: IsolationForest + AE + VAE + DeepSVDD + score_stats.json
# =============================================================================

def train_isolation_forest(X_train_scaled: np.ndarray, y_train: np.ndarray, seed: int = 42) -> IsolationForest:
    X_norm = X_train_scaled[y_train == 0]
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_norm)
    return model


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.decision_function(X)


def compute_score_stats_and_save(
    artifacts_dir: Path,
    version: str,
    feature_names: List[str],
    threshold_percentile: float,
    save_percentiles: List[float],
    y_val: np.ndarray,
    scores_val_raw: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    y_val = np.asarray(y_val).astype(int)
    normal_mask = (y_val == 0)

    out: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "version": version,
        "split_policy": "time_based",
        "threshold_policy": "percentile_on_val_normals",
        "default_threshold_percentile": float(threshold_percentile),
        "percentiles_saved": [float(p) for p in save_percentiles],
        "n_val": int(len(y_val)),
        "n_val_normals": int(normal_mask.sum()),
        "feature_names": feature_names,
        "models": {},
    }

    for name, s_val in scores_val_raw.items():
        s_norm = np.asarray(s_val)[normal_mask]
        raw_stats = zscore_fit(s_norm)
        s_norm_z = zscore_apply(s_norm, raw_stats["mu"], raw_stats["sigma"])

        out["models"][name] = {
            "score_direction": "higher_more_anomalous",
            "raw": {
                "mu": raw_stats["mu"],
                "sigma": raw_stats["sigma"],
                "percentiles": percentile_map(s_norm, save_percentiles),
                "threshold_percentile": {str(float(threshold_percentile)): percentile_threshold(s_norm, threshold_percentile)},
            },
            "zscore": {
                "raw_mu": raw_stats["mu"],
                "raw_sigma": raw_stats["sigma"],
                "percentiles": percentile_map(s_norm_z, save_percentiles),
                "threshold_percentile": {str(float(threshold_percentile)): percentile_threshold(s_norm_z, threshold_percentile)},
            },
        }

    path = artifacts_dir / "score_stats.json"
    with open(path, "w") as f:
        json.dump(convert_to_json_serializable(out), f, indent=2)
    logger.info(f"Saved score stats to {path}")
    return out


def optuna_optimize(
    model_kind: str,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int,
    trials: int,
    timeout: int,
    metric: str = "auroc"
) -> Optional[Dict[str, Any]]:
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available; skipping optimization.")
        return None

    logger.info(f"Starting Optuna for {model_kind} optimizing {metric.upper()}...")

    def objective(trial: Trial) -> float:
        hidden_1 = trial.suggest_int("hidden_1", 64, 256, step=32)
        hidden_2 = trial.suggest_int("hidden_2", 32, 128, step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        input_dim = X_train_normal.shape[1]
        hidden_dims = [hidden_1, hidden_2]

        if model_kind == "autoencoder":
            encoding_dim = trial.suggest_int("encoding_dim", 8, 64, step=8)
            model = Autoencoder(input_dim, encoding_dim, hidden_dims, dropout, "relu")
            trainer = ModelTrainer(model, "autoencoder", device, lr=lr, batch_size=batch_size, epochs=20, early_stopping=5)

        elif model_kind == "vae":
            latent_dim = trial.suggest_int("latent_dim", 8, 48, step=8)
            model = VAE(input_dim, latent_dim, hidden_dims, dropout, "relu")
            trainer = ModelTrainer(model, "vae", device, lr=lr, batch_size=batch_size, epochs=20, early_stopping=5)

        elif model_kind == "deep_svdd":
            rep_dim = trial.suggest_int("rep_dim", 8, 48, step=8)
            model = DeepSVDD(input_dim, hidden_dims, rep_dim, "relu", dropout)
            trainer = ModelTrainer(model, "deep_svdd", device, lr=lr, batch_size=batch_size, epochs=20, early_stopping=5)

        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        trainer.train(X_train_normal, X_val, y_val, optimize_metric=metric)
        return float(trainer.best_metric_value)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=trials, timeout=timeout)
    logger.info(f"Optuna best for {model_kind}: {metric.upper()}={study.best_value:.4f} params={study.best_params}")
    return study.best_params


def cmd_train(args: argparse.Namespace) -> None:
    data_path = args.data
    output_dir = Path(args.output)
    artifacts_dir = Path(args.artifacts)
    prod_dir = output_dir / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # data
    dp = DataProcessor(exclude_time=not args.include_time)
    df = dp.load(data_path)
    df = dp.engineer(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dp.split_time_based(
        df, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size
    )
    X_train_s, X_val_s, X_test_s = dp.fit_transform_scaler(X_train, y_train, X_val, X_test)
    dp.save_artifacts(artifacts_dir, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)

    # train normals
    X_train_normal = X_train_s[y_train == 0]

    # baseline iso
    logger.info("Training IsolationForest (normal train only)...")
    iso = train_isolation_forest(X_train_s, y_train, seed=args.seed)

    # optuna per model
    best_ae = best_vae = best_svdd = None
    if args.optimize:
        if args.optimize_ae or (not args.optimize_vae and not args.optimize_svdd and not args.optimize_ae):
            best_ae = optuna_optimize("autoencoder", X_train_normal, X_val_s, y_val, device, args.seed, args.optimization_trials, args.optimization_timeout, metric=args.optimize_metric)
        if args.optimize_vae:
            best_vae = optuna_optimize("vae", X_train_normal, X_val_s, y_val, device, args.seed, args.optimization_trials, args.optimization_timeout, metric=args.optimize_metric)
        if args.optimize_svdd:
            best_svdd = optuna_optimize("deep_svdd", X_train_normal, X_val_s, y_val, device, args.seed, args.optimization_trials, args.optimization_timeout, metric=args.optimize_metric)

    # build/train models
    n_features = X_train_s.shape[1]

    ae_params = {"encoding_dim": 32, "hidden_dims": [128, 64], "dropout_rate": 0.1, "activation": "relu"}
    ae_lr = args.lr
    ae_bs = args.batch_size
    if best_ae:
        ae_params["hidden_dims"] = [best_ae["hidden_1"], best_ae["hidden_2"]]
        ae_params["dropout_rate"] = float(best_ae["dropout"])
        ae_params["encoding_dim"] = int(best_ae["encoding_dim"])
        ae_lr = float(best_ae["lr"])
        ae_bs = int(best_ae["batch_size"])

    vae_params = {"latent_dim": 16, "hidden_dims": [128, 64], "dropout_rate": 0.1, "activation": "relu"}
    vae_lr = args.lr
    vae_bs = args.batch_size
    if best_vae:
        vae_params["hidden_dims"] = [best_vae["hidden_1"], best_vae["hidden_2"]]
        vae_params["dropout_rate"] = float(best_vae["dropout"])
        vae_params["latent_dim"] = int(best_vae["latent_dim"])
        vae_lr = float(best_vae["lr"])
        vae_bs = int(best_vae["batch_size"])

    svdd_params = {"rep_dim": 16, "hidden_dims": [128, 64], "dropout_rate": 0.0, "activation": "relu"}
    svdd_lr = args.lr
    svdd_bs = args.batch_size
    if best_svdd:
        svdd_params["hidden_dims"] = [best_svdd["hidden_1"], best_svdd["hidden_2"]]
        svdd_params["dropout_rate"] = float(best_svdd["dropout"])
        svdd_params["rep_dim"] = int(best_svdd["rep_dim"])
        svdd_lr = float(best_svdd["lr"])
        svdd_bs = int(best_svdd["batch_size"])

    logger.info("Training Autoencoder (normal train only)...")
    ae = Autoencoder(n_features, **ae_params)
    ae_tr = ModelTrainer(ae, "autoencoder", device, lr=ae_lr, batch_size=ae_bs, epochs=args.epochs, early_stopping=args.early_stopping)
    train_metric = args.optimize_metric
    ae_tr.train(X_train_normal, X_val_s, y_val, optimize_metric=train_metric)

    logger.info("Training VAE (normal train only)...")
    vae = VAE(n_features, **vae_params)
    vae_tr = ModelTrainer(vae, "vae", device, lr=vae_lr, batch_size=vae_bs, epochs=args.epochs, early_stopping=args.early_stopping)
    vae_tr.train(X_train_normal, X_val_s, y_val, optimize_metric=train_metric)

    logger.info("Training DeepSVDD (normal train only)...")
    svdd = DeepSVDD(n_features, hidden_dims=svdd_params["hidden_dims"], rep_dim=svdd_params["rep_dim"],
                    activation=svdd_params["activation"], dropout_rate=svdd_params["dropout_rate"])
    svdd_tr = ModelTrainer(svdd, "deep_svdd", device, lr=svdd_lr, batch_size=svdd_bs, epochs=args.epochs, early_stopping=args.early_stopping)
    svdd_tr.train(X_train_normal, X_val_s, y_val, optimize_metric=train_metric)

    # threshold-free TEST evaluation
    iso_scores_test = score_isolation_forest(iso, X_test_s)
    ae_scores_test = ae_tr.score(X_test_s)
    vae_scores_test = vae_tr.score(X_test_s)
    svdd_scores_test = svdd_tr.score(X_test_s)

    results = pd.DataFrame([
        {"Model": "IsolationForest", "AUROC": safe_auc(y_test, iso_scores_test)[0], "AUPRC": safe_auc(y_test, iso_scores_test)[1]},
        {"Model": "Autoencoder", "AUROC": safe_auc(y_test, ae_scores_test)[0], "AUPRC": safe_auc(y_test, ae_scores_test)[1]},
        {"Model": "VAE", "AUROC": safe_auc(y_test, vae_scores_test)[0], "AUPRC": safe_auc(y_test, vae_scores_test)[1]},
        {"Model": "DeepSVDD", "AUROC": safe_auc(y_test, svdd_scores_test)[0], "AUPRC": safe_auc(y_test, svdd_scores_test)[1]},
    ]).sort_values("AUPRC", ascending=False).reset_index(drop=True)

    # Save models
    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(prod_dir / f"isolation_forest_{version}.pkl", "wb") as f:
        pickle.dump(iso, f)

    torch.save({
        "model_type": "autoencoder",
        "model_state_dict": ae.state_dict(),
        "params": convert_to_json_serializable(ae_params),
        "n_features": int(n_features),
        "feature_names": dp.feature_names,
        "best_val_metric": float(ae_tr.best_metric_value),
        "version": version,
    }, prod_dir / f"autoencoder_{version}.pth")

    torch.save({
        "model_type": "vae",
        "model_state_dict": vae.state_dict(),
        "params": convert_to_json_serializable(vae_params),
        "n_features": int(n_features),
        "feature_names": dp.feature_names,
        "best_val_metric": float(vae_tr.best_metric_value),
        "version": version,
    }, prod_dir / f"vae_{version}.pth")

    # PATCH: SVDD now saves center automatically in state_dict, but we keep explicit key for backward compat
    torch.save({
        "model_type": "deep_svdd",
        "model_state_dict": svdd.state_dict(),
        "params": convert_to_json_serializable(svdd_params),
        "n_features": int(n_features),
        "feature_names": dp.feature_names,
        "center_c": svdd.center_c.detach().cpu().numpy(),
        "best_val_metric": float(svdd_tr.best_metric_value),
        "version": version,
    }, prod_dir / f"deep_svdd_{version}.pth")

    # Save score stats from VAL normals
    scores_val_raw = {
        "isolation_forest": score_isolation_forest(iso, X_val_s),
        "autoencoder": ae_tr.score(X_val_s),
        "vae": vae_tr.score(X_val_s),
        "deep_svdd": svdd_tr.score(X_val_s),
    }
    save_percentiles = [95.0, 97.5, 99.0, 99.5, 99.9]
    compute_score_stats_and_save(
        artifacts_dir=artifacts_dir,
        version=version,
        feature_names=dp.feature_names,
        threshold_percentile=args.threshold_percentile,
        save_percentiles=save_percentiles,
        y_val=y_val,
        scores_val_raw=scores_val_raw,
    )

    # Save run summary
    run_summary = {
        "created_at": datetime.now().isoformat(),
        "version": version,
        "device": str(device),
        "threshold_percentile": float(args.threshold_percentile),
        "models": {
            "isolation_forest": f"isolation_forest_{version}.pkl",
            "autoencoder": f"autoencoder_{version}.pth",
            "vae": f"vae_{version}.pth",
            "deep_svdd": f"deep_svdd_{version}.pth",
        },
        "optuna_used": bool(args.optimize and OPTUNA_AVAILABLE),
        "optuna_best_params": {"autoencoder": best_ae, "vae": best_vae, "deep_svdd": best_svdd},
        "results_test_threshold_free": results.to_dict(orient="records"),
    }
    with open(prod_dir / f"run_summary_{version}.json", "w") as f:
        json.dump(convert_to_json_serializable(run_summary), f, indent=2)

    results.to_csv(prod_dir / f"results_{version}.csv", index=False)

    print("\n" + "=" * 90)
    print("TRAINING RESULTS (TEST — threshold-free)")
    print("=" * 90)
    print(results.to_string(index=False))
    logger.info(f"✅ Training complete. Saved to {prod_dir}")


# =============================================================================
# Ensemble (config-driven)
# =============================================================================

class ModelLoader:
    def __init__(self, models_dir: Path, version: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.version = version
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        if not self.models_dir.exists():
            raise ValueError(f"Models directory not found: {models_dir}")

    def _find_latest(self, name: str, ext: str) -> Optional[Path]:
        files = list(self.models_dir.glob(f"{name}_*.{ext}"))
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0]

    def load_isolation_forest(self) -> Optional[IsolationForest]:
        p = self.models_dir / f"isolation_forest_{self.version}.pkl" if self.version else self._find_latest("isolation_forest", "pkl")
        if not p or not p.exists():
            return None
        with open(p, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded IsolationForest: {p.name}")
        return model

    def _infer_n_features(self, checkpoint: Dict[str, Any]) -> int:
        n_features = checkpoint.get("n_features", None)
        if n_features is None:
            n_features = len(checkpoint.get("feature_names", []))
        if not n_features:
            raise ValueError("Cannot infer n_features from checkpoint.")
        return int(n_features)

    def load_autoencoder(self) -> Optional[nn.Module]:
        p = self.models_dir / f"autoencoder_{self.version}.pth" if self.version else self._find_latest("autoencoder", "pth")
        if not p or not p.exists():
            return None
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        params = ckpt.get("params", {})
        n_features = self._infer_n_features(ckpt)
        model = Autoencoder(
            input_dim=n_features,
            encoding_dim=params.get("encoding_dim", 32),
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout_rate=params.get("dropout_rate", 0.1),
            activation=params.get("activation", "relu"),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        logger.info(f"Loaded Autoencoder: {p.name}")
        return model

    def load_vae(self) -> Optional[nn.Module]:
        p = self.models_dir / f"vae_{self.version}.pth" if self.version else self._find_latest("vae", "pth")
        if not p or not p.exists():
            return None
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        params = ckpt.get("params", {})
        n_features = self._infer_n_features(ckpt)
        model = VAE(
            input_dim=n_features,
            latent_dim=params.get("latent_dim", 16),
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout_rate=params.get("dropout_rate", 0.1),
            activation=params.get("activation", "relu"),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        logger.info(f"Loaded VAE: {p.name}")
        return model

    def load_deep_svdd(self) -> Optional[nn.Module]:
        p = self.models_dir / f"deep_svdd_{self.version}.pth" if self.version else self._find_latest("deep_svdd", "pth")
        if not p or not p.exists():
            return None
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        params = ckpt.get("params", {})
        n_features = self._infer_n_features(ckpt)
        model = DeepSVDD(
            input_dim=n_features,
            rep_dim=params.get("rep_dim", 16),
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout_rate=params.get("dropout_rate", 0.0),
            activation=params.get("activation", "relu"),
        )
        
        # Load state dict (center_c buffer is included in newer checkpoints)
        state = ckpt["model_state_dict"]
        # strict=False allows loading older checkpoints that don't have center_c saved
        model.load_state_dict(state, strict=False)

        # Backward compatibility: older checkpoints stored center under top-level key "center_c"
        if "center_c" not in state and "center_c" in ckpt:
            model.center_c.data = torch.tensor(ckpt["center_c"], dtype=torch.float32, device=self.device)

        model.to(self.device).eval()
        logger.info(f"Loaded DeepSVDD: {p.name}")
        return model


class ScoreGenerator:
    def __init__(self, device: torch.device):
        self.device = device

    def score_isolation_forest(self, model: IsolationForest, X: np.ndarray) -> np.ndarray:
        return -model.decision_function(X)

    @torch.no_grad()
    def score_autoencoder(self, model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
            recon = model(xb)
            sc = F.mse_loss(recon, xb, reduction="none").mean(dim=1)
            out.append(sc.detach().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def score_vae(self, model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
            recon, _, _ = model(xb)
            sc = F.mse_loss(recon, xb, reduction="none").mean(dim=1)
            out.append(sc.detach().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def score_deep_svdd(self, model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
            z = model(xb)
            # PATCH: use buffer directly
            sc = torch.sum((z - model.center_c) ** 2, dim=1)
            out.append(sc.detach().cpu().numpy())
        return np.concatenate(out)


class EnsembleSystem:
    def __init__(self, models_dir: Path, version: Optional[str] = None):
        self.loader = ModelLoader(models_dir, version)
        self.scorer = ScoreGenerator(self.loader.device)
        self.models: Dict[str, Any] = {}
        self.weights: Optional[Dict[str, float]] = None
        self.meta_model: Optional[LogisticRegression] = None
        self._load()

    def _load(self):
        iso = self.loader.load_isolation_forest()
        if iso:
            self.models["isolation_forest"] = iso
        ae = self.loader.load_autoencoder()
        if ae:
            self.models["autoencoder"] = ae
        vae = self.loader.load_vae()
        if vae:
            self.models["vae"] = vae
        svdd = self.loader.load_deep_svdd()
        if svdd:
            self.models["deep_svdd"] = svdd
        if not self.models:
            raise ValueError("No models found.")
        logger.info(f"Loaded models: {list(self.models.keys())}")

    def raw_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        s: Dict[str, np.ndarray] = {}
        if "isolation_forest" in self.models:
            s["isolation_forest"] = self.scorer.score_isolation_forest(self.models["isolation_forest"], X)
        if "autoencoder" in self.models:
            s["autoencoder"] = self.scorer.score_autoencoder(self.models["autoencoder"], X)
        if "vae" in self.models:
            s["vae"] = self.scorer.score_vae(self.models["vae"], X)
        if "deep_svdd" in self.models:
            s["deep_svdd"] = self.scorer.score_deep_svdd(self.models["deep_svdd"], X)
        return s

    def optimize_weights(self, raw_val: Dict[str, np.ndarray], y_val: np.ndarray, normalizers: Dict[str, Dict[str, float]], metric: str) -> Dict[str, float]:
        names = list(raw_val.keys())
        S = np.column_stack([zscore_apply(raw_val[n], normalizers[n]["mu"], normalizers[n]["sigma"]) for n in names])

        def objective(w_raw):
            w = softmax(w_raw)
            ens = S @ w
            if metric == "auroc":
                return -roc_auc_score(y_val, ens)
            if metric == "auprc":
                return -average_precision_score(y_val, ens)
            raise ValueError("metric must be auroc or auprc")

        x0 = np.ones(len(names))
        res = minimize(objective, x0, method="SLSQP")
        w = softmax(res.x)
        self.weights = {n: float(wi) for n, wi in zip(names, w)}
        logger.info(f"Optimized weights ({metric}): {self.weights}")
        return self.weights

    def train_stacking(self, raw_val: Dict[str, np.ndarray], y_val: np.ndarray, normalizers: Dict[str, Dict[str, float]]) -> LogisticRegression:
        names = list(raw_val.keys())
        X_meta = np.column_stack([zscore_apply(raw_val[n], normalizers[n]["mu"], normalizers[n]["sigma"]) for n in names])
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        self.meta_model.fit(X_meta, y_val)
        logger.info("Trained stacking meta-model.")
        return self.meta_model

    def ensemble_score(self, raw: Dict[str, np.ndarray], normalizers: Dict[str, Dict[str, float]], strategy: str) -> np.ndarray:
        names = list(raw.keys())
        S = np.column_stack([zscore_apply(raw[n], normalizers[n]["mu"], normalizers[n]["sigma"]) for n in names])

        if strategy == "average":
            return S.mean(axis=1)
        if strategy == "max":
            return S.max(axis=1)
        if strategy == "weighted":
            w = self.weights or {n: 1.0 for n in names}
            wv = np.array([w.get(n, 1.0) for n in names], dtype=float)
            wv = wv / (wv.sum() if wv.sum() else 1.0)
            return S @ wv
        if strategy == "stacking":
            if self.meta_model is None:
                raise ValueError("Stacking meta-model not loaded/trained.")
            return self.meta_model.predict_proba(S)[:, 1]
        raise ValueError(f"Unknown strategy: {strategy}")


def save_ensemble_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(convert_to_json_serializable(cfg), f, indent=2)
    logger.info(f"Saved ensemble config to {path}")


def load_ensemble_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores > threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "auprc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def cmd_fit_config(args: argparse.Namespace) -> None:
    system = EnsembleSystem(Path(args.models_dir), version=args.version)

    X_val = np.load(args.val_data)
    y_val = np.load(args.val_labels).astype(int)

    raw_val = system.raw_scores(X_val)

    # Fit normalizers on VAL normals only
    normalizers: Dict[str, Dict[str, float]] = {}
    for name, s in raw_val.items():
        normalizers[name] = zscore_fit(s[y_val == 0])

    # Optional weight optimization
    if args.optimize_weights:
        system.optimize_weights(raw_val, y_val, normalizers, metric=args.optimize_metric)

    # Optional stacking training
    stacking = None
    meta_path = None
    if args.train_stacking:
        system.train_stacking(raw_val, y_val, normalizers)
        cfg_path = Path(args.save_config)
        meta_path = cfg_path.parent / "meta_model.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(system.meta_model, f)
        stacking = {"meta_model_path": str(meta_path)}

    # Thresholds for individual models (z-score space), fitted on VAL normals
    model_thresholds: Dict[str, float] = {}
    for name, s_raw in raw_val.items():
        z = zscore_apply(s_raw, normalizers[name]["mu"], normalizers[name]["sigma"])
        model_thresholds[name] = percentile_threshold(z[y_val == 0], args.threshold_percentile)

    # Thresholds for ensemble strategies (ensemble score space), fitted on VAL normals
    ensemble_thresholds: Dict[str, float] = {}
    for strat in ["average", "weighted", "max"]:
        s_ens = system.ensemble_score(raw_val, normalizers, strat)
        ensemble_thresholds[strat] = percentile_threshold(s_ens[y_val == 0], args.threshold_percentile)

    if system.meta_model is not None:
        s_stack = system.ensemble_score(raw_val, normalizers, "stacking")
        ensemble_thresholds["stacking"] = percentile_threshold(s_stack[y_val == 0], args.threshold_percentile)

    cfg = {
        "created_at": datetime.now().isoformat(),
        "threshold_policy": "percentile_on_val_normals",
        "threshold_percentile": float(args.threshold_percentile),
        "normalizers": normalizers,               # per-model mu/sigma (raw-score space)
        "weights": system.weights,                # optional
        "model_thresholds": model_thresholds,      # thresholds in z-score space
        "ensemble_thresholds": ensemble_thresholds,  # thresholds in ensemble score space
        "available_models": list(raw_val.keys()),
        "available_strategies": list(ensemble_thresholds.keys()),
        "stacking": stacking,
    }

    if not args.save_config:
        raise ValueError("--save-config is required.")
    save_ensemble_config(Path(args.save_config), cfg)


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_ensemble_config(Path(args.config))
    system = EnsembleSystem(Path(args.models_dir), version=args.version)

    # load meta-model if present
    if cfg.get("stacking") and cfg["stacking"].get("meta_model_path"):
        with open(cfg["stacking"]["meta_model_path"], "rb") as f:
            system.meta_model = pickle.load(f)

    if cfg.get("weights"):
        system.weights = {k: float(v) for k, v in cfg["weights"].items()}

    X = np.load(args.data)
    y = np.load(args.labels).astype(int)

    raw = system.raw_scores(X)
    normalizers = cfg["normalizers"]

    rows = []

    # Ensemble strategies
    for strat, thr in cfg["ensemble_thresholds"].items():
        s = system.ensemble_score(raw, normalizers, strat)
        m = evaluate_scores(y, s, float(thr))
        rows.append({"Strategy": f"Ensemble_{strat}", **m})

    # Individual models (in z-score space)
    for name, thr in cfg["model_thresholds"].items():
        z = zscore_apply(raw[name], normalizers[name]["mu"], normalizers[name]["sigma"])
        m = evaluate_scores(y, z, float(thr))
        rows.append({"Strategy": f"Model_{name}", **m})

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("EVALUATION (CONFIG-DRIVEN — no leakage)")
    print("=" * 90)
    print(df.to_string(index=False))

    if args.save_results:
        out = Path(args.save_results)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved evaluation results to {out}")


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_ensemble_config(Path(args.config))
    system = EnsembleSystem(Path(args.models_dir), version=args.version)

    # load meta-model if present
    if cfg.get("stacking") and cfg["stacking"].get("meta_model_path"):
        with open(cfg["stacking"]["meta_model_path"], "rb") as f:
            system.meta_model = pickle.load(f)

    if cfg.get("weights"):
        system.weights = {k: float(v) for k, v in cfg["weights"].items()}

    X = np.load(args.data)
    raw = system.raw_scores(X)
    normalizers = cfg["normalizers"]

    # choose strategy
    strategy = args.strategy
    if strategy not in cfg["ensemble_thresholds"]:
        raise ValueError(f"Strategy '{strategy}' not in config ensemble_thresholds: {list(cfg['ensemble_thresholds'].keys())}")

    thr = float(cfg["ensemble_thresholds"][strategy])
    scores = system.ensemble_score(raw, normalizers, strategy)
    flags = (scores > thr).astype(int)

    out = {
        "strategy": strategy,
        "threshold": thr,
        "n": int(len(flags)),
        "flag_rate": float(flags.mean()),
    }
    print(json.dumps(out, indent=2))

    if args.out:
        np.save(args.out, flags)
        logger.info(f"Saved predictions to {args.out}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified trainer for fraud anomaly models + ensemble config/inference")
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="Train models, save artifacts + score_stats.json")
    t.add_argument("--data", type=str, required=True)
    t.add_argument("--output", type=str, default="models")
    t.add_argument("--artifacts", type=str, default="artifacts")

    t.add_argument("--train-size", type=float, default=0.70)
    t.add_argument("--val-size", type=float, default=0.15)
    t.add_argument("--test-size", type=float, default=0.15)
    t.add_argument("--include-time", action="store_true", help="Include raw 'Time' as a feature (default: excluded).")

    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--early-stopping", type=int, default=20)

    t.add_argument("--seed", type=int, default=42)

    # Optuna
    t.add_argument("--optimize", action="store_true")
    t.add_argument("--optimization-trials", type=int, default=20)
    t.add_argument("--optimization-timeout", type=int, default=600)
    t.add_argument("--optimize-ae", action="store_true")
    t.add_argument("--optimize-vae", action="store_true")
    t.add_argument("--optimize-svdd", action="store_true")
    t.add_argument("--optimize-metric", type=str, default="auprc", choices=["auroc", "auprc"], help="Metric to maximize in Optuna (default: auprc)")

    # Threshold policy
    t.add_argument("--threshold-percentile", type=float, default=99.0)

    # fit-config
    fc = sub.add_parser("fit-config", help="Fit normalizers + thresholds on VAL normals and save ensemble_config.json")
    fc.add_argument("--models-dir", type=str, default="models/production")
    fc.add_argument("--version", type=str, default=None)
    fc.add_argument("--val-data", type=str, default="artifacts/X_val_scaled.npy")
    fc.add_argument("--val-labels", type=str, default="artifacts/y_val.npy")
    fc.add_argument("--threshold-percentile", type=float, default=99.0)
    fc.add_argument("--save-config", type=str, required=True)

    fc.add_argument("--optimize-weights", action="store_true")
    fc.add_argument("--optimize-metric", type=str, default="auprc", choices=["auroc", "auprc"])
    fc.add_argument("--train-stacking", action="store_true")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate using saved ensemble_config.json")
    ev.add_argument("--models-dir", type=str, default="models/production")
    ev.add_argument("--version", type=str, default=None)
    ev.add_argument("--config", type=str, required=True)
    ev.add_argument("--data", type=str, required=True)
    ev.add_argument("--labels", type=str, required=True)
    ev.add_argument("--save-results", type=str, default=None)

    # predict
    pr = sub.add_parser("predict", help="Predict flags using saved ensemble_config.json (no labels needed)")
    pr.add_argument("--models-dir", type=str, default="models/production")
    pr.add_argument("--version", type=str, default=None)
    pr.add_argument("--config", type=str, required=True)
    pr.add_argument("--data", type=str, required=True)
    pr.add_argument("--strategy", type=str, default="weighted", help="Ensemble strategy to use (must exist in config)")
    pr.add_argument("--out", type=str, default=None, help="Output .npy of 0/1 flags")

    return p


def main():
    p = build_parser()
    args = p.parse_args()

    # seeds
    np.random.seed(getattr(args, "seed", 42))
    torch.manual_seed(getattr(args, "seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(getattr(args, "seed", 42))

    if args.command == "train":
        cmd_train(args)
    elif args.command == "fit-config":
        cmd_fit_config(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()