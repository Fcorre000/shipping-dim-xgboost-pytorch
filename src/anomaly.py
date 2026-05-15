"""Unsupervised second-opinion flag for the audit queue.

Two detectors run in parallel and their scores are combined:

* `IsolationForest` — tree-based, works on raw (unscaled) features
* `Autoencoder` (PyTorch Lightning, 105 → 64 → 16 → 64 → 105) — reconstruction
  error on StandardScaler-normalised features

Each row gets a percentile-rank score from each detector; the final
`anomaly_score` is the average. A pre-tuned `anomaly_threshold` (saved at
training time so the test-set fire-rate sits in the 2–8% band the handoff
specifies) decides `anomaly_flagged`.

Artifacts loaded:
    models/isolation_forest.pkl       (sklearn pickle)
    models/autoencoder.pt             (state dict + arch config)
    models/scaler_v2.pkl              (StandardScaler fit on v2 features)
    models/anomaly_threshold.json     (threshold + train-time score quantiles)
    models/feature_columns.json       (Phase 2 artifact, 105-col order)

Public API:
    score_batch(X: pd.DataFrame)           -> ndarray of percentile scores
    score_shipment(row: dict)              -> {anomaly_score, anomaly_flagged}
    AnomalyAutoencoder                     (Lightning module, exported for training)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'


class AnomalyAutoencoder(pl.LightningModule):
    """105 → 64 → 16 → 64 → 105 MSE autoencoder, AdamW + cosine LR."""

    def __init__(self, input_dim: int = 105, hidden_dim: int = 64,
                 latent_dim: int = 16, lr: float = 1e-3, max_epochs: int = 30):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _step(self, batch, stage: str):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        loss = self.loss_fn(self(x), x)
        self.log(f'{stage}_loss', loss, prog_bar=(stage == 'val'))
        return loss

    def training_step(self, batch, _):  return self._step(batch, 'train')
    def validation_step(self, batch, _): return self._step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return [opt], [sched]

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """Per-row MSE reconstruction error."""
        self.eval()
        recon = self(x)
        return ((recon - x) ** 2).mean(dim=1).cpu().numpy()


@lru_cache(maxsize=1)
def _load_artifacts():
    iso = joblib.load(MODELS / 'isolation_forest.pkl')
    scaler = joblib.load(MODELS / 'scaler_v2.pkl')

    ae_blob = torch.load(MODELS / 'autoencoder.pt', map_location='cpu', weights_only=False)
    ae = AnomalyAutoencoder(**ae_blob['hparams'])
    ae.load_state_dict(ae_blob['state_dict'])
    ae.eval()

    with open(MODELS / 'anomaly_threshold.json') as f:
        meta = json.load(f)
    with open(MODELS / 'feature_columns.json') as f:
        feature_cols = json.load(f)

    iso_ref = np.asarray(meta['iso_reference_scores'])
    ae_ref  = np.asarray(meta['ae_reference_scores'])
    return iso, scaler, ae, iso_ref, ae_ref, float(meta['threshold']), feature_cols


def _percentile_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fraction of reference points <= each value, in [0, 1]."""
    return np.searchsorted(reference, values, side='right') / len(reference)


def score_batch(X: pd.DataFrame) -> dict:
    """Compute combined anomaly score for a DataFrame of feature rows.

    Returns
    -------
    dict with keys
        iso_score (ndarray)   — percentile rank of IsolationForest score
        ae_score  (ndarray)   — percentile rank of autoencoder MSE
        anomaly_score (ndarray) — mean of the two percentiles
        anomaly_flagged (ndarray of bool)
    """
    iso, scaler, ae, iso_ref, ae_ref, threshold, _ = _load_artifacts()

    # IF: higher (less negative) score_samples = more normal; flip sign so high = anomalous
    iso_raw = -iso.score_samples(X.values)
    iso_pct = _percentile_rank(iso_ref, iso_raw)

    X_scaled = scaler.transform(X.values).astype(np.float32)
    ae_raw = ae.reconstruction_error(torch.from_numpy(X_scaled))
    ae_pct = _percentile_rank(ae_ref, ae_raw)

    combined = (iso_pct + ae_pct) / 2
    return {
        'iso_score':       iso_pct,
        'ae_score':        ae_pct,
        'anomaly_score':   combined,
        'anomaly_flagged': combined >= threshold,
    }


def score_shipment(X: pd.DataFrame) -> dict:
    """Single-row anomaly score.

    Caller is responsible for passing a one-row DataFrame already in the
    105-column v2 feature layout — produced by `predict.transform_row(raw_dict)`.
    Keeping this contract narrow avoids two parallel feature pipelines.
    """
    _, _, _, _, _, threshold, feature_cols = _load_artifacts()
    X = X[feature_cols]
    out = score_batch(X)
    return {
        'anomaly_score':   float(out['anomaly_score'][0]),
        'anomaly_flagged': bool(out['anomaly_flagged'][0]),
        'iso_score':       float(out['iso_score'][0]),
        'ae_score':        float(out['ae_score'][0]),
        'threshold':       threshold,
    }
