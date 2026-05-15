"""Replay the Phase 3 bake-off into a local MLflow store.

The metric numbers and trained model pickles already exist on disk from
`notebooks/10_model_bakeoff.ipynb`. Rather than re-train, this script reads
`models/phase_3_metrics.json` and the saved model artifacts, then logs one
MLflow run per (model, task) pair under the `phase_3_bakeoff` experiment.

Backend: local file store at `./mlruns` (no tracking server). Open the UI with
`mlflow ui --backend-store-uri ./mlruns`.

Usage:
    python src/log_to_mlflow.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'

CLS_ARTIFACTS = {
    'XGBoost v2': MODELS / 'xgb_classifier_v2_calibrated.pkl',
    'LightGBM':   MODELS / 'best_v3_classifier.pkl',
    'CatBoost':   MODELS / 'best_v3_classifier.pkl',
}
REG_ARTIFACTS = {
    'XGBoost v2': MODELS / 'xgb_regressor_v2_conformal.pkl',
    'LightGBM':   MODELS / 'best_v3_regressor.pkl',
    'CatBoost':   MODELS / 'best_v3_regressor.pkl',
}


def _log_one(model_name: str, task: str, metrics: dict, artifact_path: Path):
    with mlflow.start_run(run_name=f'{model_name} | {task}'):
        mlflow.set_tags({
            'model':  model_name,
            'task':   task,
            'phase':  'phase_3_bakeoff',
            'split':  'test',
        })
        for k, v in metrics.items():
            if k == 'model' or v is None:
                continue
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                mlflow.log_param(k, v)
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path='model')


def main():
    with open(MODELS / 'phase_3_metrics.json') as f:
        m = json.load(f)

    mlflow.set_tracking_uri(f'file://{(ROOT / "mlruns").resolve()}')
    mlflow.set_experiment('phase_3_bakeoff')

    for row in m['classification_table']:
        _log_one(row['model'], 'classification', row,
                 CLS_ARTIFACTS.get(row['model'], MODELS / 'phase_3_metrics.json'))

    for row in m['regression_table']:
        _log_one(row['model'], 'regression', row,
                 REG_ARTIFACTS.get(row['model'], MODELS / 'phase_3_metrics.json'))

    with mlflow.start_run(run_name='winners'):
        mlflow.set_tags({'phase': 'phase_3_bakeoff', 'role': 'summary'})
        mlflow.log_param('classification_winner', m['classification_winner'])
        mlflow.log_param('regression_winner',     m['regression_winner'])
        mlflow.log_artifact(str(MODELS / 'phase_3_metrics.json'))

    print(f'logged to {ROOT / "mlruns"}')
    print('open the UI: mlflow ui --backend-store-uri ./mlruns')


if __name__ == '__main__':
    main()
