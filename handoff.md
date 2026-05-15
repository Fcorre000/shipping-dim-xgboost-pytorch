# FedEx DIM Predictor: Production Improvements Handoff

## Context

The `shipping-dim-xgboost-pytorch` repo is a complete academic ML project that classifies FedEx DIM flags and predicts net shipping charges. Current best model is XGBoost on both tasks:

| Task | Metric | Current | Target after this work |
|------|--------|---------|------------------------|
| DIM classification | ROC AUC | 0.9997 | 0.9997 (already saturated, focus on calibration) |
| DIM classification | Calibrated probabilities | None | Isotonic-calibrated, used for audit threshold |
| Net charge regression | MAE | $3.88 | < $3.00 |
| Net charge regression | R² | 0.8658 | > 0.90 |
| Net charge regression | Prediction intervals | None | 95% coverage, distribution-free |
| Anomaly detection | None | None | Second-opinion flag with documented FPR |

The model exists to power the `dim-risk-engine` dashboard, so the unit of value is **flagged shipments worth reviewing**, not raw accuracy. That reframes the work: the goal is not "beat XGBoost" but "make outputs usable for an auditor."

## Repository state assumed at start

```
shipping-dim-xgboost-pytorch/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_gradient_boosting.ipynb       # current best XGBoost lives here
│   └── 07_final_comparison.ipynb
├── src/
│   ├── 02_preprocessing.py              # current feature pipeline
│   ├── 05_pytorch_classification.py
│   └── 06_pytorch_regression.py
├── data/                                # train/val/test parquet splits exist
├── models/                              # XGBoost .pkl, scaler, ckpts exist
├── figures/
└── 2years.csv                           # raw FedEx invoice export (proprietary)
```

The raw CSV has zip codes and other geographic columns that the current `02_preprocessing.py` drops. Phase 1 reverses that decision selectively.

## Goals (ordered by ROI)

1. Recover geographic and surcharge signal that was dropped in preprocessing.
2. Wrap the regression output in conformal prediction intervals so the dashboard can flag "actual outside expected range."
3. Calibrate classifier probabilities and run a model bake-off (LightGBM, CatBoost, TabPFN-2.5) against the new XGBoost baseline.
4. Add an unsupervised anomaly detector as a second-opinion flag.
5. Add minimal production scaffolding (experiment tracking, drift monitor, inference API stub).

Each phase is independently reviewable. Do not start Phase N+1 before Phase N is checked in and the success criteria are met.

---

## Phase 1: Data engineering (biggest win, do this first)

### 1.1 Re-add geographic features to preprocessing

Edit `src/02_preprocessing.py` so the following columns are kept and engineered, not dropped:

* `Shipper Postal Code`
* `Recipient Postal Code`
* `Recipient State/Province`

Then derive:

```python
# Distance between origin and destination zips
# Use uszipcode or pgeocode (pip install pgeocode) for lat/lon lookup
import pgeocode
nomi = pgeocode.Nominatim('us')

def zip_distance(shipper_zip, recipient_zip):
    # returns km; convert to miles outside if needed
    s = nomi.query_postal_code(str(shipper_zip).zfill(5))
    r = nomi.query_postal_code(str(recipient_zip).zfill(5))
    # haversine formula on (s.latitude, s.longitude) and (r.latitude, r.longitude)
    ...

df['origin_dest_miles'] = ...
df['shipper_lat'] = ...
df['shipper_lon'] = ...
df['recipient_lat'] = ...
df['recipient_lon'] = ...
```

Cache the zip-to-lat/lon lookup to a parquet file in `data/zip_lookup.parquet` so subsequent runs do not re-query pgeocode.

### 1.2 Add DAS (Delivery Area Surcharge) flag

FedEx publishes a DAS/EDAS zip list quarterly as a PDF at https://www.fedex.com/en-us/shipping/surcharges.html. Manual one-time step: download the current zip list, parse it into `data/das_zips.csv` with columns `zip, das_type` where `das_type in {NONE, DAS, EDAS, REMOTE}`.

Then in preprocessing:

```python
das_lookup = pd.read_csv('data/das_zips.csv', dtype={'zip': str})
das_map = dict(zip(das_lookup['zip'], das_lookup['das_type']))
df['das_type'] = df['Recipient Postal Code'].astype(str).str.zfill(5).map(das_map).fillna('NONE')
# one-hot encode das_type in the existing get_dummies call
```

### 1.3 Decompose the regression target

Net charge is the sum of base transport, fuel surcharge, and accessorial surcharges. The raw CSV has the breakdown in the columns currently dropped as leakage (which is correct for a single-target model). For decomposition, we want to predict each component independently from the same input features and sum them.

Create three new targets in preprocessing (only for the regression task):

```python
df['log_base_charge']   = np.log1p(df['Shipment Freight Charge Amount USD'])
df['log_fuel_charge']   = np.log1p(df['Shipment Freight Charge Amount USD'] * df['fuel_pct'])  # if you can derive
df['log_misc_charge']   = np.log1p(df['Shipment Miscellaneous Charge USD'])
```

If the fuel and miscellaneous columns are too sparse or unreliable, fall back to a single target but keep the option open.

### 1.4 Refit XGBoost as the new baseline

Add a new notebook `notebooks/08_xgboost_v2.ipynb` that:
1. Loads the new parquets.
2. Refits XGBoost classification and regression with default hyperparameters from `04_gradient_boosting.ipynb`.
3. Runs SHAP beeswarm on the new features (especially `origin_dest_miles` and `das_type_*`).
4. Saves `models/xgb_classifier_v2.pkl` and `models/xgb_regressor_v2.pkl`.

### Phase 1 success criteria

* Validation MAE for the new regressor is below $3.30 (current val MAE is roughly $3.36).
* `origin_dest_miles` appears in the top 8 SHAP features for regression.
* `models/xgb_*_v2.pkl` files exist and load cleanly.
* All Phase 1 code is in one commit; old `_v1` artifacts are preserved.

---

## Phase 2: Conformal prediction and calibration

### 2.1 Install and wire MAPIE for regression intervals

```bash
pip install mapie==0.9.*
```

Create `notebooks/09_uncertainty_quantification.ipynb`:

```python
from mapie.regression import MapieRegressor
from xgboost import XGBRegressor
import joblib

xgb_reg = joblib.load('models/xgb_regressor_v2.pkl')

# Split-conformal: fit on train, calibrate on val, evaluate on test
mapie = MapieRegressor(estimator=xgb_reg, method='base', cv='prefit')
mapie.fit(X_val, y_val)  # calibration only since estimator is prefit

# alpha = 0.05 gives 95% intervals
y_pred, y_pis = mapie.predict(X_test, alpha=0.05)
# y_pis shape: (n_test, 2, 1) -> [:, 0, 0] is lower, [:, 1, 0] is upper

# In log space, so transform back
lower = np.expm1(y_pis[:, 0, 0])
upper = np.expm1(y_pis[:, 1, 0])
pred  = np.expm1(y_pred)
```

Validate that the empirical coverage on the test set is at least 94%. If it is below, the calibration set is too small or the target distribution shifted; investigate before moving on.

### 2.2 Isotonic calibration for the classifier

The classifier sits at 0.9997 AUC, so ranking is fine, but its raw probabilities are not necessarily calibrated. For threshold-setting on the audit queue this matters.

```python
from sklearn.calibration import CalibratedClassifierCV

xgb_clf = joblib.load('models/xgb_classifier_v2.pkl')
calibrated = CalibratedClassifierCV(estimator=xgb_clf, method='isotonic', cv='prefit')
calibrated.fit(X_val, y_val)

joblib.dump(calibrated, 'models/xgb_classifier_v2_calibrated.pkl')
```

Compare reliability diagrams (predicted vs observed probability bucketed into deciles) for raw vs calibrated. Save both plots to `figures/calibration_*.png`.

### 2.3 Export an audit-ready prediction function

Create `src/predict.py` that exposes a single function the dashboard backend can call:

```python
def predict_shipment(row: dict) -> dict:
    """
    Input: dict with raw shipment fields (weight, dims, zone, zips, etc.)
    Output: {
        'dim_predicted': bool,
        'dim_probability': float,
        'dim_disagrees_with_fedex': bool,
        'charge_predicted': float,
        'charge_lower_95': float,
        'charge_upper_95': float,
        'charge_actual': float,
        'charge_outside_interval': bool
    }
    """
    ...
```

This is the contract the `dim-risk-engine` FastAPI service will import. Keep the preprocessing logic in one place (a `transform_row` helper) so it can be reused for batch and single-row paths.

### Phase 2 success criteria

* `MapieRegressor` test-set coverage at 95% target is between 94% and 96%.
* Mean interval width is documented in the notebook (lower is better; expect roughly $15 to $25).
* Calibrated classifier reliability diagram visibly better than raw.
* `src/predict.py` runs end-to-end on a single test row.

---

## Phase 3: Model bake-off

Goal: find out whether XGBoost is actually the right model now that the feature set has changed, and try TabPFN-2.5 since it claims 87% win rate vs XGBoost on datasets of this size.

Create `notebooks/10_model_bakeoff.ipynb` that trains and evaluates each of the following on the same Phase 1 splits:

### 3.1 LightGBM

```python
import lightgbm as lgb
# Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    class_weight='balanced',
    random_state=42
)
lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20)])
```

### 3.2 CatBoost

Pass categorical columns directly (no one-hot needed). Will require keeping a non-one-hot version of the splits, so either branch in `02_preprocessing.py` with a `--encoding {onehot, native}` flag, or create a parallel `data/train_native.parquet`.

```python
from catboost import CatBoostClassifier, CatBoostRegressor
cat_cols = ['zone_clean', 'service_type', 'pay_type', 'das_type']
cat_clf = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                              cat_features=cat_cols, verbose=50)
cat_clf.fit(X_train_native, y_train, eval_set=(X_val_native, y_val))
```

### 3.3 TabPFN-2.5

This is the interesting one. Released November 2025 by Prior Labs, transformer foundation model, in-context learning on tabular data. Requires `pip install tabpfn` and the dataset must fit within 50K rows and 2K features (your train is 45K rows and 42 features, which fits).

```python
from tabpfn import TabPFNClassifier, TabPFNRegressor

# Classifier
tab_clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
tab_clf.fit(X_train, y_train)
tab_pred = tab_clf.predict_proba(X_test)

# Regressor
tab_reg = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
tab_reg.fit(X_train, y_train_log)
tab_reg_pred = tab_reg.predict(X_test)
```

Note: TabPFN-2.5 base inference is slow per row (transformer forward pass). For production we would use their distillation engine to compress into a small MLP or tree ensemble, but for the bake-off just measure raw performance and note the latency cost.

If TabPFN-2.5 wins materially on either task, add a follow-up ticket for distillation. If it ties or loses, document the finding and stop.

### 3.4 AutoGluon-Tabular as honest baseline

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='dim_flag', eval_metric='roc_auc').fit(
    train_df, time_limit=3600, presets='best_quality'
)
```

This stacks LightGBM, CatBoost, XGBoost, NN, and TabPFN automatically. It is the most honest "what is the best we can do" number to anchor against. Run it once, log the leaderboard, do not productionize the stack (too heavy for the dashboard).

### Phase 3 success criteria

* All four candidates trained and logged on the same Phase 1 splits.
* Single comparison table in the notebook with MAE, R², AUC, F1, and inference latency per model.
* Decision documented: which single model goes to production, and why.
* Best model checkpointed under `models/best_v3.pkl` (or `.ckpt` for TabPFN).

---

## Phase 4: Anomaly detection second opinion

The supervised models say "I expected X, you charged Y." A second unsupervised model says "this shipment looks unusual." The intersection of "model disagrees with FedEx" and "shipment is anomalous" is the high-precision audit queue.

Create `src/anomaly.py`:

### 4.1 Isolation Forest baseline

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(X_train)
anomaly_score = -iso.score_samples(X_test)  # higher = more anomalous
```

### 4.2 Autoencoder

Train a small PyTorch Lightning autoencoder on the scaled features. Reconstruction error per row is the anomaly score. Reuse the lightning scaffolding from `05_pytorch_classification.py`.

Target architecture: input dim -> 64 -> 16 -> 64 -> input dim, MSE loss, AdamW, 30 epochs.

### 4.3 PyOD ensemble (optional)

```python
from pyod.models.combination import aom
# combine isolation forest, autoencoder, LOF, COPOD scores
```

### 4.4 Combined flag in `predict.py`

Add to the prediction output:

```python
'anomaly_score': float,           # 0 to 1, percentile rank
'anomaly_flagged': bool,          # score > 0.95
'review_recommended': bool,        # dim_disagrees OR charge_outside_interval OR anomaly_flagged
'review_priority': 'high'|'medium'|'low'  # high if 2+ signals fire
```

### Phase 4 success criteria

* Isolation Forest and autoencoder both produce per-row anomaly scores.
* On held-out test data, the combined `review_recommended` flag fires on between 2% and 8% of shipments. Outside that range the threshold is too loose or too tight; retune.
* `src/anomaly.py` exposes a single `score_shipment(row)` function.

---

## Phase 5: Production scaffolding

This phase is light and exists so the dashboard team can integrate cleanly.

### 5.1 Experiment tracking

```bash
pip install mlflow
```

Wrap every model fit in `mlflow.start_run()` and log hyperparameters, metrics, and the model artifact. Local file backend is fine (`./mlruns`). Do not set up a tracking server.

### 5.2 Drift detection stub

```python
# src/drift.py
from scipy.stats import wasserstein_distance

def population_stability_index(reference, current, bins=10):
    # standard PSI; PSI > 0.2 flags drift
    ...

def check_drift(reference_parquet: str, current_parquet: str) -> dict:
    """Returns {feature: psi_value} for all features."""
    ...
```

No scheduler, no alerts, just a function the dashboard backend can call monthly.

### 5.3 Inference API contract

Write `src/api_contract.md` documenting the JSON shape of `predict_shipment` input and output. This is what the FastAPI service in `dim-risk-engine` will conform to. Keep it short, one page.

### Phase 5 success criteria

* `mlruns/` directory exists with at least the Phase 3 bake-off runs logged.
* `src/drift.py` runs end-to-end on train vs test as a sanity check.
* `src/api_contract.md` exists.

---

## Dependencies to add

```
mapie>=0.9.0
lightgbm>=4.3.0
catboost>=1.2.5
tabpfn>=2.5.0
pyod>=2.0.0
pgeocode>=0.5.0
mlflow>=2.10.0
autogluon-tabular>=1.1.0   # optional, large install
```

Append to `requirements.txt` only as each phase uses them. Do not install everything up front; AutoGluon especially is heavy.

## Out of scope (do not do these)

* Hyperparameter tuning the existing XGBoost. The marginal gain is below what Phase 1 will produce. Tune the *new* model in Phase 3 if it warrants it.
* Replacing the existing PyTorch FFNN. It was an academic comparison and is documented.
* Building the dashboard frontend. That lives in `dim-risk-engine`.
* Distilling TabPFN-2.5 into an MLP. Only do this if TabPFN wins Phase 3 by a meaningful margin.
* Online learning or active learning. Important eventually, but premature without operator labels.

## Working agreement

* One commit per phase, with a brief markdown summary of metrics and decisions in `documentation/phase_N_summary.md`.
* Do not delete `_v1` artifacts. Side-by-side, not in-place.
* If a phase fails to hit its success criteria, stop and write up what was tried before moving on. A negative result is a finding.
* Preserve the existing dark-themed HTML notes workflow conventions for any new documentation.

## First-pass execution order

1. Phase 1.1 and 1.2 (zip codes + DAS), commit, verify Phase 1 success criteria.
2. Phase 1.3 and 1.4 (target decomposition + new XGBoost), commit.
3. Phase 2 in full, commit.
4. Phase 3 in full, commit.
5. Phase 4 in full, commit.
6. Phase 5 in full, commit.

Estimated total wall-clock: 3 to 5 focused days for someone familiar with the repo.
