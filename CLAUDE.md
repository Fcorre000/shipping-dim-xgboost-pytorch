# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FedEx DIM Weight & Shipping Cost Predictor — a ML pipeline for a mattress manufacturer using 53,014 real FedEx shipments to:
1. **Classify** whether a package will be DIM-flagged before pickup (binary classification)
2. **Predict** the net shipping charge in USD (regression)

Built for CSE 4309: Fundamentals of Machine Learning, UT Arlington (Spring 2026). Author: Fernando Correa.

## Environment Setup

```bash
conda create -n fedex-ml python=3.10 && conda activate fedex-ml
pip install -r requirements.txt
```

**Critical compatibility constraint**: `numpy>=1.24,<2.0` — TensorBoard is incompatible with NumPy 2.0+. `setuptools==69.5.1` is pinned because TensorBoard depends on `pkg_resources`.

## Running the Pipeline

```bash
# Step 1: Preprocessing (generates parquet splits in data/)
python src/02_preprocessing.py

# Step 2: Train PyTorch models
python src/05_pytorch_classification.py
python src/06_pytorch_regression.py

# Step 3: Monitor training
tensorboard --logdir=logs/

# Step 4: Open notebooks for baseline/boosting models and final comparison
jupyter notebook
```

Notebooks run in order: `01_eda` → `03_baseline_models` → `04_gradient_boosting` → `07_final_comparison`.

## Architecture

### Pipeline Flow
```
FedEx_ShipmentDetail.xlsx (65 features, 53,014 rows)
  → 01_eda.ipynb          (analysis, produces figures/)
  → src/02_preprocessing.py   (feature engineering → parquet splits in data/)
  → 03_baseline_models.ipynb  (Logistic Regression, Linear Regression)
  → 04_gradient_boosting.ipynb (AdaBoost, XGBoost + SHAP)
  → src/05_pytorch_classification.py  (PyTorch Lightning FFNN classifier)
  → src/06_pytorch_regression.py      (PyTorch Lightning FFNN regressor)
  → 07_final_comparison.ipynb         (model comparison table)
```

### Two Parallel Tasks
Both classification and regression run through the same pipeline stages. Models are compared: Logistic/Linear Regression → AdaBoost/XGBoost → PyTorch FFNN.

### Key ML Decisions
- **Leakage guard**: `Rated Weight` must be excluded from Task 1 features — it is derived from the DIM flag (correlation ~0.95)
- **Class imbalance**: 40.7% DIM=Y vs 59.3% DIM=N (ratio 1.46:1) — mild, use `class_weight='balanced'`
- **Regression loss**: Huber loss for robustness to the right-skewed heavy tail in net charge ($0–$3000+)
- **Decision threshold**: Tune classification threshold on val set to minimize false negatives (missed DIM flags = direct cost)
- **Interpretability**: SHAP beeswarm plots for XGBoost models
- **Optimizer**: AdamW + Cosine LR schedule for both PyTorch modules

### Engineered Features (created in preprocessing)
- `volume = height × width × length`
- `dim_weight_calc = volume / 139` (FedEx domestic DIM divisor)
- `dim_weight_ratio = dim_weight_calc / actual_weight` (>1.0 = DIM trigger condition)
- `cost_per_pound = freight_charge / actual_weight`
- `has_dimensions`: binary flag (1 if all dims > 0, else 0)
- `zone_clean`: standardized pricing zone (raw data has "2" vs "02" dirty values)

### Preprocessing Decision Log (from `documentation/01_eda_notes.docx`)

**Rows to remove:**
- 485 `NonTrans` rows (no DIM flag, non-transport billing entries)
- 3 DIM=Y rows with zero dimensions (physically impossible — data errors)
- 90 international shipments (out of scope for domestic model)

**Columns to drop:**
- High-null: Department Number (99.9%), Customs Value Currency Code (100%), Recipient Original State/Province (99.2%)
- Zero-variance: Weight Type Code (always 'lb'), Exchange Rate to USD (always 1), Billed Currency Code (always 'USD')
- Identifiers: tracking number, invoice number, all address/name columns
- **`Shipment Rated Weight (Pounds)` — leaked feature, excluded from BOTH tasks**

**Categorical encoding (one-hot):**
- Service Type (~6 meaningful categories)
- Pay Type (3 categories: Bill_Sender_Prepaid, Bill_Third_Party, Bill_Recipient)
- Pricing Zone — normalize first with `f'{int(z):02d}'`, collapse non-standard codes (D, A, C, N, 51+) to `'Other'`

**Scaling:**
- `StandardScaler` on numeric features for PyTorch models — save as `models/preprocessor.pkl`
- Tree models (XGBoost, AdaBoost) use raw unscaled values — save both versions as separate parquets

**Target transforms:**
- Task 1: DIM Flag → binary 0/1
- Task 2: `np.log1p(net_charge)` for training; recover with `np.expm1(pred)` for dollar-amount metrics

**Train/val/test split:** 80/10/10 stratified on DIM flag, `random_state=42`
Save as: `data/train.parquet`, `data/val.parquet`, `data/test.parquet`

### Key EDA Findings
- **DIM flag AUC check**: actual weight AUC = 0.9056 (legitimate); rated weight effective AUC = 0.8489 (leaked — inverted, raw roc_auc prints as 0.1511)
- **Net charge skewness = 23.2** — extreme right tail; log transform is mandatory for regression
- **DIM=N costs more on average** ($67.30) than DIM=Y ($46.59) — heavier packages dominate cost even without DIM billing
- **DIM rate by service type**: Home Delivery 67%, Return Manager 76%, Ground 26%, Express 18%
- **DIM rate by zone**: Zones 02–03 ~50%, Zone 08 (cross-country) ~23%

### Data Notes
- Raw file: `FedEx_ShipmentDetail.xlsx` (not committed; load from local path)
- Processed splits saved as parquet in `data/`; model checkpoints in `models/`
- Fitted preprocessor saved as `models/preprocessor.pkl` for inference-time use

## Implementation Status

- [x] `notebooks/01_eda.ipynb` — complete
- [ ] `src/02_preprocessing.py` — empty stub
- [ ] `notebooks/03_baseline_models.ipynb` — empty stub
- [ ] `notebooks/04_gradient_boosting.ipynb` — empty stub
- [ ] `src/05_pytorch_classification.py` — empty stub
- [ ] `src/06_pytorch_regression.py` — empty stub
- [ ] `notebooks/07_final_comparison.ipynb` — empty stub
