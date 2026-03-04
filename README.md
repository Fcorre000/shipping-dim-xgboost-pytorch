<div align="center">

# 📦 FedEx DIM Weight & Shipping Cost Predictor

### End-to-end machine learning on 53,000 real-world shipments — from raw invoice data to a deployable pre-shipment decision tool

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.x-792EE5?style=flat-square&logo=lightning&logoColor=white)](https://lightning.ai)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FF6600?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## The Problem

FedEx charges by **dimensional (DIM) weight** when a package's volume exceeds a threshold relative to its actual weight — often resulting in shipping costs far above what the actual weight alone would suggest. For high-volume shippers like furniture and mattress manufacturers, this creates a significant and largely *avoidable* cost: if you know a package will be DIM-flagged before it ships, you can repack it.

This project uses a real FedEx invoice export from a mattress manufacturing company to build two production-oriented ML models:

| Task | Type | Target | Business Value |
|------|------|--------|----------------|
| **Task 1** | Binary Classification | DIM flag (Y/N) | Flag packages for repacking *before* pickup |
| **Task 2** | Regression | Net shipping charge ($) | Accurate pre-dispatch cost estimation |

> **Key stat from the data:** 40% of shipments are DIM-flagged. Affected packages are billed on average **15.8 lbs heavier** than their actual weight. On a fleet of 53,000 annual shipments, the cost exposure is substantial.

---

## Results at a Glance

### Task 1 — DIM Flag Classification

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression *(baseline)* | — | — | — | — |
| AdaBoost | — | — | — | — |
| **XGBoost** | — | — | — | — |
| **PyTorch Lightning FFNN** | — | — | — | — |

### Task 2 — Net Charge Regression

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression *(baseline)* | — | — | — |
| **XGBoost** | — | — | — |
| **PyTorch Lightning FFNN** | — | — | — |

> *Results will be populated as training completes. See [`07_final_comparison.ipynb`](notebooks/07_final_comparison.ipynb) for the full comparison.*

---

## Tech Stack

```
Data & Features      scikit-learn · pandas · NumPy · SHAP
Gradient Boosting    XGBoost · AdaBoost (scikit-learn)
Deep Learning        PyTorch Lightning · AdamW · Huber Loss · Cosine LR Schedule
Experiment Tracking  TensorBoard / W&B (optional)
Hyperparameter Opt   Optuna
Environment          Python 3.10 · conda
```

---

## Dataset

- **Source:** Real FedEx invoice export (domestic shipments, mattress manufacturer)
- **Size:** 53,014 shipments × 65 features
- **Not included in repo** (proprietary invoice data). Preprocessed parquet splits are provided instead.

**Key raw features:**

```
Numeric    actual weight, rated weight, height, width, length,
           freight charge, discount amount, misc charges

Categorical  pricing zone (2–8), service type (Ground / Home Delivery / etc.),
             shipper company, pay type
```

**Engineered features:**

```python
volume          = height × width × length
dim_weight_ratio = volume / 139          # FedEx DIM divisor
cost_per_pound  = freight_charge / actual_weight
```

---

## Repository Structure

```
fedex-dim-predictor/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory analysis & audit
│   ├── 03_baseline_models.ipynb      # Logistic + Linear Regression
│   ├── 04_gradient_boosting.ipynb    # AdaBoost + XGBoost + SHAP
│   └── 07_final_comparison.ipynb     # Full model comparison tables & plots
│
├── src/
│   ├── 02_preprocessing.py           # Feature engineering pipeline
│   ├── 05_pytorch_classification.py  # PL LightningModule — DIM classifier
│   └── 06_pytorch_regression.py      # PL LightningModule — charge regressor
│
├── models/                           # Saved checkpoints (.pkl / .ckpt)
├── data/                             # Preprocessed parquet splits
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/fedex-dim-predictor.git
cd fedex-dim-predictor
conda create -n fedex-ml python=3.10 && conda activate fedex-ml
pip install -r requirements.txt

# 2. Run preprocessing
python src/02_preprocessing.py

# 3. Train gradient boosting models
jupyter notebook notebooks/04_gradient_boosting.ipynb

# 4. Train PyTorch Lightning models
python src/05_pytorch_classification.py
python src/06_pytorch_regression.py

# 5. View full comparison
jupyter notebook notebooks/07_final_comparison.ipynb
```

> **GPU Note:** PyTorch Lightning will automatically use a CUDA GPU if available. For free GPU access, open the training scripts directly in [Google Colab](https://colab.research.google.com/).

---

## ML Design Decisions Worth Noting

**Why compare gradient boosting vs. deep learning on tabular data?**
Tree-based ensembles (especially XGBoost) tend to outperform neural networks on structured tabular data. This project tests that assumption directly on a real dataset — a more honest comparison than most tutorials provide.

**Leakage guard on Task 1:**
`rated_weight` is derived *from* the DIM flag and is excluded from classification inputs. Including it would produce a near-perfect but entirely useless model.

**Decision threshold tuning:**
Rather than defaulting to 0.5, the classification threshold is tuned on the validation set to minimize false negatives — because a missed DIM shipment has a direct, measurable cost.

**Huber loss for regression:**
Shipping charges have heavy-tailed outliers (very large or bulky shipments). Huber loss makes the neural network robust to these without requiring outlier removal.

**SHAP for interpretability:**
XGBoost feature importances are explained using SHAP beeswarm plots, making the model's decisions auditable — important for any tool used in an operational context.

---

## Practical Output

The end goal is a lightweight **pre-shipment decision tool**: given a package's dimensions and weight, it returns a DIM risk prediction and an estimated net charge. This is the kind of tooling a shipping manager could run before scheduling a pickup.

---

## Academic Context

Built as a semester project for **CSE 4309: Fundamentals of Machine Learning** at the University of Texas at Arlington (Spring 2026). The dataset is sourced from a real industry context — not a Kaggle benchmark — which makes the preprocessing, leakage analysis, and feature engineering more representative of actual ML engineering work.

---

## Author

**Fernando Correa**
B.S. Computer Science · University of Texas at Arlington


---

<div align="center">
<sub>Built with real data · real business constraints · and no clean benchmark to hide behind.</sub>
</div>
