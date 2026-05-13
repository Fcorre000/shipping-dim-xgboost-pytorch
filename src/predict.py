"""Audit-ready inference for the dim-risk-engine dashboard.

`predict_shipment(row)` is the single contract the FastAPI service calls. It runs
the v2 calibrated classifier and the conformal-wrapped v2 regressor against a
single shipment and returns the schema documented below.

Artifacts loaded (all produced in Phase 1 and Phase 2):
    models/xgb_classifier_v2_calibrated.pkl
    models/xgb_regressor_v2_conformal.pkl       (SplitConformalRegressor)
    models/feature_columns.json                  (105-feature column order)
    data/zip_lookup.parquet                      (pgeocode cache)
    data/das_zips.csv                            (FedEx DAS tier lookup)
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CM_PER_IN = 2.54
DIM_DIVISOR = 139
RATE_CARD_BASE_YEAR = 2024


@lru_cache(maxsize=1)
def _load_artifacts():
    clf = joblib.load(ROOT / 'models' / 'xgb_classifier_v2_calibrated.pkl')
    reg = joblib.load(ROOT / 'models' / 'xgb_regressor_v2_conformal.pkl')
    with open(ROOT / 'models' / 'feature_columns.json') as f:
        feature_cols = json.load(f)

    zip_df = pd.read_parquet(ROOT / 'data' / 'zip_lookup.parquet')
    zip_lookup = {row['zip']: (row['latitude'], row['longitude'])
                  for _, row in zip_df.iterrows()}

    das_path = ROOT / 'data' / 'das_zips.csv'
    if das_path.exists():
        das_df = pd.read_csv(das_path, dtype={'zip': str})
        das_map = dict(zip(das_df['zip'].str.zfill(5), das_df['das_type']))
    else:
        das_map = {}

    return clf, reg, feature_cols, zip_lookup, das_map


def _normalize_zip(z) -> str:
    digits = ''.join(c for c in str(z) if c.isdigit())
    return digits[:5] if len(digits) >= 5 else digits.zfill(5)


def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * R * asin(sqrt(max(0.0, min(1.0, a))))


def _clean_zone(z) -> str:
    try:
        num = int(float(z))
        if num > 50 or num < 1:
            return 'Other'
        return f'{num:02d}'
    except (ValueError, TypeError):
        return 'Other'


def transform_row(row: dict) -> pd.DataFrame:
    """Map a raw shipment dict to the 105-column feature row the v2 models expect.

    Mirrors `src/02_preprocessing.py` exactly: same dim divisor, same zone
    cleanup, same one-hot scheme, same NaN→0 fallback.
    """
    _, _, feature_cols, zip_lookup, das_map = _load_artifacts()

    # Numeric base
    weight = float(row.get('Original Weight (Pounds)') or 0)
    h_cm = float(row.get('Dimmed Height (cm)') or 0)
    w_cm = float(row.get('Dimmed Width (cm)') or 0)
    l_cm = float(row.get('Dimmed Length (cm)') or 0)
    h, w, l = h_cm / CM_PER_IN, w_cm / CM_PER_IN, l_cm / CM_PER_IN

    volume = h * w * l
    dim_calc = volume / DIM_DIVISOR
    dim_ratio = (dim_calc / weight) if weight > 0 else 0.0
    if not np.isfinite(dim_ratio):
        dim_ratio = 0.0
    has_dims = int(h > 0 and w > 0 and l > 0)
    billable = max(weight, dim_calc)
    billable_ceil = math.ceil(billable) if billable > 0 else 0

    # Time
    yyyymm = int(row.get('Invoice Month (yyyymm)') or 0)
    ship_year = yyyymm // 100 if yyyymm else 0
    ship_month = yyyymm % 100 if yyyymm else 0
    months_since_start = (ship_year - RATE_CARD_BASE_YEAR) * 12 + ship_month if yyyymm else 0

    # Geographic
    s_zip = _normalize_zip(row.get('Shipper Postal Code', ''))
    r_zip = _normalize_zip(row.get('Recipient Postal Code', ''))
    s_lat, s_lon = zip_lookup.get(s_zip, (np.nan, np.nan))
    r_lat, r_lon = zip_lookup.get(r_zip, (np.nan, np.nan))

    if all(np.isfinite([s_lat, s_lon, r_lat, r_lon])):
        miles = _haversine_miles(s_lat, s_lon, r_lat, r_lon)
    else:
        miles = 0.0
    s_lat = 0.0 if not np.isfinite(s_lat) else s_lat
    s_lon = 0.0 if not np.isfinite(s_lon) else s_lon
    r_lat = 0.0 if not np.isfinite(r_lat) else r_lat
    r_lon = 0.0 if not np.isfinite(r_lon) else r_lon

    # Assemble feature dict, all OHE columns default to 0
    out = {col: 0.0 for col in feature_cols}
    numeric = {
        'Original Weight (Pounds)': weight,
        'Dimmed Height (cm)': h_cm,
        'Dimmed Width (cm)': w_cm,
        'Dimmed Length (cm)': l_cm,
        'volume': volume,
        'dim_weight_calculator': dim_calc,
        'dim_weight_ratio': dim_ratio,
        'has_dimensions': float(has_dims),
        'billable_weight': billable,
        'billable_weight_ceil': float(billable_ceil),
        'ship_year': float(ship_year),
        'ship_month': float(ship_month),
        'months_since_start': float(months_since_start),
        'shipper_lat': s_lat,
        'shipper_lon': s_lon,
        'recipient_lat': r_lat,
        'recipient_lon': r_lon,
        'origin_dest_miles': miles,
    }
    for k, v in numeric.items():
        if k in out:
            out[k] = float(v)

    # One-hot encodings — silently no-op if the trained vocabulary doesn't include
    # the supplied value (matches scikit-learn's behavior on unseen categories)
    def _set_ohe(prefix: str, value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return
        col = f'{prefix}_{value}'
        if col in out:
            out[col] = 1.0

    _set_ohe('zone_clean',                _clean_zone(row.get('Pricing Zone')))
    _set_ohe('Service Type',              row.get('Service Type'))
    _set_ohe('Pay Type',                  row.get('Pay Type'))
    _set_ohe('Recipient State/Province',  row.get('Recipient State/Province'))
    _set_ohe('das_type',                  das_map.get(r_zip, 'NONE'))

    return pd.DataFrame([[out[c] for c in feature_cols]], columns=feature_cols)


def predict_shipment(row: dict) -> dict:
    """Audit-ready prediction for a single shipment.

    Input keys (raw FedEx column names; missing values default to 0/None):
        Original Weight (Pounds), Dimmed Height/Width/Length (cm),
        Pricing Zone, Service Type, Pay Type,
        Shipper Postal Code, Recipient Postal Code, Recipient State/Province,
        Invoice Month (yyyymm).
        Optional ground-truth fields for audit:
            Shipment DIM Flag (Y or N), Net Charge Billed Currency

    Output:
        dim_predicted:           bool
        dim_probability:         float in [0, 1] (isotonic-calibrated)
        dim_disagrees_with_fedex: bool | None
        charge_predicted:        float (USD)
        charge_lower_95:         float (USD)
        charge_upper_95:         float (USD)
        charge_actual:           float | None
        charge_outside_interval: bool | None
    """
    clf, reg, _, _, _ = _load_artifacts()
    X = transform_row(row)

    # Classifier — calibrated probabilities; positive class assumed to be label 1
    proba = clf.predict_proba(X)[0]
    cls_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
    p_dim = float(proba[cls_idx])
    dim_pred = bool(p_dim >= 0.5)

    # Regression — conformal returns (point_pred, interval) in log space
    log_point, log_interval = reg.predict_interval(X)
    charge_pred = float(np.expm1(log_point[0]))
    charge_lo = float(np.expm1(log_interval[0, 0, 0]))
    charge_hi = float(np.expm1(log_interval[0, 1, 0]))

    # Audit comparisons (only meaningful when ground truth is supplied)
    actual_flag = row.get('Shipment DIM Flag (Y or N)')
    if actual_flag is not None:
        fedex_says_dim = str(actual_flag).strip().upper() == 'Y'
        disagrees = bool(dim_pred != fedex_says_dim)
    else:
        disagrees = None

    charge_actual = row.get('Net Charge Billed Currency')
    if charge_actual is not None:
        charge_actual = float(charge_actual)
        outside = bool(charge_actual < charge_lo or charge_actual > charge_hi)
    else:
        outside = None

    return {
        'dim_predicted': dim_pred,
        'dim_probability': p_dim,
        'dim_disagrees_with_fedex': disagrees,
        'charge_predicted': charge_pred,
        'charge_lower_95': charge_lo,
        'charge_upper_95': charge_hi,
        'charge_actual': charge_actual,
        'charge_outside_interval': outside,
    }


if __name__ == '__main__':
    # End-to-end smoke test on a single row from the test parquet
    test_df = pd.read_parquet(ROOT / 'data' / 'test.parquet')
    sample = test_df.iloc[0]

    # The parquet is already feature-engineered. Build a raw-looking dict by
    # back-filling the few raw fields we need; numeric features are taken
    # straight from the parquet to verify the transform reproduces the model
    # input layout. Use any zone OHE that fires.
    def _picked(prefix):
        for c in test_df.columns:
            if c.startswith(prefix) and sample[c] == 1:
                return c[len(prefix):]
        return None

    raw = {
        'Original Weight (Pounds)': float(sample['Original Weight (Pounds)']),
        'Dimmed Height (cm)': float(sample['Dimmed Height (cm)']),
        'Dimmed Width (cm)':  float(sample['Dimmed Width (cm)']),
        'Dimmed Length (cm)': float(sample['Dimmed Length (cm)']),
        'Pricing Zone':            _picked('zone_clean_'),
        'Service Type':            _picked('Service Type_'),
        'Pay Type':                _picked('Pay Type_'),
        'Recipient State/Province': _picked('Recipient State/Province_'),
        'Shipper Postal Code': '76019',
        'Recipient Postal Code': '90210',
        'Invoice Month (yyyymm)': int(sample['ship_year'] * 100 + sample['ship_month']),
        'Shipment DIM Flag (Y or N)': 'Y' if sample['dim_flag'] == 1 else 'N',
        'Net Charge Billed Currency': float(sample['Net Charge Billed Currency']),
    }

    result = predict_shipment(raw)
    print(json.dumps(result, indent=2, default=str))
