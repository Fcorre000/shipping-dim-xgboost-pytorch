"""Population Stability Index drift check.

Tiny module the dashboard backend can call monthly:

    from src.drift import check_drift
    report = check_drift('data/train.parquet', 'data/recent_month.parquet')
    # {feature_name: psi_value, ...}

Convention (Siddiqi 2006):

    PSI < 0.10  : no significant change
    0.10 ≤ PSI < 0.25 : moderate shift — investigate
    PSI ≥ 0.25  : major shift — re-train candidate

No scheduler, no alerts. The caller decides what to do with the dict.

CLI sanity check:

    python src/drift.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Targets and label columns the parquets carry — never compare drift on these
NON_FEATURE_COLS = {
    'dim_flag', 'log_net_charge', 'Net Charge Billed Currency',
    'log_base_charge', 'log_misc_charge',
}


def population_stability_index(reference: np.ndarray, current: np.ndarray,
                                bins: int = 10, eps: float = 1e-4) -> float:
    """Standard PSI on a single 1-D feature.

    Bin edges come from the reference quantiles so the reference itself sits
    near-uniformly across bins. The current sample is then re-bucketed against
    those edges; PSI sums (cur_pct − ref_pct) · ln(cur_pct / ref_pct).

    `eps` floor avoids `log(0)` and infinite PSI when a bucket is empty in one
    sample.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current,   dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float('nan')

    # Quantile edges from reference; deduplicate to handle zero-variance / binary cols
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        # Degenerate (e.g. one-hot column that's almost all 0). Compare proportions directly.
        ref_p = np.mean(ref > 0)
        cur_p = np.mean(cur > 0)
        ref_p = max(ref_p, eps); cur_p = max(cur_p, eps)
        return float((cur_p - ref_p) * np.log(cur_p / ref_p)
                     + ((1 - cur_p) - (1 - ref_p)) * np.log((1 - cur_p) / (1 - ref_p)))

    edges[0]  = -np.inf
    edges[-1] =  np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = np.maximum(ref_counts / ref_counts.sum(), eps)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_drift(reference_parquet: str, current_parquet: str,
                bins: int = 10) -> dict[str, float]:
    """Run PSI on every feature column shared between the two parquet files."""
    ref_df = pd.read_parquet(reference_parquet)
    cur_df = pd.read_parquet(current_parquet)
    shared = [c for c in ref_df.columns
              if c in cur_df.columns and c not in NON_FEATURE_COLS]
    return {c: population_stability_index(ref_df[c].values, cur_df[c].values, bins=bins)
            for c in shared}


def summarize(report: dict[str, float], top_n: int = 10) -> pd.DataFrame:
    """Bucket each feature into none/moderate/major and return the top-N PSIs."""
    df = pd.DataFrame({'feature': list(report.keys()), 'psi': list(report.values())})
    df = df.dropna().sort_values('psi', ascending=False)

    def _bucket(p):
        if p < 0.10:
            return 'none'
        if p < 0.25:
            return 'moderate'
        return 'major'

    df['drift'] = df['psi'].apply(_bucket)
    return df.head(top_n).reset_index(drop=True)


if __name__ == '__main__':
    # Sanity check: train vs test from the same split should have near-zero PSI
    # everywhere (stratified random split, no temporal drift).
    train = ROOT / 'data' / 'train.parquet'
    test  = ROOT / 'data' / 'test.parquet'
    if not train.exists() or not test.exists():
        raise SystemExit('run src/02_preprocessing.py first to generate the parquets')

    report = check_drift(str(train), str(test))
    top = summarize(report, top_n=10)

    print(f'features compared: {len(report)}')
    print(f'mean PSI:   {np.nanmean(list(report.values())):.4f}')
    print(f'max  PSI:   {np.nanmax(list(report.values())):.4f}')
    print(f'major  (>=0.25): {sum(v >= 0.25 for v in report.values())}')
    print(f'moderate (>=0.10): {sum(0.10 <= v < 0.25 for v in report.values())}')
    print()
    print('=== top 10 by PSI ===')
    print(top.to_string(index=False))

    out_path = ROOT / 'models' / 'drift_train_vs_test.json'
    with open(out_path, 'w') as f:
        json.dump({k: round(v, 6) if np.isfinite(v) else None
                   for k, v in report.items()}, f, indent=2)
    print(f'\nwrote {out_path}')
