# `predict_shipment` — Inference API Contract

Single-shipment inference for the `dim-risk-engine` FastAPI service. One Python
call (`src.predict.predict_shipment(row)`) maps to one HTTP request: same JSON
shape on the wire, same dict in Python.

## Request — JSON object

| Key | Type | Required | Notes |
|---|---|---|---|
| `Original Weight (Pounds)` | number | yes | Actual scale weight |
| `Dimmed Height (cm)` | number | yes | Use 0 if undimensioned |
| `Dimmed Width (cm)` | number | yes | Use 0 if undimensioned |
| `Dimmed Length (cm)` | number | yes | Use 0 if undimensioned |
| `Pricing Zone` | string\|number | yes | `"02"` … `"08"` (or numeric `2`-`8`); other codes bucket to `Other` |
| `Service Type` | string | yes | FedEx service code (`Ground`, `Home Delivery`, `Express`, …) |
| `Pay Type` | string | yes | `Bill_Sender_Prepaid`, `Bill_Third_Party`, `Bill_Recipient` |
| `Shipper Postal Code` | string | yes | 5-digit US ZIP — drives `origin_dest_miles` |
| `Recipient Postal Code` | string | yes | 5-digit US ZIP — also drives `das_type` |
| `Recipient State/Province` | string | yes | 2-letter US state abbreviation |
| `Invoice Month (yyyymm)` | integer | yes | e.g. `202604`; drives `months_since_start` |
| `Shipment DIM Flag (Y or N)` | `"Y"` \| `"N"` | optional | If supplied, response includes `dim_disagrees_with_fedex` |
| `Net Charge Billed Currency` | number | optional | If supplied, response includes `charge_outside_interval` |

Unknown one-hot categories (e.g. an unseen `Service Type`) silently produce all-zero
indicators — mirroring scikit-learn's `handle_unknown='ignore'` behavior. The model
still returns a prediction; it just degrades to the column means for that feature.

### Example request

```json
{
  "Original Weight (Pounds)": 18.4,
  "Dimmed Height (cm)": 45.0,
  "Dimmed Width (cm)": 50.0,
  "Dimmed Length (cm)": 60.0,
  "Pricing Zone": "05",
  "Service Type": "Ground",
  "Pay Type": "Bill_Sender_Prepaid",
  "Shipper Postal Code": "76019",
  "Recipient Postal Code": "90210",
  "Recipient State/Province": "CA",
  "Invoice Month (yyyymm)": 202604,
  "Shipment DIM Flag (Y or N)": "Y",
  "Net Charge Billed Currency": 127.23
}
```

## Response — JSON object

| Key | Type | Notes |
|---|---|---|
| `dim_predicted` | bool | `True` iff isotonic-calibrated prob ≥ 0.5 |
| `dim_probability` | float in `[0, 1]` | Calibrated P(DIM flagged) |
| `dim_disagrees_with_fedex` | bool \| `null` | `null` when ground-truth flag not supplied |
| `charge_predicted` | float (USD) | Conformal point prediction (`np.expm1` of log-space) |
| `charge_lower_95` | float (USD) | Lower bound of 95% prediction interval |
| `charge_upper_95` | float (USD) | Upper bound of 95% prediction interval |
| `charge_actual` | float \| `null` | Echo of ground-truth charge when supplied |
| `charge_outside_interval` | bool \| `null` | `null` when ground-truth charge not supplied |
| `anomaly_score` | float in `[0, 1]` \| `null` | Mean of IsolationForest + autoencoder percentile ranks |
| `anomaly_flagged` | bool \| `null` | `True` iff `anomaly_score >= 0.95` (Phase 4 calibrated threshold) |
| `review_recommended` | bool | `True` if **any** of the three audit signals fired |
| `review_priority` | `"high"` \| `"medium"` \| `"low"` | `high` ≥ 2 signals, `medium` = 1, `low` = 0 |

The three audit signals fed into `review_priority`:

1. `dim_disagrees_with_fedex == True`
2. `charge_outside_interval == True`
3. `anomaly_flagged == True`

### Example response

```json
{
  "dim_predicted": true,
  "dim_probability": 0.998,
  "dim_disagrees_with_fedex": false,
  "charge_predicted": 53.34,
  "charge_lower_95": 45.47,
  "charge_upper_95": 62.55,
  "charge_actual": 127.23,
  "charge_outside_interval": true,
  "anomaly_score": 0.984,
  "anomaly_flagged": true,
  "review_recommended": true,
  "review_priority": "high"
}
```

## Errors

The function does not raise on missing optional fields — they map to `null` in
the audit-comparison keys. It **will** raise:

* `FileNotFoundError` — model artifact missing (deployment/packaging issue)
* `KeyError` — when a required input field above is absent from the request
* `ValueError` — when a numeric field is non-numeric

The FastAPI wrapper should translate these to `500`, `400`, and `422` respectively.

## Performance

| Stage | Wall-clock per row (M1 Pro CPU) |
|---|---|
| `transform_row` (one-hot + haversine) | ~0.6 ms |
| Classifier (isotonic-calibrated XGBoost v2) | ~0.05 ms |
| Regressor (conformal-wrapped XGBoost v2) | ~0.10 ms |
| Anomaly (IsolationForest + 105→16 AE) | ~1.2 ms |
| **End-to-end** | **~2 ms** |

Throughput on the same hardware: ~500 req/s single-threaded. The FastAPI service
can fan out across workers but a single process is plenty for the dashboard's
audit queue cadence.

## Versioning

The contract is tied to the v2 feature set (105 columns; see
`models/feature_columns.json`). When the feature set changes the contract
version should bump and `feature_columns.json` should ship in the same release
artifact as the pickles.
