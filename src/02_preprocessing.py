import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from math import radians, cos, sin, asin, sqrt

try:
    import pgeocode
    _PGEOCODE_AVAILABLE = True
except ImportError:
    _PGEOCODE_AVAILABLE = False
    print("WARNING: pgeocode not installed — geographic features will be NaN. Run: pip install pgeocode")

_NOMI = None

def _get_nomi():
    global _NOMI
    if _NOMI is None:
        _NOMI = pgeocode.Nominatim('us')
    return _NOMI

def _haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    a = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2
    return 2 * R * asin(sqrt(max(0.0, min(1.0, a))))

def _build_zip_lookup(zips: pd.Series, cache_path: str = 'data/zip_lookup.parquet') -> dict:
    """Return {zip_str: {'latitude': float, 'longitude': float}} with parquet cache."""
    unique_zips = [str(z).zfill(5) for z in zips.dropna().unique()]
    if os.path.exists(cache_path):
        cached_df = pd.read_parquet(cache_path)
        cached = {row['zip']: {'latitude': row['latitude'], 'longitude': row['longitude']}
                  for _, row in cached_df.iterrows()}
    else:
        cached = {}
    missing = [z for z in unique_zips if z not in cached]
    if missing and _PGEOCODE_AVAILABLE:
        nomi = _get_nomi()
        rows = []
        for z in missing:
            r = nomi.query_postal_code(z)
            lat = float(r.latitude) if pd.notna(r.latitude) else float('nan')
            lon = float(r.longitude) if pd.notna(r.longitude) else float('nan')
            cached[z] = {'latitude': lat, 'longitude': lon}
            rows.append({'zip': z, 'latitude': lat, 'longitude': lon})
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        new_df = pd.DataFrame(rows)
        if os.path.exists(cache_path):
            existing = pd.read_parquet(cache_path)
            pd.concat([existing, new_df], ignore_index=True).to_parquet(cache_path, index=False)
        else:
            new_df.to_parquet(cache_path, index=False)
    return cached

df = pd.read_csv('2years.csv', low_memory=False)


##### ROW REMOVAL ###########################################

#keeps only rows where DIM flag column is not null, not real shipments can't be used for training
# drops 270 NonTrans rows (non-transport billing entries with blank DIM flag) per EDA
df = df[df['Shipment DIM Flag (Y or N)'].notna()]
# cap extreme right tail (~0.50% of rows, 285 shipments above $200 in 2years.csv).
# removing this heavy tail was worth ~10% XGBoost gain on the previous dataset
# and the new distribution has the same skew (~23.1), so the cap still applies.
df = df[df['Net Charge Billed Currency'] <= 200]
# "~" inverts the condition, so you keep everything EXCEPT DIM=Y rows that have all three dimensions at zero.
# In 2years.csv this filter is a no-op (0 matches) but kept as a guard against future data errors.
df = df[~(
    (df['Shipment DIM Flag (Y or N)'] == 'Y') &
    (df['Dimmed Height (cm)'] == 0) &
    (df['Dimmed Width (cm)'] == 0) &
    (df['Dimmed Length (cm)'] == 0)
)]

#keep only domestic shipments, DIM divisor is 139 for domestic and 169 for intl
#keeping international would confuse the model (2years.csv has 176 intl rows)
df = df[df['Domestic/Intl'] ==  'Domestic']

##### DECOMPOSED REGRESSION TARGETS (Phase 1.3) ###########################################
# Computed here before the charge columns are dropped from the feature space.
# These become additional target columns saved to parquet — never used as features.
# log_base_charge: base transport cost (largest component of net charge)
# log_misc_charge: accessorial/surcharges (residential, DAS, etc.)
df['log_base_charge'] = np.log1p(df['Shipment Freight Charge Amount USD'].fillna(0))
df['log_misc_charge'] = np.log1p(df['Shipment Miscellaneous Charge USD'].clip(lower=0).fillna(0))

###### COLUMN DROPPING ###########################################

#need to drop columns that are high-null, zero-variance, identifiers or leaked
cols_to_drop = [
    #high null columns
    'Department Number',
    'Customs Value Currency  Code',     # double space in 2years.csv
    'Recipient Original State/Province',

    #zero-variance columns(same value in every row)
    'Weight Type Code',         #always 'lb'
    'Billed Currency Code',     #always 'USD'
    'Exchange Rate to USD',     #always 1
    'Domestic/Intl',            #always 'Domestic' after filtering

    # new columns in 2years.csv — zero/low value for domestic shipments
    'Shipment Declared Value Amount',
    'Customs Value',
    'Postal Identification Number',

    #Identifier columns(unique per shipment, no predictive value)
    'Shipment Tracking Number',
    'Invoice Number',
    'Master Tracking Number',
    'Payer Account',
    'OPCO',
    # 'Invoice Month (yyyymm)' is NOT dropped here — it is parsed below into
    # ship_year / ship_month / months_since_start so the regressor can learn
    # monthly fuel-surcharge and annual rate-card changes (±40% cost swing
    # across the 25-month window).
    'Reference Notes Line 1',
    'Reference Notes Line 2',
    'Reference Notes Line 3',
    'PO Number',
    'Customer Order Number',
    'Invoice Date (mm/dd/yyyy)',
    'Shipment Delivery Time',           # renamed in 2years.csv (no "(12 Hours)")
    'Package Type',

    # Address & name columns
    # NOTE: Shipper Postal Code, Recipient Postal Code, Recipient State/Province
    # are intentionally kept — they are engineered into geographic features below.
    'Shipper Name',
    'Shipper Company Name',
    'Shipper Address',
    'Shipper City',
    'Shipper State/Province',
    'Shipper Country/Territory',
    'Proof Of Delivery Recipient',      # capital O in 2years.csv
    'Recipient Name',
    'Recipient Company Name',
    'Recipient Address',
    'Recipient City',
    'Recipient Country/Territory',
    'Recipient Original Address',
    'Recipient Original City',
    'Recipient Original Postal Code',
    'Recipient Original Country/Territory',
    'Shipment Delivery Date (mm/dd/yyyy)',
    'Shipment Date (mm/dd/yyyy)',

    # Redundant with Service Type
    'Service Description',

    #LEAKED FEATURES, critical to remove
    'Shipment Rated Weight (Pounds)',    # space before paren in 2years.csv
    'Shipment Freight Charge Amount USD',
    'Shipment Miscellaneous Charge USD',
    'Shipment Duty and Tax Charge USD',
    'Shipment Discount Amount USD',
    'Net Charge Amount USD',
    'Shipment Freight Charge Billed Currency',
    'Shipment Miscellaneous Charge Billed Currency',
    'Shipment Duty And Tax Charge Billed Currency',  # capital A in 2years.csv
    'Shipment Discount Billed Currency',
]

#drop columns listed above, if column isn't there or spelling mistake, won't crash
df = df.drop(columns=cols_to_drop, errors='ignore')

##### TARGET VARIABLE ENCODING ###########################################

#Task 1 target, converts 'Y'->1 and 'N'->0.
#Class balance in 2years.csv: 68.0% N / 32.0% Y (2.12:1). More imbalanced than the
#previous dataset (1.46:1), so downstream classifiers should lean on class_weight='balanced'.
df['dim_flag'] = (df['Shipment DIM Flag (Y or N)'] == "Y").astype(int)

#Task 2 target, computes log transform of net charge because of high skewness (~23.1 in 2years.csv)
#without this model would focus too much on extreme charges
df['log_net_charge'] = np.log1p(df['Net Charge Billed Currency'])

##### FEATURE ENGINEERING ###########################################

# 2years.csv stores dimensions in cm; convert to inches so the FedEx domestic
# DIM divisor of 139 (in³/lb) stays correct
CM_PER_IN = 2.54
height = df['Dimmed Height (cm)'] / CM_PER_IN
width  = df['Dimmed Width (cm)']  / CM_PER_IN
length = df['Dimmed Length (cm)'] / CM_PER_IN
weight = df['Original Weight (Pounds)']

df['volume'] = height * width * length
df['dim_weight_calculator'] = df['volume'] / 139
df['dim_weight_ratio'] = df['dim_weight_calculator'] / weight
df['has_dimensions'] = ((height > 0) & (width > 0) & (length > 0)).astype(int)

# Billable weight is max(actual_weight, dim_weight) — this is the quantity FedEx
# actually prices on, so giving the regressor this explicitly is a huge signal.
# Ceiling to the next pound matches FedEx's rate-card rounding behavior.
df['billable_weight'] = np.maximum(weight, df['dim_weight_calculator'])
df['billable_weight_ceil'] = np.ceil(df['billable_weight'])

# Time features — FedEx rate cards update annually, fuel surcharges reset monthly,
# and holiday season adds surcharges. Without these the regressor can't explain
# the ~40% mean-charge swing across the 25-month window. Parse Invoice Month (yyyymm)
# (format: 202407 = July 2024) into absolute and seasonal components, then drop the raw col.
df['ship_year'] = df['Invoice Month (yyyymm)'] // 100
df['ship_month'] = df['Invoice Month (yyyymm)'] % 100
# Linear time index from the first month in the dataset (Apr 2024 → 4, ..., Apr 2026 → 28).
# Lets trees split cleanly on rate-card transitions instead of re-learning year+month jointly.
df['months_since_start'] = (df['ship_year'] - 2024) * 12 + df['ship_month']
df = df.drop(columns=['Invoice Month (yyyymm)'])

# Pieces In Shipment is 99.2% equal to 1 in 2years.csv (near zero-variance);
# drop so the model isn't burning split capacity on noise.
df = df.drop(columns=['Pieces In Shipment'], errors='ignore')

##### GEOGRAPHIC FEATURES (Phase 1.1) ###########################################
# Recover zip-code-derived distance and lat/lon signal that was previously dropped.
# zip_lookup.parquet caches pgeocode results so reruns don't re-query the GeoNames DB.
#
# The raw data stores many postal codes as 9-digit ZIP+4 barcodes (e.g., 894415600).
# We normalize to the first 5 digits before lookup.

def _normalize_zip(z):
    digits = ''.join(c for c in str(z) if c.isdigit())
    return digits[:5] if len(digits) >= 5 else digits.zfill(5)

shipper_zips   = df['Shipper Postal Code'].apply(_normalize_zip)
recipient_zips = df['Recipient Postal Code'].apply(_normalize_zip)
all_zips = pd.concat([shipper_zips, recipient_zips])

zip_map = _build_zip_lookup(all_zips, cache_path='data/zip_lookup.parquet')

def _lat(z):
    return zip_map.get(_normalize_zip(z), {}).get('latitude', float('nan'))

def _lon(z):
    return zip_map.get(_normalize_zip(z), {}).get('longitude', float('nan'))

df['shipper_lat']   = df['Shipper Postal Code'].apply(_lat)
df['shipper_lon']   = df['Shipper Postal Code'].apply(_lon)
df['recipient_lat'] = df['Recipient Postal Code'].apply(_lat)
df['recipient_lon'] = df['Recipient Postal Code'].apply(_lon)

def _dist_row(row):
    vals = [row['shipper_lat'], row['shipper_lon'], row['recipient_lat'], row['recipient_lon']]
    if any(v != v for v in vals):   # NaN check without importing math.isnan
        return float('nan')
    return _haversine_miles(row['shipper_lat'], row['shipper_lon'],
                             row['recipient_lat'], row['recipient_lon'])

df['origin_dest_miles'] = df.apply(_dist_row, axis=1)

##### DAS SURCHARGE FLAG (Phase 1.2) ###########################################
# FedEx DAS/EDAS zip list must be created manually as data/das_zips.csv
# (columns: zip, das_type where das_type in {NONE, DAS, EDAS, REMOTE}).
# If the file is absent all rows get das_type='NONE' and the columns are still
# one-hot encoded, so the feature matrix shape is consistent once the file is added.

DAS_PATH = 'data/das_zips.csv'
if os.path.exists(DAS_PATH):
    das_lookup = pd.read_csv(DAS_PATH, dtype={'zip': str})
    das_map = dict(zip(das_lookup['zip'].str.zfill(5), das_lookup['das_type']))
    print(f"DAS lookup loaded: {len(das_map):,} zips")
else:
    das_map = {}
    print("WARNING: data/das_zips.csv not found — das_type='NONE' for all rows. "
          "Download the FedEx DAS zip list and save as data/das_zips.csv to activate this feature.")

df['das_type'] = (df['Recipient Postal Code'].apply(_normalize_zip)
                  .map(das_map).fillna('NONE'))

# Raw postal code columns have been fully encoded into numeric features — drop them now.
df = df.drop(columns=['Shipper Postal Code', 'Recipient Postal Code'], errors='ignore')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

##### PRICING ZONE CLEANUP ###########################################

#Convert zone numbers '2' and '02' to just '2'
#if zones are above 50 or letters like 'D', group to other
def clean_zone(z):
    try:
        num = int(float(z))
        if num > 50 or num < 1:
            return 'Other'
        return f'{num:02d}'
    except (ValueError, TypeError):
        if str(z).upper() in ('D', 'A', 'C', 'N'):
            return 'Other'
        return 'Other'

df['zone_clean'] = df['Pricing Zone'].apply(clean_zone)  

###### ONE-HOT ENCODING ###########################################

#convert values to 0 or 1
df = pd.get_dummies(df, columns=['Service Type', 'Pay Type', 'zone_clean',
                                  'Recipient State/Province', 'das_type'], drop_first=False)

#drop original DIM flag column because we created our own, and drop other non-numeric columns
df = df.drop(columns=['Shipment DIM Flag (Y or N)', 'Pricing Zone'], errors='ignore')

##### TRAIN/VAL/TEST SPLIT ###########################################

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, 
    stratify=df['dim_flag'] #ensures the 32.0/68.0 DIM ratio (2years.csv) is preserved in both halves
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42,
    stratify=temp_df['dim_flag']
)
    

##### SAVE PARQUET FILES (TWO VERSIONS) ###########################################

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

#Version 1- Unscaled(for tree models like XGBoost):
train_df.to_parquet('data/train.parquet', index=False)
val_df.to_parquet('data/val.parquet', index=False)
test_df.to_parquet('data/test.parquet', index=False)

#Version 2- Scaled(for pytorch neural networks)
target_cols = ['dim_flag', 'log_net_charge', 'Net Charge Billed Currency',
               'log_base_charge', 'log_misc_charge']
feature_cols = [c for c in train_df.columns if c not in target_cols]

scaler = StandardScaler()
train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

"""
fit_transform on train, but only transform on val/test.
- fit_transform calculates the mean and standard deviation from the training data, then applies the
formula: (value - mean) / std
- transform uses the same mean and std from training to scale val/test. If you re-fit on val/test,
you'd leak information about the val/test distributions into the scaling.
"""
train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

train_scaled.to_parquet('data/train_scaled.parquet', index=False)
val_scaled.to_parquet('data/val_scaled.parquet', index=False)
test_scaled.to_parquet('data/test_scaled.parquet', index=False)

joblib.dump(scaler, 'models/preprocessor.pkl')

##### PRINT SUMMARY ###########################################

print(f"Train: {len(train_df):,} rows")
print(f"Val:   {len(val_df):,} rows")
print(f"Test:  {len(test_df):,} rows")
print(f"Features: {len(feature_cols)}  (was 42 before Phase 1)")
print(f"Targets: {target_cols}")
print(f"DIM rate - Train: {train_df['dim_flag'].mean():.3f}")
print(f"DIM rate - Val:   {val_df['dim_flag'].mean():.3f}")
print(f"DIM rate - Test:  {test_df['dim_flag'].mean():.3f}")
# dim rates should all be very close to 0.322, confirming stratification worked
geo_cols = ['origin_dest_miles', 'shipper_lat', 'shipper_lon', 'recipient_lat', 'recipient_lon']
print(f"\nGeographic feature non-zero rate (train):")
for col in geo_cols:
    nonzero = (train_df[col] != 0).mean()
    print(f"  {col}: {nonzero:.1%} non-zero")