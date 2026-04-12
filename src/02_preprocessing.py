import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

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
    'Shipper Name',
    'Shipper Company Name',
    'Shipper Address',
    'Shipper City',
    'Shipper State/Province',
    'Shipper Country/Territory',
    'Shipper Postal Code',
    'Proof Of Delivery Recipient',      # capital O in 2years.csv
    'Recipient Name',
    'Recipient Company Name',
    'Recipient Address',
    'Recipient City',
    'Recipient State/Province',
    'Recipient Country/Territory',
    'Recipient Postal Code',
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
df = pd.get_dummies(df,columns = ['Service Type', 'Pay Type', 'zone_clean'], drop_first=False)

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
feature_cols = [c for c in train_df.columns if c not in ['dim_flag', 'log_net_charge','Net Charge Billed Currency']]

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
print(f"Val: {len(val_df):,} rows")
print(f"Test: {len(test_df):,} rows")
print(f"Features: {len(feature_cols)}")
print(f"DIM rate - Train: {train_df['dim_flag'].mean():.3f}")
print(f"DIM rate - Val: {val_df['dim_flag'].mean():.3f}")
print(f"DIM rate - Test: {test_df['dim_flag'].mean():.3f}")
#dim rates should all be very close to 0.322, confirming stratification worked