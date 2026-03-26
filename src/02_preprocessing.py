import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

df = pd.read_excel('notebooks/FedEx_ShipmentDetail.xlsx')


##### ROW REMOVAL ###########################################

#keeps only rows where DIM flag column is not null, not real shipments can't be used for training
df = df[df['Shipment DIM Flag (Y or N)'].notna()]

# "~" inverts the condition, so you keep everything EXCEPT DIM=Y rows that have all three dimensions at zero
df = df[~(
    (df['Shipment DIM Flag (Y or N)'] == 'Y') &
    (df['Dimmed Height (in)'] == 0) &
    (df['Dimmed Width (in)'] == 0) &
    (df['Dimmed Length (in)'] == 0) 
)]

#keep only domestic shipments, DIM divisor is 139 for domestic and 169 for intl
#keeping international would confuse the model
df = df[df['Domestic/Intl'] ==  'Domestic']

###### COLUMN DROPPING ###########################################

#need to drop columns that are high-null, zero-variance, identifiers or leaked
cols_to_drop = [
    #high null columns
    'Department Number',
    'Customs Value Currency Code',
    'Recipient Original State/Province',

    #zero-variance columns(same value in every row)
    'Weight Type Code',         #always 'lb'
    'Billed Currency Code',     #always 'USD'
    'Exchange Rate to USD',     #always 1
    'Domestic/Intl',            #always 'Domestic' after filtering

    #Identifier columns(unique per shipment, no predictive value)
    'Tracking Number',
    'Shipment Tracking Number',                                                                                                                                        
    'Invoice Number',                                                                                                                                                
    'Master Tracking Number',
    'Payer Account',
    'OPCO',
    'Invoice Month (yyyymm)',
    'Reference Notes Line 1',                                                                                                                                          
    'Reference Notes Line 2',
    'Reference Notes Line 3',                                                                                                                                          
    'PO Number',                                                                                                                                                     
    'Customer Order Number',
    'Invoice Date (mm/dd/yyyy)',
    'Shipment Delivery Time (12 Hours)',
    'Package Type',

    # Address & name columns                                                                                                                                           
    'Shipper Name',
    'Shipper Company Name',                                                                                                                                            
    'Shipper Address',                                                                                                                                               
    'Shipper City',
    'Shipper State/Province',
    'Shipper Country/Territory',
    'Shipper Postal Code',
    'Proof of Delivery Recipient',                                                                                                                                     
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

    #LEAKED FEATURE, critical to remove
    'Shipment Rated Weight(Pounds)'
]

#drop columns listed above, if column isn't there or spelling mistake, won't crash
df = df.drop(columns=cols_to_drop, errors='ignore')

##### TARGET VARIABLE ENCODING ###########################################

#Task 1 target, converts 'Y'->1 and 'N'->0
df['dim_flag'] = (df['Shipment DIM Flag (Y or N)'] == "Y").astype(int)

#Task 2 target, computes log transform of net charge because of high skewness(23.2)
#without this model would focus too much on extreme charges
df['log_net_charge'] = np.log1p(df['Net Charge Billed Currency'])

##### FEATURE ENGINEERING ###########################################

height = df['Package Dimensions Height (inches)']
width = df['Package Dimensions Width (inches)']
length = df['Package Dimensions Length (inches)']
weight = df['Original Weight (Pounds)']

df['volume'] = height * width * length
df['dim_weight_calculator'] = df['volume'] / 139
df['dim_weight_ratio'] = df['dim_weight_calculator'] / weight
df['cost_per_pound'] = df["Net Charge Billed Currency"] / weight
df['has_dimensions'] = ((height > 0) & (width > 0) & (length > 0)).astype(int)

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
    stratify=df['dim_flag'] #ensures the 40.7/59.3 DIM ratio is preserved in both halves
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
#dim rates should all be very close to 0.407, confirming stratification worked