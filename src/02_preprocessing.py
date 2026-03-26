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
    (df['Package Dimensions Height (inches)'] == 0) &
    (df['Package Dimensions Width (inches)'] == 0) &
    (df['Package Dimensions Length (inches)'] == 0) 
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

    # Redundant with Service Type                                                                                                                                      
    'Service Description',

    #LEAKED FEATURE, critical to remove
    'Shipment Rated Weight (Pounds)'
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