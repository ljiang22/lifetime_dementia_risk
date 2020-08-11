"""
--------------------------------------------------------------------------------------------------------------------------
Load and preprocess the National Alzheimer Coordinating Center (NACC) Uniform Data Sets (UDS)
--------------------------------------------------------------------------------------------------------------------------

"""


import numpy as np
import pandas as pd
from pandas import set_option

# Define the display parameters
set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
pd.options.mode.chained_assignment = None

# Define the file name and path of the input files
input_file = './raw data/UDS_A1.csv'
data_input_ori = pd.read_csv(input_file)
data_keys = data_input_ori.keys()

data_input = data_input_ori[['UDS_A1SUBDEMODATA ID', 'Subject', 'LIVSIT','INDEPEND', 'RESIDENC', 'MARISTAT']]
M, N = data_input.shape
print(M, N)

# Preprocessing missing value

# 1 = Lives alone
# 2 = Lives with spouse or partner
# 3 = Lives with relative or friend
# 4 = Lives with group
# 5 = Other
# 9 = Unknown
idx = data_input['Subject']
hwr = []
for i in range(M-1):
    tmp1 = idx[i]
    tmp2 = idx[i+1]

    # Living situation - LIVSIT
    if np.isnan(data_input.LIVSIT[i]):
        data_input.LIVSIT[i] = 9

    # Level of independence
    if np.isnan(data_input.INDEPEND[i]):
        data_input.INDEPEND[i] = 9

    # Type of residence
    if np.isnan(data_input.RESIDENC[i]):
        data_input.RESIDENC[i] = 9

    # Marital status
    if np.isnan(data_input.MARISTAT[i]):
        data_input.MARISTAT[i] = 9

data_input.to_csv('./raw data_edit/UDS_A1_edit.csv')

# Feature engineering on family history
input_file = './raw data/UDS_A3.csv'    # The file name of an input file
data_input_ori = pd.read_csv(input_file)

data_keys = data_input_ori.keys()
print(data_keys[0:20])
print(data_keys[20:40])
print(data_keys[40:60])
print(data_keys[60:80])
print(data_keys[80:100])
print(data_keys[100:120])
print(data_keys[120:200])
print(data_keys[200:260])
print(data_keys[260:320])
print(data_keys[320:])

data_input = data_input_ori[['UDS_A3SBFMHSTDATA ID', 'Subject', 'MOMDEM',  'DADDEM', 'SIB1DEM', 'SIB2DEM', 'SIB3DEM', 'SIB4DEM',  'SIB5DEM', 'SIB6DEM',
        'SIB7DEM', 'SIB8DEM', 'SIB9DEM', 'SIB10DEM', 'SIB11DEM', 'SIB12DEM']]
M, N = data_input.shape
print(M, N)

idx = data_input['Subject']
dem1 = []
dem2 = []
dem3 = []
dem4 = []
dem5 = []
dem_idx = []
dem_all = 0.0
j = 0
ni = 4081  # the start number of the last patient
for i in range(M-1):
    tmp1 = idx[i]
    tmp2 = idx[i+1]

    if tmp1 == tmp2:
        if data_input.MOMDEM[i] == 1:
            dem1.append(data_input.MOMDEM[i])
        if data_input.DADDEM[i] == 1:
            dem2.append(data_input.DADDEM[i])
        if data_input.SIB1DEM[i] == 1:
            dem3.append(data_input.SIB1DEM[i])
        if data_input.SIB2DEM[i] == 1:
            dem4.append(data_input.SIB2DEM[i])
        if data_input.SIB3DEM[i] == 1:
            dem5.append(data_input.SIB3DEM[i])
    else:
        if data_input.MOMDEM[i] == 1:
            dem1.append(data_input.MOMDEM[i])
        if data_input.DADDEM[i] == 1:
            dem2.append(data_input.DADDEM[i])
        if data_input.SIB1DEM[i] == 1:
            dem3.append(data_input.SIB1DEM[i])
        if data_input.SIB2DEM[i] == 1:
            dem4.append(data_input.SIB2DEM[i])
        if data_input.SIB3DEM[i] == 1:
            dem5.append(data_input.SIB3DEM[i])

        if len(dem1) > 0:
            dem_all = dem_all + 1.0
        if len(dem2) > 0:
            dem_all = dem_all + 1.0
        if len(dem3) > 0:
            dem_all = dem_all + 1.0
        if len(dem4) > 0:
            dem_all = dem_all + 1.0
        if len(dem5) > 0:
            dem_all = dem_all + 1.0

        #print(tmp1, tmp2, dem1, dem2, dem3, dem4, dem5, dem_all)

        if len(dem1) > 0 or len(dem2) > 0 or len(dem3) > 0 or len(dem4) > 0 or len(dem5) > 0:
            dem = dem_all
        else:
            dem = 0.0
        dem_idx.append(dem)
        dem_all = 0.0
        #print(i, tmp1, tmp2, dem)

        dem1 = []
        dem2 = []
        dem3 = []
        dem4 = []
        dem5 = []

    if i == M-2:
        if data_input.MOMDEM[i+1] == 1:
            dem1.append(data_input.MOMDEM[i])
        if data_input.DADDEM[i+1] == 1:
            dem2.append(data_input.DADDEM[i])
        if data_input.SIB1DEM[i+1] == 1:
            dem3.append(data_input.SIB1DEM[i])
        if data_input.SIB2DEM[i+1] == 1:
            dem4.append(data_input.SIB2DEM[i])
        if data_input.SIB3DEM[i+1] == 1:
            dem5.append(data_input.SIB3DEM[i])

        if len(dem1) > 0:
            dem_all = dem_all + 1.0
        if len(dem2) > 0:
            dem_all = dem_all + 1.0
        if len(dem3) > 0:
            dem_all = dem_all + 1.0
        if len(dem4) > 0:
            dem_all = dem_all + 1.0
        if len(dem5) > 0:
            dem_all = dem_all + 1.0

        #print(tmp1, tmp2, dem1, dem2, dem3, dem4, dem5, dem_all)
        if len(dem1) > 0 or len(dem2) > 0 or len(dem3) > 0 or len(dem4) > 0 or len(dem5) > 0:
            dem = dem_all
        else:
            dem = 0.0
        dem_idx.append(dem)
        dem_all = 0.0

        #print(i, tmp1, tmp2, dem)
    #print(i, tmp1, tmp2)

dem_idx_tmp = [0] * (M)
#print(dem_idx_tmp)
data_input['dem_idx'] = dem_idx_tmp
j= 0
for i in range(M-1):
    tmp1 = idx[i]
    tmp2 = idx[i+1]

    if j < len(dem_idx)-1:
        if tmp1 == tmp2:
            data_input.dem_idx[i] = dem_idx[j]
            #print(i, tmp1, tmp2, data_input.dem_idx[i], j, dem_idx[j])
        else:
            data_input.dem_idx[i] = dem_idx[j]
            #print(i, tmp1, tmp2, data_input.dem_idx[i], j, dem_idx[j])
            j = j + 1
    else:
        #print(j)
        data_input.dem_idx[i] = dem_idx[j]
        #print(i, tmp1, tmp2, data_input.dem_idx[i], j, dem_idx[j])

    if i == M-2:
        data_input.dem_idx[i + 1] = dem_idx[j]
        #print(i, tmp1, tmp2, data_input.dem_idx[i], j, dem_idx[j])

data_input.to_csv('./raw data_edit/UDS_A3_edit.csv')