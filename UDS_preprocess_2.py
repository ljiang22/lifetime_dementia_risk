"""
--------------------------------------------------------------------------------------------------------------------------
Load and preprocess the National Alzheimer Coordinating Center (NACC) Uniform Data Sets (UDS)
--------------------------------------------------------------------------------------------------------------------------

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option

set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
pd.options.mode.chained_assignment = None

input_file = './raw data/UDS_A5.csv'
data_input_ori = pd.read_csv(input_file)
data_keys = data_input_ori.keys()
data_input = data_input_ori[['UDS_A5SUBHSTDATA ID', 'Subject', 'CVHATT', 'CVAFIB', 'CVANGIO',
        'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA', 'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF',
        'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF',
        'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'ABUSOTHR', 'PSYCDIS']] # 'QUITSMOK',
M, N = data_input.shape
idx = data_input['Subject']
for i in range(M):
    if np.isnan(data_input.TOBAC30[i]):
        data_input.TOBAC30[i] = 0

    if np.isnan(data_input.SMOKYRS[i]):
        data_input.SMOKYRS[i] = 0

    if np.isnan(data_input.PACKSPER[i]):
        data_input.PACKSPER[i] = 0

    if data_input.SMOKYRS[i] == 88 or data_input.SMOKYRS[i] == 99:
        data_input.SMOKYRS[i] = 0

    if data_input.SMOKYRS[i] >= 20 and (data_input.PACKSPER[i] == 8 or data_input.PACKSPER[i] == 9):
        data_input.PACKSPER[i] = 4.0
    elif data_input.SMOKYRS[i] >= 10 and (data_input.PACKSPER[i] == 8 or data_input.PACKSPER[i] == 9):
        data_input.PACKSPER[i] = 3.0
    elif data_input.SMOKYRS[i] > 2.0 and (data_input.PACKSPER[i] == 8 or data_input.PACKSPER[i] == 9):
        data_input.PACKSPER[i] = 2.0
    elif data_input.SMOKYRS[i] < 2.0 and (data_input.PACKSPER[i] == 8 or data_input.PACKSPER[i] == 9):
        data_input.PACKSPER[i] = 1.0
    else:
        data_input.PACKSPER[i] = data_input.PACKSPER[i]

data_keys = data_input.keys()
data_keys = data_keys[2:]

# Remove missing data and replace 9(unknown) with 0
for col in data_keys:
    #print(col)
    data_tmp = data_input[str(col)]
    #print(data_tmp)
    for i in range(M):
        if np.isnan(data_tmp[i]):
            #print(data_tmp[i])
            data_tmp[i] = 0

        if data_tmp[i] == 9:
            data_tmp[i] = 0

    data_input[str(col)] = data_tmp

data_keys_tmp = ['CVHATT', 'CVAFIB', 'CVANGIO',
        'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA', 'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF',
        'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF',
        'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'ABUSOTHR', 'PSYCDIS']

# Swap 1 and 2.
for col in data_keys_tmp:
    #print(col)
    data_tmp = data_input[col]
    #print(data_tmp)
    for i in range(M):
        if data_tmp[i] == 2:
            #print(data_tmp[i])
            data_tmp[i] = 1.0
        elif data_tmp[i] == 1:
            data_tmp[i] = 2.0
        else:
            data_tmp[i] = 0.0

    data_input[col] = data_tmp

data_input.to_csv('./raw data_edit/UDS_A5_edit.csv')

input_file = './raw data/UDS_B6.csv'
data_input_ori = pd.read_csv(input_file)

data_keys = data_input_ori.keys()
print(data_keys)

data_input = data_input_ori[['UDS_B6BEVGDSDATA ID', 'Subject', 'GDS']]
M, N = data_input.shape
print(M, N)

idx = data_input['Subject']

for i in range(M):
    if np.isnan(data_input.GDS[i]) or data_input.GDS[i] > 15:
        if data_input.Subject[i] == data_input.Subject[i-1] and data_input.Subject[i] == data_input.Subject[i+1] and \
                data_input.GDS[i+1]>15 and data_input.GDS[i-1]<15:
            data_input.GDS[i] = data_input.GDS[i-1]
        elif data_input.Subject[i] == data_input.Subject[i-1] and data_input.Subject[i] == data_input.Subject[i+1] and \
                data_input.GDS[i-1]>15 and data_input.GDS[i+1]<15:
            data_input.GDS[i] = data_input.GDS[i+1]
        elif data_input.Subject[i] == data_input.Subject[i-1] and data_input.Subject[i] == data_input.Subject[i+1]:
            data_input.GDS[i] = (data_input.GDS[i-1] + data_input.GDS[i+1]) / 2.0
        elif data_input.Subject[i] == data_input.Subject[i-1]:
            data_input.GDS[i] = data_input.GDS[i - 1]
        elif data_input.Subject[i] == data_input.Subject[i+1]:
            data_input.GDS[i] = data_input.GDS[i+1]
        else:
            data_input.GDS[i] = 7.0  # Use avearage values of GDS in this case

    if np.isnan(data_input.GDS[i]) or data_input.GDS[i] > 15:
        print(i, data_input.Subject[i - 1:i + 2], data_input.GDS[i - 1:i + 2])

data_input.to_csv('./raw data_edit/UDS_B6_edit.csv')


input_file = './raw data/ADRC.csv'    # The well name of an input file
data_input_ori = pd.read_csv(input_file)
data_keys = data_input_ori.keys()
data_input = data_input_ori[['ADRC_ADRCCLINICALDATA ID', 'Subject', 'Age', 'mmse', 'ageAtEntry', 'cdr', 'commun',
        'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe', 'sumbox', 'height', 'weight', 'M/F', 'Hand', 'Education', 'Race']]
M, N = data_input.shape

subject_id = data_input['ADRC_ADRCCLINICALDATA ID']
age0 = data_input['ageAtEntry']
age = []
for i in range(M):
    tmp1 = subject_id[i]
    tmp1 = int(tmp1[23:])

    if tmp1 == 0:
        age_tmp = age0[i]
        tmp2 = tmp1
    else:
        age_tmp = age0[i] + (tmp1 - tmp2) / 365.0
    age.append(age_tmp)

data_input['Age']  = age
idx = data_input['Subject']
data_edit1 = data_input_ori[['mmse', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox']]
data_keys = data_edit1.keys()

# Remove missing data and replace 9(unknown) with 0
for col in data_keys:
    #print(col)
    data_tmp = data_edit1[str(col)]
    #print(data_tmp)
    for i in range(M):
        if np.isnan(data_tmp[i]):
            #print(data_tmp[i])
            data_tmp[i] = 0

        if data_tmp[i] == 9:
            data_tmp[i] = 0
    data_input[str(col)] = data_tmp

for i in range(M):
    if np.isnan(data_input.apoe[i]):
        data_input.apoe[i] = 23

    if data_input.apoe[i] == 22:
        data_input.apoe[i] = 2.0
    elif data_input.apoe[i] == 23 or data_input.apoe[i] == 33:
        data_input.apoe[i] = 1.0
    else:
        data_input.apoe[i] = 0.0
    #print(data_input.apoe[i])

# Calculate the BMI - body mass index
hwr = []
for i in range(M):
    if np.isnan(data_input.height[i]) or np.isnan(data_input.weight[i]):

        hwr_tmp = 21.0
    else:
        hwr_tmp = data_input.weight[i] * 0.453592 / (data_input.height[i] * 0.0254) ** 2.0  # w / h2, unit: (kg/m2)
    hwr.append(hwr_tmp)

    if data_input['M/F'][i] == 'F':
        data_input['M/F'][i] = 0.0
    else:
        data_input['M/F'][i] = 1.0

    if data_input.Hand[i] == 'R':
        data_input.Hand[i] = 2.0
    elif data_input.Hand[i] == 'L':
        data_input.Hand[i] = 1.0
    else:
        data_input.Hand[i] = 0.0

    if np.isnan(data_input.Education[i]):
        data_input.Education[i] = 15.0
    if data_input.Race[i] == 'Caucasian':
        data_input.Race[i] = 1.0
    elif data_input.Race[i] == 'African American':
        data_input.Race[i] = 2.0
    elif data_input.Race[i] == 'Asian':
        data_input.Race[i] =3.0
    else:
        data_input.Race[i] = 4.0

for i in range(M):
    if np.isnan(data_input.height[i]) or np.isnan(data_input.weight[i]):
        if i == 0 and ~np.isnan(data_input.height[i+1]) and ~np.isnan(data_input.weight[i+1]) and \
                data_input.Subject[i] == data_input.Subject[i+1]:
            hwr[i] = hwr[i+1]
        elif np.isnan(data_input.height[i]) or np.isnan(data_input.weight[i]) and i > 0:
            if data_input.Subject[i] == data_input.Subject[i-1]:
                hwr[i] = hwr[i - 1]
            elif data_input.Subject[i] == data_input.Subject[i+1] and ~np.isnan(data_input.height[i+1]) and ~np.isnan(data_input.weight[i+1]):
                hwr[i] = hwr[i + 1]
            elif data_input.Subject[i] == data_input.Subject[i+2] and ~np.isnan(data_input.height[i+2]) and ~np.isnan(data_input.weight[i+2]):
                hwr[i] = hwr[i + 2]
            elif data_input.Subject[i] == data_input.Subject[i+3] and ~np.isnan(data_input.height[i+3]) and ~np.isnan(data_input.weight[i+3]):
                hwr[i] = hwr[i + 3]
            elif data_input.Subject[i] == data_input.Subject[i+4] and ~np.isnan(data_input.height[i+4]) and ~np.isnan(data_input.weight[i+4]):
                hwr[i] = hwr[i + 4]
            elif data_input.Subject[i] == data_input.Subject[i+5] and ~np.isnan(data_input.height[i+5]) and ~np.isnan(data_input.weight[i+5]):
                hwr[i] = hwr[i + 5]
            elif data_input.Subject[i] == data_input.Subject[i+6] and ~np.isnan(data_input.height[i+6]) and ~np.isnan(data_input.weight[i+6]):
                hwr[i] = hwr[i + 6]
            elif data_input.Subject[i] == data_input.Subject[i+7] and ~np.isnan(data_input.height[i+7]) and ~np.isnan(data_input.weight[i+7]):
                hwr[i] = hwr[i + 7]
            elif data_input.Subject[i] == data_input.Subject[i+8] and ~np.isnan(data_input.height[i+8]) and ~np.isnan(data_input.weight[i+8]):
                hwr[i] = hwr[i + 8]
            elif data_input.Subject[i] == data_input.Subject[i+18] and ~np.isnan(data_input.height[i+18]) and ~np.isnan(data_input.weight[i+18]):
                hwr[i] = hwr[i + 18]
            elif data_input.Subject[i] != data_input.Subject[i+1] and data_input.Subject[i] != data_input.Subject[i-1]:
                hwr[i] = 21.0
            else:
                hwr[i] = 21.0
                print(i, data_input.Subject[i], hwr[i])
        else:
            hwr[i] = 9999
            print(i, data_input.Subject[i], hwr[i])

    if hwr[i] > 50.0:
        hwr[i] = 50.0

"""
1 = White
2 = Black or African American
3 = American Indian or Alaska Native
4 = Native Hawaiian or Other Pacific Islander
5 = Asian
50 = Other
99 = Unknown
"""

# BMI (body mass index) is a measure of whether you're a healthy weight for your height
# Less than 18.5 = Underweight, Between 18.5 - 24.9 = Healthy Weight, Between 25 - 29.9 = Overweight, Over 30 = Obese
data_input['BMI'] = hwr
data_input1 = data_input[['ADRC_ADRCCLINICALDATA ID', 'Subject', 'Age', 'mmse', 'ageAtEntry', 'cdr', 'commun',
        'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe', 'sumbox', 'M/F', 'Hand', 'Education', 'Race', 'BMI']]

data_input1.to_csv('./raw data_edit/ADRC_edit.csv')