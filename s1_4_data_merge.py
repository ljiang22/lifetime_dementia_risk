import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
import os

set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
#set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

input_file = './raw data/ADRC.csv'    # The well name of an input file
data_input_ori = pd.read_csv(input_file)

data_path = './raw data_edit/'
files = sorted(os.listdir(data_path))
print(files)

files = ['ADRC_edit.csv', 'UDS_A1_edit.csv', 'UDS_A3_edit.csv', 'UDS_A5_edit.csv', 'UDS_B6_edit.csv', 'data_edit info']

data_tmp1 = pd.read_csv(data_path + files[0])
data_tmp2 = pd.read_csv(data_path + files[1])
data_tmp3 = pd.read_csv(data_path + files[2])
data_tmp4 = pd.read_csv(data_path + files[3])
data_tmp5 = pd.read_csv(data_path + files[4])

M1, N1 = data_tmp1.shape
print(M1, N1)
M2, N2 = data_tmp2.shape
print(M2, N2)
M3, N3 = data_tmp3.shape
print(M3, N3)

key1 = data_tmp1.keys()
print(key1)
key2 = data_tmp2.keys()
print(key2)
key3 = data_tmp3.keys()
print(key3)
key4 = data_tmp4.keys()
print(key4)
key5 = data_tmp5.keys()
print(key5)

id1 = data_tmp1['ADRC_ADRCCLINICALDATA ID']
id2 = data_tmp2['UDS_A1SUBDEMODATA ID']
id3 = data_tmp3['UDS_A3SBFMHSTDATA ID']
id4 = data_tmp4['UDS_A5SUBHSTDATA ID']
id5 = data_tmp5['UDS_B6BEVGDSDATA ID']

j = 0
idx = []
idx1 = []
for i in range(M1):
    if j < M2 - 1 and  i < M1 -1:
        #print(i, j, j+1)
        tmp1 = id1[i]
        tmp1 = int(tmp1[23:])
        tmp1c = data_tmp1.Subject[i]
        tmp1d = data_tmp1.Subject[i + 1]

        tmp2 = id2[j]
        tmp2 = int(tmp2[16:])
        tmp2c = data_tmp2.Subject[j]
        tmp2d = data_tmp2.Subject[j + 1]

        # print(tmp1, tmp2)
        if abs(tmp1 - tmp2) < 150 and tmp1c == tmp2c:
            j_tmp = j
            if tmp2c == tmp2d and tmp1d != tmp2d:
                jc = j
                for nj in range(10):
                    if data_tmp2.Subject[jc + nj] == data_tmp2.Subject[jc + nj + 1]:
                        j = j + 1
                j = j + 1
            else:
                j += 1

            print(tmp1c, tmp2c, i, j - 1, tmp1, tmp2)
            idx.append(i)
            idx1.append(j_tmp)

            # print(jc, j)
        if abs(tmp1 - tmp2) > 150 and tmp1c == tmp2c and tmp2c != tmp2d and tmp1c != tmp1d:
            j += 1
            print(tmp1c, tmp2c, i, j - 1, tmp1, tmp2)
    else:
        if tmp1d == tmp2d:
            print(tmp1d, tmp2d, i, j)

#print(idx)
data_tmp1_edit = pd.DataFrame(data_tmp1, columns = ['ADRC_ADRCCLINICALDATA ID', 'Subject', 'Age', 'mmse', 'ageAtEntry',
            'cdr', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe', 'sumbox', 'M/F', 'Hand',
                  'Education', 'Race', 'BMI'], index=idx)
data_tmp2_edit = pd.DataFrame(data_tmp2, columns = ['UDS_A1SUBDEMODATA ID', 'Subject', 'LIVSIT', 'INDEPEND', 'RESIDENC', 'MARISTAT'], index=idx1)
data_tmp3_edit = pd.DataFrame(data_tmp3, columns = ['UDS_A3SBFMHSTDATA ID', 'Subject', 'dem_idx'], index=idx1)
data_tmp4_edit = pd.DataFrame(data_tmp4, columns = ['UDS_A5SUBHSTDATA ID', 'Subject', 'CVHATT', 'CVAFIB', 'CVANGIO',
    'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA', 'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF',
    'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF',
     'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'ABUSOTHR', 'PSYCDIS'], index=idx1)
data_tmp5_edit = pd.DataFrame(data_tmp5, columns = ['UDS_B6BEVGDSDATA ID', 'Subject', 'GDS'], index=idx1)

# Quality control
data_tmp1_edit.reset_index(drop=True, inplace=True) # Reindex the dataset in a continuous number
data_tmp2_edit.reset_index(drop=True, inplace=True)
data_tmp3_edit.reset_index(drop=True, inplace=True)
data_tmp4_edit.reset_index(drop=True, inplace=True)
data_tmp5_edit.reset_index(drop=True, inplace=True)

for i in range(len(idx)):
    #print(i)
    #print(data_tmp1_edit.Subject[i])
    #print(data_tmp1_edit.Subject[i], data_tmp2_edit.Subject[i])
    if data_tmp1_edit.Subject[i] != data_tmp2_edit.Subject[i]:
        print('1', data_tmp1_edit.Subject[i], data_tmp2_edit.Subject[i])

    if data_tmp2_edit.Subject[i] != data_tmp3_edit.Subject[i]:
        print('2', data_tmp2_edit.Subject[i], data_tmp3_edit.Subject[i])

    if data_tmp2_edit.Subject[i] != data_tmp4_edit.Subject[i]:
        print('3', data_tmp2_edit.Subject[i], data_tmp4_edit.Subject[i])

    if data_tmp2_edit.Subject[i] != data_tmp5_edit.Subject[i]:
        print('4', data_tmp2_edit.Subject[i], data_tmp5_edit.Subject[i])

data_tmp2_edit = data_tmp2_edit.drop(columns=['UDS_A1SUBDEMODATA ID', 'Subject'])
data_tmp3_edit = data_tmp3_edit.drop(columns=['UDS_A3SBFMHSTDATA ID', 'Subject'])
data_tmp4_edit = data_tmp4_edit.drop(columns=['UDS_A5SUBHSTDATA ID', 'Subject'])
data_tmp5_edit = data_tmp5_edit.drop(columns=['UDS_B6BEVGDSDATA ID', 'Subject'])

print(data_tmp1_edit)
#print(data_tmp1_edit.index(), data_tmp2_edit.index())

data_all = pd.concat([data_tmp1_edit, data_tmp2_edit, data_tmp3_edit, data_tmp4_edit, data_tmp5_edit], axis=1)
print(data_tmp1_edit.shape, data_tmp2_edit.shape, data_all.shape)

M, N = data_all.shape
for i in range(M):
    if data_all.LIVSIT[i] == 9:
        data_all.LIVSIT[i] = 1.0
    data_all.LIVSIT[i] = data_all.LIVSIT[i] - 1.0

    if data_all.INDEPEND[i] == 9:
        data_all.INDEPEND[i] = 1.0
    data_all.INDEPEND[i] = data_all.INDEPEND[i] - 1.0

    if data_all.RESIDENC[i] == 9:
        data_all.RESIDENC[i] = 5.0
    data_all.RESIDENC[i] = data_all.RESIDENC[i] - 1.0


    # Treate 'living as married" as 'married'
    if data_all.MARISTAT[i] == 1:
        data_all.MARISTAT[i] = 0.0

    if data_all.MARISTAT[i] == 6:
        data_all.MARISTAT[i] = 1.0

    if data_all.MARISTAT[i] == 9:
        data_all.MARISTAT[i] = 2.0

    if data_all.MARISTAT[i] == 8:
        data_all.MARISTAT[i] = 2.0



    if data_all.TOBAC30[i] > 1:
        data_all.TOBAC30[i] = 1

    if data_all.TOBAC100[i] > 1:
        data_all.TOBAC100[i] = 1

data_all.to_csv('./raw data_edit/data_merge.csv')



