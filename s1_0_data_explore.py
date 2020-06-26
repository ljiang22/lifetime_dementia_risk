import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option

set_option('display.width', 500)
pd. set_option("display.max_rows", 100, "display.max_columns", 300)
set_option('precision', 3)
#set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None
#names = ['Depth', 'CALI', 'GR', 'RD', 'RS', 'NPHI', 'PHIE', 'PEF', 'VSH', 'DT', 'RHOB', 'DTS']
input_file = 'ADRC_0529.csv'    # The well name of an input file
data_input = pd.read_csv(input_file)

data_keys = data_input.keys()
print(data_keys)

data_input_ori = data_input
"""data_input = data_input_ori[['ADRC_ADRCCLINICALDATA ID', 'Subject', 'mmse', 'ageAtEntry', 'cdr','commun', 'dx1', 'dx2', 'dx3', 'dx4', 'dx5',
       'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe', 'sumbox', 'height', 'weight',
       'M/F', 'Hand', 'Education', 'Race']]"""
data_input = data_input_ori[['ADRC_ADRCCLINICALDATA ID', 'Subject', 'Age', 'mmse', 'ageAtEntry', 'cdr','commun',
       'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe', 'sumbox', 'height', 'weight',
       'M/F', 'Hand', 'Education', 'Race']]
M, N = data_input.shape

age = data_input['ADRC_ADRCCLINICALDATA ID']
age0 = data_input['ageAtEntry']
for i in range(len(age)):
    tmp1 = age[i]
    tmp1 = int(tmp1[23:])

    if tmp1 == 0:
        age[i] = age0[i]
        tmp2 = tmp1
    else:
        age[i] = age0[i] + (tmp1 - tmp2) / 365.0
    #print(age0[i], tmp1, tmp2, (tmp1 - tmp2) / 365.0, age[i])

    #print(tmp, age[i])
#print(age.shape)
#print(age)
#age1 = int(age[6][23:])
#print(type(age1))
#print(age1)
data_input['Age']  = age
print(data_input['Age'])

print(data_input.shape)
peek = data_input.head(5)
print(peek)
data_keys = data_input.keys()
print(data_keys)

well_stat = data_input['Age'].describe()
print(well_stat)

ip_counts = data_input.groupby('ageAtEntry').size()
print(ip_counts)


correlations = data_input.corr(method='pearson')
print(correlations)
#data_input.hist()


#data_input_edit = data_input[(data_input.cdr == 0.5) & (data_input.cdr <= 5.0)]
data_input_edit = data_input[(data_input.cdr == 2.0)]
#data_input_edit = data_input[(data_input.ageAtEntry > 65) & (data_input.ageAtEntry < 97) & (data_input.cdr>=0.0) & (data_input.Subject == 'OAS30076')]
print(data_input_edit)
#plt.figure(1)
age = data_input_edit['Age']

age.hist()


cdr = data_input_edit['cdr']

# Plot the results
font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)


f, bx = plt.subplots(nrows=1, ncols=1)
bx.hist(data_input['Age'], bins=10, color='black')
#bx.legend()
#bx.set_ylim(ztop, zbot)
#bx.invert_yaxis()
#bx.grid()
bx.locator_params(axis='x', nbins=7)
bx.set_xlabel("Age")
bx.set_ylabel("Count")
#bx.set_xlim(np.min(Udry) - 3, np.max(Udry) + 5)

f, bx = plt.subplots(nrows=1, ncols=1)
bx.hist(age, bins=10, color='black')
#bx.legend()
#bx.set_ylim(ztop, zbot)
#bx.invert_yaxis()
#bx.grid()
bx.locator_params(axis='x', nbins=7)
bx.set_xlabel("Age")
bx.set_ylabel("Count")
#bx.set_xlim(np.min(Udry) - 3, np.max(Udry) + 5)

f, bx = plt.subplots(nrows=1, ncols=1)
bx.scatter(age, data_input_edit.cdr, color='black')
#bx.legend()
#bx.set_ylim(ztop, zbot)
#bx.invert_yaxis()
#bx.grid()
bx.locator_params(axis='x', nbins=7)
bx.set_xlabel("Age")
bx.set_ylabel("CDR")
#bx.set_xlim(np.min(Udry) - 3, np.max(Udry) + 5)

plt.show()