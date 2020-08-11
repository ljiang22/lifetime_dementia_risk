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

input_file = './raw data_edit/data_merge.csv'    # The well name of an input file
data_input_ori = pd.read_csv(input_file)
j=0
for i in range(data_input_ori.shape[0]):
    sumbox_clc = data_input_ori.commun[i] + data_input_ori.homehobb[i] + data_input_ori.judgment[i] + \
        data_input_ori.memory[i] + data_input_ori.orient[i] + data_input_ori.perscare[i]
    if sumbox_clc != data_input_ori.sumbox[i]:
        #print(sumbox_clc, data_input_ori.sumbox[i])
        data_input_ori.sumbox[i] = sumbox_clc
        j = j+1
print(j)

#data_input = data_input_ori[(data_input_ori.cdr < 1.0) | (data_input_ori.sumbox > 2.0)]
data_input = data_input_ori

M, N = data_input.shape
print(M, N)
keys = data_input.keys()
print(keys)

# Calculate clinic dementia rating for label
data_diag = data_input[['cdr', 'sumbox', 'mmse', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare']]
feature = data_input[['mmse', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare']]
label = data_input['sumbox']

feature_st = data_diag.describe()
#print(feature_st)
# Plot the results
font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)


# Effect of family history
data_input_edit2 = data_input[(data_input.Age > 40.0) & (data_input.Age < 99) &
                    (data_input.Race==1) & (data_input.INDEPEND==0.0)]
data_input_edit3 = data_input[ (data_input.Age > 40.0) & (data_input.Age < 99) &
                    (data_input.Race==1) & (data_input.INDEPEND>=1.0)]

attr = data_input_edit2['cdr']
attr1 = data_input_edit3['cdr']
attr_all = []

attr_list = list(attr.values)
attr_list1 = list(attr1.values)
attr_all.append(attr_list)
attr_all.append(attr_list1)
print(attr_all)

colors =['black', 'r']
labels = ['Able to live independently', 'Not able to live independently']
f, bx = plt.subplots(nrows=1, ncols=1)
bx.hist(attr_all, bins=10, normed=True, color=colors, histtype='bar', label=labels)
#bx.hist(attr1, bins=22, density=True, color='r', histtype='bar', label='CDR < 0.5')
bx.legend()
#bx.set_ylim(ztop, zbot)
#bx.invert_yaxis()
#bx.grid()
bx.locator_params(axis='x', nbins=7)
bx.set_xlabel("CDR")
bx.set_ylabel("Probability density")


plt.show()



