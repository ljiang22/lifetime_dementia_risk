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
input_file = './raw data_edit/data_merge.csv'    # The well name of an input file
data_input = pd.read_csv(input_file)

data_keys = data_input.keys()
print('input data shape is:', data_input.shape)
print(data_keys)
M, N = data_input.shape
"""data_input_edit = data_input[(data_input['M/F'] == 'M') & (data_input.Race==1) & (data_input.Hand == 0) & (data_input.BMI > 18.0) &
                             (data_input.apoe == 33) & (data_input.LIVSIT==1)]"""
data_input_edit = data_input[(data_input['M/F'] == 1) & (data_input.Race==1) & (data_input.Hand == 0) & (data_input.BMI > 18.0) &
                             (data_input.BMI < 30.0) & (data_input.apoe == 0)& (data_input.LIVSIT==2) & (data_input.MARISTAT ==1)&
                             (data_input.dem_idx ==0)]

print(data_input_edit.shape)
#plt.figure(1)
age = data_input_edit['Age']


sumbox = data_input_edit['sumbox']

# Plot the results
font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)


f, bx = plt.subplots(nrows=1, ncols=1)
bx.scatter(age, sumbox, color='black')
#bx.legend()
#bx.set_ylim(ztop, zbot)
#bx.invert_yaxis()
#bx.grid()
bx.locator_params(axis='x', nbins=7)
bx.set_xlabel("Age")
bx.set_ylabel("CDR")
#bx.set_xlim(np.min(Udry) - 3, np.max(Udry) + 5)

plt.show()