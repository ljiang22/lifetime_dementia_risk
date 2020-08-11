# Analyze the effect of APOE genes on brain health
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option

# Define the display width of data
set_option('display.width', 500)
set_option("display.max_rows", 100, "display.max_columns", 300)
set_option('precision', 3)
pd.options.mode.chained_assignment = None

input_file = './raw data/ADRC.csv'
data_input = pd.read_csv(input_file)
data_keys = data_input.keys()

data_input_ori = data_input
data_input = data_input_ori[['ADRC_ADRCCLINICALDATA ID', 'Subject', 'Age', 'mmse', 'ageAtEntry', 'cdr','commun',
       'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe', 'sumbox', 'height', 'weight',
       'M/F', 'Hand', 'Education', 'Race']]
M, N = data_input.shape

age = data_input['ADRC_ADRCCLINICALDATA ID']
age0 = data_input['ageAtEntry']
hwr = []
# Calculate the BMI and age using the length of test days
for i in range(len(age)):
    tmp1 = age[i]
    tmp1 = int(tmp1[23:])
    if tmp1 == 0:
        age[i] = age0[i]
        tmp2 = tmp1
    else:
        age[i] = age0[i] + (tmp1 - tmp2) / 365.0
    hwr_tmp = data_input.weight[i] * 0.453592 / (data_input.height[i] * 0.0254) ** 2.0  # w / h2, unit: (kg/m2)
    hwr.append(hwr_tmp)

# BMI (body mass index) is a measure of whether you have a healthy weight for your height
# Less than 18.5 = Underweight, Between 18.5 - 24.9 = Healthy Weight, Between 25 - 29.9 = Overweight, Over 30 = Obese
data_input['BMI'] = hwr
data_input['Age'] = age

peek = data_input.head(5)
print(peek)

"""data_input_edit = data_input[(data_input.cdr == 2.0) & (data_input.Age > 75.0) & (data_input.Age < 80) &
          (data_input['M/F'] == 'M') & (data_input.Race=='Caucasian') & (data_input.Hand == 'R') & (data_input.BMI > 18.0) &
                             (data_input.apoe == 33) & (data_input.homehobb == 0) & (data_input.sumbox == 0)]"""
data_input_edit = data_input[(data_input.cdr >= 0.5) & (data_input.Age > 75.0) & (data_input.Age < 80.0) &
           (data_input.Race=='Caucasian') & (data_input.BMI > 25.0) & (data_input.BMI < 35.0) &
                             (data_input.homehobb >= 0) & (data_input.sumbox >= 0)]
data_input_edit1 = data_input[(data_input.cdr < 0.5) & (data_input.Age > 75.0) & (data_input.Age < 80) &
                    (data_input.Race=='Caucasian') & (data_input.BMI > 25.0) & (data_input.BMI < 35.0)]
#data_input_edit = data_input[(data_input.ageAtEntry > 65) & (data_input.ageAtEntry < 97) & (data_input.cdr>=0.0) & (data_input.Subject == 'OAS30076')]
data_input_edit2 = data_input[(data_input.Age > 75.0) & (data_input.Age < 80) &
                    (data_input.Race=='Caucasian') & (data_input.BMI > 25.0) & (data_input.BMI < 35.0) & (data_input.apoe==33)]
data_input_edit3 = data_input[ (data_input.Age > 75.0) & (data_input.Age < 80) &
                    (data_input.Race=='Caucasian') & (data_input.BMI > 25.0) & (data_input.BMI < 35.0) & (data_input.apoe==34)]

print('input data edit', data_input_edit)
#plt.figure(1)

#attr = data_input_edit['apoe']
#attr1 = data_input_edit1['apoe']
attr = data_input_edit2['cdr']
attr1 = data_input_edit3['cdr']
attr_all = []

print(len(attr1), len(data_input_edit1))
#print('attr', attr)

attr_list = list(attr.values)
attr_list1 = list(attr1.values)
attr_all.append(attr_list)
attr_all.append(attr_list1)
print(attr_all)

cdr = data_input_edit['cdr']
# Plot the results
font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)

colors =['black', 'r']
labels = ['APOE 3', 'APOE 4']
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