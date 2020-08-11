import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option

set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
#set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

input_file = 'C:\Study2017\study and work2018_syn_amazon\Insight program\Oasis project\AD prediction1\\raw data_edit\data_merge.csv'
data_input_ori = pd.read_csv(input_file)
data_input = data_input_ori

M, N = data_input.shape
print(M, N)

# Effect of BMI
data_input_edit2 = data_input[(data_input.Age > 70.0) & (data_input.Age < 90) &
                    (data_input.Race==1) & (data_input.BMI<25.0)& (data_input.BMI>18.5)]
data_input_edit3 = data_input[ (data_input.Age > 70.0) & (data_input.Age < 90) &
                    (data_input.Race==1) & (data_input.BMI<17.0)]

attr = data_input_edit2['cdr']
attr1 = data_input_edit3['cdr']
attr_all = []

attr_list = list(attr.values)
attr_list1 = list(attr1.values)
attr_all.append(attr_list)
attr_all.append(attr_list1)

font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)

colors =['black', 'r']
labels = ['18.5 < BMI < 25.0', 'BMI >= 25.0']
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



