import pandas as pd
from mllib import utils
import matplotlib.pyplot as plt
from pandas import set_option


#----------------------------------------------------------------------------
# Merge the data from different tables and do some preliminary analysis on the data
def main():
    # Set the display parameters for panda dataframe
    set_option('display.width', 2000)
    pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
    set_option('precision', 3)

    data_path = './raw data_edit/'
    files = ['ADRC_edit.csv', 'UDS_A1_edit.csv', 'UDS_A3_edit.csv', 'UDS_A5_edit.csv', 'UDS_B6_edit.csv']
    data_tmp1 = pd.read_csv(data_path + files[0])
    data_tmp2 = pd.read_csv(data_path + files[1])
    data_tmp3 = pd.read_csv(data_path + files[2])
    data_tmp4 = pd.read_csv(data_path + files[3])
    data_tmp5 = pd.read_csv(data_path + files[4])

    # Merge the data from different tables/source
    data_merge = utils.data_merge(data_tmp1, data_tmp2, data_tmp3, data_tmp4, data_tmp5)

    # Save the data to csv file
    data_merge.to_csv('./raw data_edit/data_merge.csv')

    data_input_ori = data_merge
    j = 0
    # Replace the wrong input with the calculated sumbox (SCDR)
    for i in range(data_input_ori.shape[0]):
        sumbox_clc = data_input_ori.commun[i] + data_input_ori.homehobb[i] + data_input_ori.judgment[i] + \
                     data_input_ori.memory[i] + data_input_ori.orient[i] + data_input_ori.perscare[i]
        if sumbox_clc != data_input_ori.sumbox[i]:
            data_input_ori.sumbox[i] = sumbox_clc
            j = j + 1
    data_input = data_input_ori
    M, N = data_input.shape

    data_diag = data_input[['cdr', 'sumbox', 'mmse', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare']]

    # Plot pie plot for each class
    x1 = len(data_diag[(data_diag.sumbox < 0.5)]) / M
    x2 = len(data_diag[(data_diag.sumbox >= 0.5) & (data_diag.sumbox < 5.0)]) / M
    x3 = len(data_diag[(data_diag.sumbox >= 5.0) & (data_diag.sumbox < 10.0)]) / M
    x4 = len(data_diag[(data_diag.sumbox >= 10.0) & (data_diag.sumbox < 15.0)]) / M
    x5 = len(data_diag[(data_diag.sumbox >= 15.0) & (data_diag.sumbox < 35.0)]) / M

    font = {'size': 18}
    plt.rc('font', **font)
    data_pie = [x1, x2, x3, x4, x5]
    explode = (0.1, 0.0, 0, 0, 0.0)
    labels = ['Normal', 'Mild impairment', '', '', 'Dementia']
    f, bx = plt.subplots(nrows=1, ncols=1)
    bx.pie(data_pie, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.2f%%')

    data_fd = data_input[['Subject', 'sumbox', 'Age', 'apoe', 'M/F', 'Hand', 'Education', 'Race', 'LIVSIT', 'INDEPEND',
                          'RESIDENC', 'MARISTAT', 'BMI', 'dem_idx', 'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE',
                          'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA', 'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF',
                          'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF', 'THYROID', 'INCONTU',
                          'INCONTF', 'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'ABUSOTHR',
                          'PSYCDIS', 'GDS']]

    # Save the training data of machine learning to csv
    data_fd.to_csv('./raw data_edit/data_fd.csv')
    data_fd1 = data_input[['Age', 'apoe', 'M/F', 'Hand', 'Education', 'Race', 'LIVSIT', 'INDEPEND', 'RESIDENC']]
    data_fd2 = data_input[['MARISTAT', 'BMI', 'dem_idx', 'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 'CVCHF']]
    data_fd3 = data_input[['CBSTROKE', 'CBTIA', 'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR']]
    data_fd4 = data_input[['HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR']]
    data_fd5 = data_input[['ALCOHOL', 'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'ABUSOTHR', 'PSYCDIS', 'GDS']]

    # Effect of genes
    data_input_edit2 = data_input[(data_input.Age > 75.0) & (data_input.Age < 80) &
                                  (data_input.Race == 1) & (data_input.apoe == 1)]
    data_input_edit3 = data_input[(data_input.Age > 75.0) & (data_input.Age < 80) &
                                  (data_input.Race == 1) & (data_input.apoe == 0)]

    attr = data_input_edit2['cdr']
    attr1 = data_input_edit3['cdr']
    attr_all = []

    attr_list = list(attr.values)
    attr_list1 = list(attr1.values)
    attr_all.append(attr_list)
    attr_all.append(attr_list1)

    colors = ['black', 'r']
    labels = ['APOE 3', 'APOE 4']
    f, bx = plt.subplots(nrows=1, ncols=1)
    bx.hist(attr_all, bins=10, density=True, color=colors, label=labels)
    bx.legend()
    bx.locator_params(axis='x', nbins=7)
    bx.set_xlabel("CDR")
    bx.set_ylabel("Probability Density Function")

    data_fd_st = data_fd.describe()
    print('The statistic summary of the data:')
    print(data_fd_st)

    data_fd1.hist(grid=False, bins=6)
    data_fd2.hist(grid=False, bins=6)
    data_fd3.hist(grid=False, bins=6)
    data_fd4.hist(grid=False, bins=6)
    data_fd5.hist(grid=False, bins=6)
    plt.show()


#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
