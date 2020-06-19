import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from mllib.preprocess import normalize
from mllib.preprocess import unnormalize
from mllib.preprocess import params_clc
from mllib.networks import evaluate
from mllib.networks import MLP_train_opt
from mllib.networks import MLP_plot
from mllib.networks import MLP_REG_v1
from mllib.networks import MLPR_v0
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
import pickle


set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
#set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

input_file = './raw data_edit/data_fd.csv'    # The well name of an input file
data_input_ori = pd.read_csv(input_file)
print(data_input_ori.head())

#training_method ='SVR'
#training_method ='MLP'
training_method ='RF'
opt_flag = 0
NT = 2

keys = data_input_ori.keys()
print(keys)

#print(data_edit[0:50])

data_input = data_input_ori
print(data_input.shape, type(data_input))
Nt = data_input.shape[0]
age_max = 110
data_new = data_input.loc[0:2]
print(data_new)
age_tmp = []
sumbox_tmp = []
aging_avg = 0.6
nfg = 3
for nt in range(Nt-1):
    id1 = data_input.Subject[nt]
    id2 = data_input.Subject[nt+1]

    if id1 == id2:
        #print(nt, data_input.Age[nt])
        age_tmp.append(data_input.Age[nt])
        sumbox_tmp.append(data_input.sumbox[nt])
    else:
        age_tmp.append(data_input.Age[nt])
        sumbox_tmp.append(data_input.sumbox[nt])
        M = len(age_tmp)

        #print(age_tmp)

        N = int((age_max - age_tmp[M-1]) / nfg)
        data_new_tmp = data_input.loc[nt-M+1 : nt]
        data_tmp = data_input.loc[nt]
        #print(nt, M, N, age_tmp, age_tmp[M-1])
        #print(data_new_tmp)
        data_new = data_new.append(data_new_tmp, ignore_index=True)

        # Calculate the brain health declining rate with age. The assumption is other factors keep constant.
        if M > 1:
            aging_rate = (sumbox_tmp[M-1] - sumbox_tmp[0])/ (age_tmp[M-1]- age_tmp[0])
            if aging_rate < aging_avg:
                if age_tmp[M-1] < 70:
                    aging_rate = aging_avg * 0.7
                else:
                    aging_rate = aging_avg
        else:
            if age_tmp[M - 1] < 70:
                aging_rate = aging_avg * 0.7
            else:
                aging_rate = aging_avg

        for n in range(1, N):
            data_tmp.Age = age_tmp[M-1] + n * nfg

            data_tmp.sumbox = sumbox_tmp[M-1] + aging_rate * n * nfg
            if data_tmp.sumbox <= 18.0:
                #data_tmp.sumbox = 18.0
                data_new = data_new.append(data_tmp, ignore_index=True)
            #print(data_new.shape)
            #print(data_tmp)
        age_tmp = []
        sumbox_tmp = []
    if nt % 20 == 0:
        print(nt)

data_new = data_new.loc[3:]

data_new.to_csv('./raw data_edit/data_fd_age.csv')




