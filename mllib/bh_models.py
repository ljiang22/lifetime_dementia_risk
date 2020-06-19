import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

class model_base():
    def __init__(self):
        super(model_base, self).__init__()

class bh_model(model_base):
    def __init__(self, sumbox_max=18.0, thd=30.0):
        super(bh_model, self).__init__()
        self.sumbox_max = sumbox_max
        self.data_mean = None
        self.data_std = None
        self.thd = thd
        self.score = None
        self.model = None
        self.input_data = None
        self.input_nor = None
        self.data_pars = None
        self.diff= None
        self.age = None
        self.apoe = None
        self.dem_idx = None
        self.alcohol = None
        self.dep2yrs = None
        self.gds = None
        self.hyperten = None
        self.diabetes= None
        self.b12def = None


    def res_nor(self, input):
        pars = self.data_pars
        Nt = pars.shape[1]
        input_nor = []
        input = np.reshape(input, (Nt))
        #st.write(input)
        #st.write(input.shape, pars.shape)
        for nt in range(Nt):
            x_tmp = (input[nt] - pars[1, nt]) / pars[0, nt]
            input_nor.append(x_tmp)
        input_nor = np.asarray(input_nor)
        #st.write(input_nor)
        return input_nor

    def bh_score(self, input):
        svy_res = input[['commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare']]

        #svy_res = svy_res.value
        svy_res = np.asarray(svy_res)
        svy_res = np.reshape(svy_res, (6))
        #st.write(svy_res.shape)

        sumbox = np.sum(svy_res)
        #st.write(sumbox)
        self.score = 100 - (sumbox / self.sumbox_max) * 100
        return self.score

    def feed_back(self, input):
        svy_res1 = input[['CBTIA', 'ALCOHOL', 'CVAFIB', 'TRAUMBRF', 'TOBAC100', 'PSYCDIS', 'Hand', 'B12DEF', 'CVANGIO',
                        'DEP2YRS', 'Race', 'DEPOTHR', 'THYROID', 'INCONTU', 'CVOTHR', 'PACKSPER', 'M/F', 'HYPERCHO',
                    'apoe', 'LIVSIT', 'dem_idx', 'RESIDENC', 'MARISTAT', 'HYPERTEN', 'SMOKYRS', 'GDS', 'Education',
                          'INDEPEND', 'BMI', 'Age']]  # The feature order is very important. Make sure the order is consistent with the model used in ths code
        self.input_data = svy_res1
        self.data_pars = np.load('./raw data_edit/data_params.npy')  # 0: std, 1: mean, 2: min, 3: max for each variable
        #print(self.data_pars)

        # load the model from disk
        filename = './final models/RF_finalized_age_model.sav'
        self.model = pickle.load(open(filename, 'rb'))
        svy_res2 = np.asarray(svy_res1.values)

        self.input_nor = self.res_nor(svy_res2)
        self.input_nor = np.reshape(self.input_nor, (1, len(self.input_nor)))
        #print(svy_res2.shape)
        y_pred = self.model.predict(self.input_nor)
        self.data_mean = 6.855  # Make sure use the right data mean and std for y_pred
        self.data_std = 5.724
        y_pred = y_pred * self.data_std + self.data_mean
        score_pred = 100 - (y_pred / self.sumbox_max) * 100
        self.diff = score_pred - self.score
        diff_pred = abs(score_pred - self.score) / self.score * 100
        score_pred = score_pred - self.diff  # Calibrate the prediction with clinical dementia rate

        if score_pred >= 80:
            out_put = 'You have a good brain health so far'
        elif score_pred < 80 and score_pred >= 60:
            out_put = 'Your brain health is in avarage. You need to take some actions to maintain or improve your brain health'
        elif score_pred >= 30 and score_pred < 60:
            out_put = 'Your brain health is in a bad condition, and please go to see a doctor as soon as possible'
        else:
            out_put = 'Your brain health is in a worst condition, and please go to see a doctor as soon as possible'

        return out_put, score_pred, diff_pred

    def age_analysis(self):
        age0 = float(self.input_data['Age'])
        age_max = self.data_pars[3, -1]  # Make sure use the the right indices
        age_std = self.data_pars[0, -1]
        age_mean = self.data_pars[1, -1]
        #st.write(age_max, age_std, age_mean)
        score_age = []
        #st.write(age0, age_max)
        if age0 < age_max:
            N = int((age_max - age0) / 2.0)
            for i in range(N):
                age_tmp = age0 + 2.0 * i
                age_tmp_nor = (age_tmp - age_mean) / age_std
                self.input_nor[0, -1] = age_tmp_nor  # Make sure use the right indices. The order matters most!!
                #print(self.input_nor)
                score_tmp = self.model.predict(self.input_nor)
                #st.write(age_tmp_nor, score_tmp)
                y_pred = score_tmp * self.data_std + self.data_mean
                score_pred = 100 - (y_pred / self.sumbox_max) * 100 - self.diff
                if score_pred < 0:
                    score_pred = 0
                score_age.append(score_pred)
                #st.write(score_pred)
        score_age = np.array(score_age)
        score_pred = score_age[0]
        #print(score_age)
        age_dementia_all = []
        idx = []
        if score_pred >= 56:
            for j in range(N):
                if score_age[j] < 56:
                    age_dementia = age0 + j * 2.0
                    age_dementia_all.append(age_dementia)
                    idx.append(j)
                    #time_dementia = j * 2.0
                    #print(age_dementia, time_dementia)

                    #return out_put
            if len(age_dementia_all)>=1:
                out_put = ['You will have a potential dementia risk at age:', age_dementia_all[0]]
            else:
                out_put = 'You have a very low risk to get dementia in your lifetime'
                    #return out_put
        else:
            out_put = 'Your brain health is in a very bad situation. Please go to see a doctor as soon as possible.'
            #return out_put

        x_axis = np.linspace(age0, age_max, N)
        """x_axis = np.linspace(age0, age_max, N)
        font ={'family': 'normal', 'size': 18}
        plt.rc('font', **font)
        if len(idx)>=1:
            plt.figure(1)
            plt.scatter(x_axis, score_age, color='black', label='BHS for all ages')
            plt.scatter(x_axis[idx[0]], score_age[idx[0]], color='red', marker='*', linewidths=5.0,
                        label='Age at dementia')
            plt.legend()
            plt.plot()
            plt.xlabel('Age')
            plt.ylabel('Brain Health Score')

        else:
            plt.figure(1)
            plt.scatter(x_axis, score_age, color='black', label='BHS for all ages')
            plt.legend()
            plt.plot()
            plt.xlabel('Age')
            plt.ylabel('Brain Health Score')

        plt.show()"""


        #print(type(out_put))

        return out_put, x_axis, score_age












