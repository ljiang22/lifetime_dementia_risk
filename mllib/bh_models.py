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
        svy_res1 = input[['Age', 'BMI', 'Education', 'GDS', 'LIVSIT', 'INCONTU', 'apoe', 'RESIDENC', 'INCONTF', 'SMOKYRS',
                          'B12DEF', 'DEP2YRS', 'dem_idx', 'M/F', 'MARISTAT', 'PACKSPER', 'HYPERTEN', 'HYPERCHO', 'CVOTHR',
                          'DEPOTHR', 'CVANGIO', 'Hand', 'PSYCDIS']]  # The feature order is very important. Make sure the order is consistent with the model used in ths code

        self.input_data = svy_res1

        # load the model from disk
        filename = './final models/RF_finalized_model_aug.sav'
        self.model = pickle.load(open(filename, 'rb'))
        svy_res2 = np.asarray(svy_res1.values)
        #st.write(svy_res2.shape)

        self.input_nor = svy_res2
        self.input_nor = np.reshape(self.input_nor, (1, np.size(self.input_nor)))
        #print(svy_res2.shape)
        y_pred = self.model.predict(self.input_nor)
        score_pred = 100 - (y_pred / self.sumbox_max) * 100
        self.diff = score_pred - self.score
        diff_pred = abs(score_pred - self.score) / self.score * 100
        score_pred = score_pred - self.diff  # Calibrate the prediction with clinical dementia rate

        if score_pred >= 80:
            out_put = 'You have a good brain health so far'
        elif score_pred < 80 and score_pred >= 60:
            out_put = 'Your brain health is in avarage. You need to take some actions to improve your brain health'
        elif score_pred >= 30 and score_pred < 60:
            out_put = 'Your brain health is in a bad condition, and please go to see a doctor as soon as possible'
        else:
            out_put = 'Your brain health is in a worst condition, and please go to see a doctor as soon as possible'

        return out_put, score_pred, diff_pred

    def age_analysis(self):
        age0 = float(self.input_data['Age'])
        age_max = 110  # Make sure use the the right indices

        #st.write(age_max, age_std, age_mean)
        score_age = []
        #st.write(self.input_nor[0, -2])

        if age0 < age_max:
            N = int((age_max - age0) / 2.0)
            for i in range(N):
                age_tmp = age0 + 2.0 * i
                age_tmp_nor = age_tmp
                self.input_nor[0, 0] = age_tmp_nor  # Make sure use the right indices. The order matters most!!
                #print(self.input_nor)
                score_tmp = self.model.predict(self.input_nor)
                #st.write(age_tmp_nor, score_tmp)
                y_pred = score_tmp
                score_pred = 100 - (y_pred / self.sumbox_max) * 100 - self.diff
                if score_pred < 0:
                    score_pred = 0
                score_age.append(score_pred)

        score_age = np.array(score_age)
        #st.write(score_age[0], len(score_age), self.diff)
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

    def risk_factor(self):
        factors = []

        bmi = float(self.input_data['BMI'])
        #st.write(bmi, bmi_max, bmi_mean)
        if bmi >= 25.0:
            factors.append('Overweight')
            bmi_tmp = 20.0
            bmi_tmp_nor = bmi_tmp
            self.input_nor[0, 1] = bmi_tmp_nor  # Make sure use the right indices. The order matters most!!
            #st.write(self.input_nor[0, -2])

        gds = float(self.input_data['GDS'])
        if gds <= 7.0:
            factors.append('Depression')
            gds_tmp = 15.0
            gds_tmp_nor = gds_tmp
            #st.write(self.input_nor[0, -2])
            self.input_nor[0, 3] = gds_tmp_nor  # Make sure use the right indices. The order matters most!!

        edu = float(self.input_data['Education'])
        #st.write(edu, edu_max, edu_mean)
        if edu <= 15.0:
            #factors.append('Education')
            edu_tmp = 20.0
            edu_tmp_nor = edu_tmp
            #st.write(self.input_nor[0, -2])
            self.input_nor[0, 2] = edu_tmp_nor  # Make sure use the right indices. The order matters most!!


        smokyrs = float(self.input_data['SMOKYRS'])
        if smokyrs >= 20.0:
            factors.append('Smoking')
            smokyrs_tmp = 0.0
            smokyrs_tmp_nor = smokyrs_tmp
            #st.write(self.input_nor[0, -2])
            self.input_nor[0, 9] = smokyrs_tmp_nor  # Make sure use the right indices. The order matters most!!



        #st.write(hyperten, hyperten_max, hyperten_mean)

        gene = float(self.input_data['apoe'])
        if gene == 0.0:
            factors.append('Gene')

        hyperten = float(self.input_data['HYPERTEN'])
        if hyperten >= 1.0:
            factors.append('Hypertension')
            hyperten_tmp = 0.0
            hyperten_tmp_nor = hyperten_tmp
            self.input_nor[0, 16] = hyperten_tmp_nor  # Make sure use the right indices. The order matters most!!


        """out_put, x_axis, score_age = self.age_analysis()

        score_age1 = []
        for i in range(len(score_age)):
            score_tmp = float(score_age[i])
            # st.write(i, score_tmp)
            score_age1.append(score_tmp)
        score_age1 = np.asarray(score_age1)"""

        return factors

    """def risk_bmi(self):
        bmi = float(self.input_data['BMI'])
        bmi_max = self.data_pars[3, -2]  # Make sure use the the right indices
        bmi_std = self.data_pars[0, -2]
        bmi_mean = self.data_pars[1, -2]
        #st.write(bmi, bmi_max, bmi_mean)
        if bmi >= 25.0:
            bmi_tmp = 20.0
            bmi_tmp_nor = (bmi_tmp - bmi_mean) / bmi_std
            self.input_nor[0, -2] = bmi_tmp_nor  # Make sure use the right indices. The order matters most!!
            #st.write(self.input_nor[0, -2])
            out_put, x_axis, score_age = self.age_analysis()

        score_age1 = []
        for i in range(len(score_age)):
            score_tmp = float(score_age[i])
            # st.write(i, score_tmp)
            score_age1.append(score_tmp)
        score_age1 = np.asarray(score_age1)

        return out_put, score_age1

    def risk_edu(self):
        bmi = float(self.input_data['Education'])
        bmi_max = self.data_pars[3, -4]  # Make sure use the the right indices
        bmi_std = self.data_pars[0, -4]
        bmi_mean = self.data_pars[1, -4]
        #st.write(bmi, bmi_max, bmi_mean)
        if bmi <= 15.0:
            bmi_tmp = 20.0
            bmi_tmp_nor = (bmi_tmp - bmi_mean) / bmi_std
            #st.write(self.input_nor[0, -2])
            self.input_nor[0, -4] = bmi_tmp_nor  # Make sure use the right indices. The order matters most!!
            #st.write(self.input_nor[0, -2])
            out_put, x_axis, score_age = self.age_analysis()

        score_age1 = []
        for i in range(len(score_age)):
            score_tmp = float(score_age[i])
            # st.write(i, score_tmp)
            score_age1.append(score_tmp)
        score_age1 = np.asarray(score_age1)

        return out_put, score_age1


    def risk_gds(self):
        bmi = float(self.input_data['GDS'])
        bmi_max = self.data_pars[3, -5]  # Make sure use the the right indices
        bmi_std = self.data_pars[0, -5]
        bmi_mean = self.data_pars[1, -5]
        #st.write(bmi, bmi_max, bmi_mean)
        if bmi <= 7.0:
            bmi_tmp = 13.0
            bmi_tmp_nor = (bmi_tmp - bmi_mean) / bmi_std
            #st.write(self.input_nor[0, -2])
            self.input_nor[0, -5] = bmi_tmp_nor  # Make sure use the right indices. The order matters most!!
            #st.write(self.input_nor[0, -2])
            out_put, x_axis, score_age = self.age_analysis()

        score_age1 = []
        for i in range(len(score_age)):
            score_tmp = float(score_age[i])
            # st.write(i, score_tmp)
            score_age1.append(score_tmp)
        score_age1 = np.asarray(score_age1)

        return out_put, score_age1"""



















