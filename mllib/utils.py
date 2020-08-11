
import pandas as pd
import matplotlib.pyplot as plt


# This function only works for the specific dataset (OASIS 3)
def data_merge(data_tmp1, data_tmp2, data_tmp3, data_tmp4, data_tmp5):
    M1, N1 = data_tmp1.shape
    M2, N2 = data_tmp2.shape
    id1 = data_tmp1['ADRC_ADRCCLINICALDATA ID']
    id2 = data_tmp2['UDS_A1SUBDEMODATA ID']
    j = 0
    idx = []
    idx1 = []

    # Warning: the data needs to be edited manually due to the wrong inputs
    for i in range(M1):
        if j < M2 - 1 and i < M1 - 1:
            tmp1 = id1[i]
            tmp1 = int(tmp1[23:])
            tmp1c = data_tmp1.Subject[i]
            tmp1d = data_tmp1.Subject[i + 1]
            tmp2 = id2[j]
            tmp2 = int(tmp2[16:])
            tmp2c = data_tmp2.Subject[j]
            tmp2d = data_tmp2.Subject[j + 1]

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

                #print(tmp1c, tmp2c, i, j - 1, tmp1, tmp2)
                idx.append(i)
                idx1.append(j_tmp)

            if abs(tmp1 - tmp2) > 150 and tmp1c == tmp2c and tmp2c != tmp2d and tmp1c != tmp1d:
                j += 1
                #print(tmp1c, tmp2c, i, j - 1, tmp1, tmp2)
        #else:
            #if tmp1d == tmp2d:
                #print(tmp1d, tmp2d, i, j)

    data_tmp1_edit = pd.DataFrame(data_tmp1,
                                  columns=['ADRC_ADRCCLINICALDATA ID', 'Subject', 'Age', 'mmse', 'ageAtEntry',
                                           'cdr', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare',
                                           'apoe', 'sumbox', 'M/F', 'Hand',
                                           'Education', 'Race', 'BMI'], index=idx)
    data_tmp2_edit = pd.DataFrame(data_tmp2,
                                  columns=['UDS_A1SUBDEMODATA ID', 'Subject', 'LIVSIT', 'INDEPEND', 'RESIDENC',
                                           'MARISTAT'], index=idx1)
    data_tmp3_edit = pd.DataFrame(data_tmp3, columns=['UDS_A3SBFMHSTDATA ID', 'Subject', 'dem_idx'], index=idx1)
    data_tmp4_edit = pd.DataFrame(data_tmp4, columns=['UDS_A5SUBHSTDATA ID', 'Subject', 'CVHATT', 'CVAFIB', 'CVANGIO',
                                                      'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA',
                                                      'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF',
                                                      'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO',
                                                      'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF',
                                                      'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'TOBAC30', 'TOBAC100', 'SMOKYRS',
                                                      'PACKSPER', 'ABUSOTHR', 'PSYCDIS'], index=idx1)
    data_tmp5_edit = pd.DataFrame(data_tmp5, columns=['UDS_B6BEVGDSDATA ID', 'Subject', 'GDS'], index=idx1)

    # Quality control
    # Reindex the dataset in a continuous number
    data_tmp1_edit.reset_index(drop=True, inplace=True)
    data_tmp2_edit.reset_index(drop=True, inplace=True)
    data_tmp3_edit.reset_index(drop=True, inplace=True)
    data_tmp4_edit.reset_index(drop=True, inplace=True)
    data_tmp5_edit.reset_index(drop=True, inplace=True)

    # Quality control on the edited data
    for i in range(len(idx)):
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

    data_all = pd.concat([data_tmp1_edit, data_tmp2_edit, data_tmp3_edit, data_tmp4_edit, data_tmp5_edit], axis=1)

    M, N = data_all.shape
    # Feature engineering
    # Remove the warning of modifying a dataframe
    pd.options.mode.chained_assignment = None  # default='warn'
    data_all_edit = data_all.copy()
    for i in range(M):
        if data_all.LIVSIT[i] == 9:
            data_all.LIVSIT[i] = 1.0
        data_all_edit.LIVSIT[i] = data_all.LIVSIT[i] - 1.0

        if data_all.INDEPEND[i] == 9:
            data_all.INDEPEND[i] = 1.0
        data_all_edit.INDEPEND[i] = data_all.INDEPEND[i] - 1.0

        if data_all.RESIDENC[i] == 9:
            data_all.RESIDENC[i] = 5.0
        data_all_edit.RESIDENC[i] = data_all.RESIDENC[i] - 1.0

        # Treate 'living as married" as 'married'
        if data_all.MARISTAT[i] == 1:
            data_all_edit.MARISTAT[i] = 0.0

        if data_all.MARISTAT[i] == 6:
            data_all_edit.MARISTAT[i] = 1.0

        if data_all.MARISTAT[i] == 9:
            data_all_edit.MARISTAT[i] = 2.0

        if data_all.MARISTAT[i] == 8:
            data_all_edit.MARISTAT[i] = 2.0

        if data_all.TOBAC30[i] > 1:
            data_all_edit.TOBAC30[i] = 1

        if data_all.TOBAC100[i] > 1:
            data_all_edit.TOBAC100[i] = 1

    return data_all_edit


# PCA analysis
from sklearn.decomposition import PCA
def pca_analysis(features, thd=0.95):
    pca = PCA(n_components=features.shape[1])
    pca.fit(features)
    var = pca.explained_variance_ratio_
    var_tmp = 0
    var_per = []
    n_component = []
    for i in range(len(var)):
        var_tmp += var[i]
        var_per.append(1 - var_tmp)
        if var_tmp >= thd:
            n_tmp = i+1
            n_component.append(n_tmp)

    pca_opt = PCA(n_components=min(n_component))
    pca_opt.fit(features)
    features_new = pca_opt.transform(features)

    # Define the font size
    font = {'size': 18}
    plt.rc('font', **font)

    plt.figure(1)
    plt.plot(var_per, '.-', color='black')
    # plt.grid()
    plt.xlabel('Principal components')
    plt.ylabel('Percenage of unexplained variances')
    plt.show()

    return features_new

import numpy as np
import sklearn
def data_split(data, subject, rdst=42):
    sub_id = subject['Subject'].unique()
    len0 = len(sub_id)
    idx = np.arange(len0)
    idx = sklearn.utils.shuffle(idx, random_state=rdst)
    test_len = int(len0 * 0.15)
    val_len = int((len0 - test_len) * 0.15)
    subject_id = subject.values
    data_edit = np.concatenate((data, subject_id), axis=1)
    print(data_edit.shape)

    test_id_group = sub_id[idx[0:test_len]]
    val_id_group = sub_id[idx[test_len: test_len + val_len]]
    train_id_group = sub_id[idx[test_len + val_len:]]
    print(len(test_id_group))

    train_set = []
    val_set = []
    test_set = []
    n_feature = data_edit.shape[1] - 1
    for n in range(data_edit.shape[0]):
        # print(data_nor[n, n_feature])
        if data_edit[n, n_feature] in test_id_group:
            test_set.append(data_edit[n, :])
        elif data_edit[n, n_feature] in val_id_group:
            val_set.append(data_edit[n, :])
        else:
            train_set.append(data_edit[n, :])
    print(len(test_set), len(val_set), len(train_set))
    test_set = np.asarray(test_set)
    val_set = np.asarray(val_set)
    train_set = np.asarray(train_set)
    print(test_set.shape)

    X_train = train_set[:, 0:n_feature-1]
    y_train = train_set[:, n_feature-1]
    X_val = val_set[:, 0:n_feature-1]
    y_val = val_set[:, n_feature-1]
    X_train0 = np.concatenate((X_train, X_val), axis=0)
    y_train0 = np.concatenate((y_train, y_val))
    X_test = test_set[:, 0:n_feature-1]
    y_test = test_set[:, n_feature-1]
    print(X_train0.shape, y_train0.shape, X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape,
          y_test.shape)

    return X_train0, y_train0, X_train, X_val, X_test, y_train, y_val, y_test

from sklearn.metrics import r2_score
def evaluate(y_real, y_predict):
    y_predict = np.asarray(y_predict, dtype=np.float32)
    y_real = np.asarray(y_real, dtype=np.float32)
    print(y_real.shape, y_predict.shape)
    errors = abs(y_predict - y_real)
    err_avg = np.mean(errors)
    r2 = r2_score(y_real, y_predict)
    corr = np.corrcoef(y_real, y_predict)
    return r2, err_avg, corr[0, 1]

# Extrapolate the data to older ages using linear relationship between CDR and age
def data_augmentation(data_input):
    Nt = data_input.shape[0]
    age_max = 110
    data_new = data_input.loc[0:2]
    age_tmp = []
    sumbox_tmp = []
    aging_avg = 0.5
    nfg = 2
    for nt in range(Nt - 1):
        id1 = data_input.Subject[nt]
        id2 = data_input.Subject[nt + 1]

        if id1 == id2:
            # print(nt, data_input.Age[nt])
            age_tmp.append(data_input.Age[nt])
            sumbox_tmp.append(data_input.sumbox[nt])
        else:
            age_tmp.append(data_input.Age[nt])
            sumbox_tmp.append(data_input.sumbox[nt])
            M = len(age_tmp)
            N = int((age_max - age_tmp[M - 1]) / nfg)
            data_new_tmp = data_input.loc[nt - M + 1: nt]
            data_tmp = data_input.loc[nt]
            # print(nt, M, N, age_tmp, age_tmp[M-1])
            # print(data_new_tmp)
            data_new = data_new.append(data_new_tmp, ignore_index=True)

            # Calculate the brain health declining rate with age. The assumption is that other factors keep constant.
            if M > 2:
                # print(sumbox_tmp, age_tmp)
                fit_coef = np.polyfit(age_tmp, sumbox_tmp, 1)
                # print(nt, M, fit_coef[0])
                # aging_rate = (sumbox_tmp[M-1] - sumbox_tmp[0]) / (age_tmp[M-1]- age_tmp[0])
                aging_rate = fit_coef[0]
                """if aging_rate < 0:
                    aging_rate = 0.0

                if aging_rate < aging_avg:
                    if age_tmp[M-1] < 70:
                        aging_rate = 0.05
                    else:
                        aging_rate = 0.07"""
                if aging_rate > aging_avg:
                    for n in range(1, N):
                        data_tmp.Age = age_tmp[M - 1] + n * nfg

                        data_tmp.sumbox = sumbox_tmp[M - 1] + aging_rate * n * nfg
                        # if data_tmp.sumbox >= 18.0:
                        # data_tmp.sumbox = 18.0
                        if data_tmp.sumbox <= 18.0:
                            data_new = data_new.append(data_tmp, ignore_index=True)

            age_tmp = []
            sumbox_tmp = []
        if nt % 20 == 0:
            print(nt)

    data_new = data_new.loc[3:]
    data_new.to_csv('./raw data_edit/data_fd_age.csv')
    return data_new


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
import pickle
def predictive_model(training_method, X_train0, y_train0, X_train, X_val, X_test, y_train, y_val, y_test, keys0):
    train_real = y_train
    val_real = y_val
    test_real = y_test
    keys0 = keys0[0:-1]

    if training_method == 'RF':
        """param_grid = {'n_estimators': [10, 50, 100, 150, 200, 400, 600],
                      'min_samples_split': [2, 4, 6, 8, 10],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'min_samples_leaf': [1, 3, 5],
                      'bootstrap': [True],
                      'max_samples': [50, 100, 200, 500, 1000]}"""

        """grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
        grid.fit(X_train0, y_train0)
        print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
        par_opt = grid.best_params_
        print("Best parameters: ", grid.best_params_)
        print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))"""

        """Best cross-validation accuracy: 0.62
        'bootstrap': True, 'max_features': 'auto', 'max_samples': 1000, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 600
        Test set accuracy: 0.58"""

        rf_estimator = RandomForestRegressor(bootstrap=True, max_features='log2', max_depth=8, min_samples_split=2,
                        max_samples=2500, n_estimators=500, min_samples_leaf=2, verbose=0, random_state=42, n_jobs=6)
        rf_estimator.fit(X_train, y_train)
        train_pred_nor = rf_estimator.predict(X_train)
        val_pred_nor = rf_estimator.predict(X_val)
        test_pred_nor = rf_estimator.predict(X_test)

        # Feature importance analysis
        feature_imp = rf_estimator.feature_importances_
        print(sorted(feature_imp))
        indices4 = np.argsort(feature_imp)
        keys1 = keys0[indices4]
        print('keys1 is', keys1)

        filename = './final models/' + training_method + '_finalized_model.sav'
        model = rf_estimator
        pickle.dump(model, open(filename, 'wb'))

        # Calculate the prediction for training data
        train_pred = train_pred_nor
        val_pred = val_pred_nor
        test_pred = test_pred_nor

    if training_method == 'xgboost':
        # for illustration purposes only, don't use this code!
        """param_grid = {'n_estimators': [150, 200, 250],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2],
                      'max_depth': [11, 13, 15],
                      'colsample_bytree': [0.6, 0.7, 0.8],
                      'reg_alpha': [3, 5, 7]
                      }

        grid = GridSearchCV(XGBRegressor(), param_grid=param_grid, cv=5)
        grid.fit(X_train0, y_train0)
        print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
        par_opt = grid.best_params_
        print("Best parameters: ", grid.best_params_)
        print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))"""

        """Best cross-validation accuracy: 0.38
           Best parameters:  {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 13, 'n_estimators': 250, 'reg_alpha': 3}
           Test set accuracy: 0.51"""

        # rf_estimator = XGBRegressor(learning_rate=0.05, max_depth=13, colsample_bytree=0.7,
        # n_estimators=250, reg_alpha=3, random_state=0, n_jobs=5)
        xgb_estimator = XGBRegressor(learning_rate=0.03, max_depth=8,
                                     n_estimators=500, random_state=42, n_jobs=5)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        xgb_estimator.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=True)
        results = xgb_estimator.evals_result()
        xgb_estimator.fit(X_train, y_train, eval_metric="mae", early_stopping_rounds=10, eval_set=eval_set,
                          verbose=True)

        train_pred_nor = xgb_estimator.predict(X_train)
        val_pred_nor = xgb_estimator.predict(X_val)
        test_pred_nor = xgb_estimator.predict(X_test)

        # Feature importance analysis
        feature_imp = xgb_estimator.feature_importances_
        print(sorted(feature_imp))
        indices1 = np.argsort(feature_imp)
        keys1 = keys0[indices4]
        print('keys1 is', keys1)

        filename = './final models/' + training_method + '_finalized_model.sav'
        model = xgb_estimator
        pickle.dump(model, open(filename, 'wb'))

        # Calculate the prediction for training data
        train_pred = train_pred_nor
        val_pred = val_pred_nor
        test_pred = test_pred_nor

    r2_train, err_train, corr_train = evaluate(train_real, train_pred)
    print(training_method + ' Train R2 score = {:0.2f}'.format(r2_train))
    print(training_method + ' Train avg. error = {:0.3f}'.format(err_train))
    print(training_method + ' Train correlation = {:0.2f}'.format(corr_train))

    r2_train, err_train, corr_train = evaluate(val_real, val_pred)
    print(training_method + ' Validation R2 score = {:0.2f}'.format(r2_train))
    print(training_method + ' Validation avg. error = {:0.3f}'.format(err_train))
    print(training_method + ' Validation correlation = {:0.2f}'.format(corr_train))

    r2_train, err_train, corr_train = evaluate(test_real, test_pred)
    print(training_method + ' Test R2 score = {:0.2f}'.format(r2_train))
    print(training_method + ' Test avg. error = {:0.3f}'.format(err_train))
    print(training_method + ' Test correlation = {:0.2f}'.format(corr_train))

    # Plot training curve
    # epochs = len(results['validation_0']['mae'])
    # x_axis = range(0, epochs)

    # Feature importance analysis
    keys1 = ['Psychiatric disorders', 'B12 deficiency', 'Hand', 'Hypertension', 'Depression (Other)',
             'Hypercholesterolemia', 'Angioplasty', 'Cardiovascular (others)', 'Marriage state', 'Packs per day', 'Incontinence (urinary)',
             'Residence type', 'Family history', 'Gender', 'Smoking years', 'Depression (2 years)', 'Incontinence (bowel)',
             'Living situation', 'APOE', 'Education', 'BMI', 'GDS', 'Age']

    font = {'size': 18}
    plt.rc('font', **font)

    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices4)), sorted(feature_imp), color='b', align='center')
    plt.yticks(range(len(indices4)), [keys1[i] for i in range(len(indices4))])
    plt.xlabel('Relative Importance')
    plt.show()

    plt.figure(2)
    plt.scatter(train_real, train_pred, color='black')
    plt.xlabel("CDR")
    plt.ylabel("CDR (Predicted)")
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)

    plt.figure(3)
    plt.scatter(val_real, val_pred, color='black')
    plt.xlabel("CDR")
    plt.ylabel("CDR (Predicted)")
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)

    plt.figure(4)
    plt.scatter(test_real, test_pred, color='black')
    plt.xlabel("CDR")
    plt.ylabel("CDR (Predicted)")
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)

    """plt.figure(5)
    plt.plot(x_axis, results['validation_0']['mae'], label='Train', color='black')
    plt.plot(x_axis, results['validation_1']['mae'], label='Validation', color='r')
    plt.legend()
    plt.xlabel('Number of estimator')
    plt.ylabel('Mean absolute error')
    #plt.title('XGBoost Log Loss')"""

    plt.show()


def predictive_model_aug(training_method, X_train0, y_train0, X_train, X_val, X_test, y_train, y_val, y_test, keys0):
    train_real = y_train
    val_real = y_val
    test_real = y_test
    keys0 = keys0[0:-1]

    if training_method == 'RF':
        """param_grid = {'n_estimators': [10, 50, 100, 150, 200, 400, 600],
                      'min_samples_split': [2, 4, 6, 8, 10],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'min_samples_leaf': [1, 3, 5],
                      'bootstrap': [True],
                      'max_samples': [50, 100, 200, 500, 1000]}"""

        """grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
        grid.fit(X_train0, y_train0)
        print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
        par_opt = grid.best_params_
        print("Best parameters: ", grid.best_params_)
        print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))"""

        """Best cross-validation accuracy: 0.62
        'bootstrap': True, 'max_features': 'auto', 'max_samples': 1000, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 600
        Test set accuracy: 0.58"""

        rf_estimator = RandomForestRegressor(bootstrap=True, max_features='log2', max_depth=8, min_samples_split=2,
                        max_samples=2800, n_estimators=600, min_samples_leaf=2, verbose=0, random_state=42, n_jobs=6)
        rf_estimator.fit(X_train, y_train)
        train_pred_nor = rf_estimator.predict(X_train)
        val_pred_nor = rf_estimator.predict(X_val)
        test_pred_nor = rf_estimator.predict(X_test)

        # Feature importance analysis
        feature_imp = rf_estimator.feature_importances_
        print(sorted(feature_imp))
        indices4 = np.argsort(feature_imp)
        keys1 = keys0[indices4]
        print('keys1 is', keys1)

        filename = './final models/' + training_method + '_finalized_model_aug.sav'
        model = rf_estimator
        pickle.dump(model, open(filename, 'wb'))

        # Calculate the prediction for training data
        train_pred = train_pred_nor
        val_pred = val_pred_nor
        test_pred = test_pred_nor

    if training_method == 'xgboost':
        # for illustration purposes only, don't use this code!
        """param_grid = {'n_estimators': [150, 200, 250],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2],
                      'max_depth': [11, 13, 15],
                      'colsample_bytree': [0.6, 0.7, 0.8],
                      'reg_alpha': [3, 5, 7]
                      }

        grid = GridSearchCV(XGBRegressor(), param_grid=param_grid, cv=5)
        grid.fit(X_train0, y_train0)
        print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
        par_opt = grid.best_params_
        print("Best parameters: ", grid.best_params_)
        print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))"""

        """Best cross-validation accuracy: 0.38
           Best parameters:  {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 13, 'n_estimators': 250, 'reg_alpha': 3}
           Test set accuracy: 0.51"""

        # rf_estimator = XGBRegressor(learning_rate=0.05, max_depth=13, colsample_bytree=0.7,
        # n_estimators=250, reg_alpha=3, random_state=0, n_jobs=5)
        xgb_estimator = XGBRegressor(learning_rate=0.03, max_depth=8,
                                     n_estimators=500, random_state=42, n_jobs=5)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        xgb_estimator.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=True)
        results = xgb_estimator.evals_result()
        xgb_estimator.fit(X_train, y_train, eval_metric="mae", early_stopping_rounds=10, eval_set=eval_set,
                          verbose=True)

        train_pred_nor = xgb_estimator.predict(X_train)
        val_pred_nor = xgb_estimator.predict(X_val)
        test_pred_nor = xgb_estimator.predict(X_test)

        # Feature importance analysis
        feature_imp = xgb_estimator.feature_importances_
        print(sorted(feature_imp))
        indices1 = np.argsort(feature_imp)
        keys1 = keys0[indices4]
        print('keys1 is', keys1)

        filename = './final models/' + training_method + '_finalized_model.sav'
        model = xgb_estimator
        pickle.dump(model, open(filename, 'wb'))

        # Calculate the prediction for training data
        train_pred = train_pred_nor
        val_pred = val_pred_nor
        test_pred = test_pred_nor

    r2_train, err_train, corr_train = evaluate(train_real, train_pred)
    print(training_method + ' Train R2 score = {:0.2f}'.format(r2_train))
    print(training_method + ' Train avg. error = {:0.3f}'.format(err_train))
    print(training_method + ' Train correlation = {:0.2f}'.format(corr_train))

    r2_train, err_train, corr_train = evaluate(val_real, val_pred)
    print(training_method + ' Validation R2 score = {:0.2f}'.format(r2_train))
    print(training_method + ' Validation avg. error = {:0.3f}'.format(err_train))
    print(training_method + ' Validation correlation = {:0.2f}'.format(corr_train))

    r2_train, err_train, corr_train = evaluate(test_real, test_pred)
    print(training_method + ' Test R2 score = {:0.2f}'.format(r2_train))
    print(training_method + ' Test avg. error = {:0.3f}'.format(err_train))
    print(training_method + ' Test correlation = {:0.2f}'.format(corr_train))

    # Find the data with a bad prediction
    thd = 5.0
    test_set_bad = []
    test_pred_good = []
    test_real_good = []
    for ns in range(len(test_pred)):
        diff = test_pred[ns] - test_real[ns]
        if abs(diff) >= thd:
            data_tmp = X_test[ns, :]
            test_set_bad.append(data_tmp)
            # print(data_tmp[n_feature], data_tmp[0], data_tmp[3])
        else:
            test_pred_good.append(test_pred[ns])
            test_real_good.append(test_real[ns])
    test_set_bad = np.asarray(test_set_bad)
    print(test_set_bad.shape)
    print(len(test_set_bad))

    r2_train, err_train, corr_train = evaluate(test_real_good, test_pred_good)
    print(training_method + ' Test R2 score for good test set= {:0.2f}'.format(r2_train))
    print(training_method + ' Test avg. error for good test set= {:0.3f}'.format(err_train))
    print(training_method + ' Test correlation for good test set= {:0.2f}'.format(corr_train))

    font = {'size': 18}
    plt.rc('font', **font)

    plt.figure(1)
    plt.scatter(train_real, train_pred, color='black')
    plt.xlabel("CDR")
    plt.ylabel("CDR (Predicted)")
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)

    plt.figure(2)
    plt.scatter(val_real, val_pred, color='black')
    plt.xlabel("CDR")
    plt.ylabel("CDR (Predicted)")
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)

    plt.figure(3)
    plt.scatter(test_real, test_pred, color='black')
    plt.xlabel("CDR")
    plt.ylabel("CDR (Predicted)")
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)

    plt.figure(4)
    plt.scatter(test_real_good, test_pred_good, color='black')
    plt.xlabel("CDR")
    plt.ylabel("CDR (Predicted)")
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)

    plt.show()

