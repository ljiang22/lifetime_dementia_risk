# Try to treat this issue as a classification problem, but the result is not good, the data with cdr greater than 0.5 did not work well, so, needs to go back to regression model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from mllib.preprocess import normalize
from mllib.preprocess import unnormalize
from mllib.networks import evaluate
from mllib.networks import MLP_train_opt
from mllib.networks import MLP_plot
from mllib.networks import MLP_REG_v1
from mllib.networks import MLPR_v0
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import sklearn
from sklearn.metrics import SCORERS
from sklearn.metrics import confusion_matrix
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

data_input_ori = data_input_ori.drop(columns=['Subject'])
keys = data_input_ori.keys()
print(keys)

var = 0.0   # Remove features with low variance. 0.8 means remove all features that are either one or zero (on or off) in more than 80% of the samples
sel = VarianceThreshold(threshold=var)
data_edit = sel.fit_transform(data_input_ori)
indices = sel.get_support(indices=True)
print(indices)
keys1 = keys[indices]
keys1 = keys1[2:]
print(keys1)

M, N = data_input_ori.shape
print(data_input_ori.shape, data_edit.shape, type(data_edit))

#print(data_edit[0:50])

data_input = data_edit
print(data_input.shape, type(data_input))
#print(max(data_input[:, 1]))

label = data_input[:, 1]
for i in range(M):
    if label[i] < 0.5:
        label[i] = 0
        #print(i, label[i])
    elif label[i] >= 0.5 and label[i] < 5:
        label[i] = 1
        #print(i, label[i])
    elif label[i] >= 5 and label[i] < 8:
        label[i] = 2
    elif label[i] >= 8 and label[i] < 15:
        label[i] = 3
    else:
        label[i] = 3
        print(i, label[i])
label = np.reshape(label, (M, 1))


from sklearn.preprocessing import OneHotEncoder
# Setting sparse=False means OneHotEncode will return a numpy array, not a sparse matrix
ohe = OneHotEncoder(sparse=False)
label_ohe = ohe.fit_transform(label)
print(label_ohe.shape)

print(sorted(sklearn.metrics.SCORERS.keys()))

data_nor = data_input[:, 2:]
print('data_nor shape is', data_nor.shape)

features = data_nor
print(features.shape, label_ohe.shape)

N1 = 45
from sklearn.feature_selection import chi2
sltk = SelectKBest(chi2, k=N1)
fv, pv = chi2(features, label_ohe)
features_new = sltk.fit_transform(features, label_ohe)
indices2 = sltk.get_support(indices=True)
keys2 = keys1[indices2]
print(keys2)
# 'Race', Atrial fibrillation (CVAFIB), 'DIABETES', Thyroid disease (THYROID), 100 lifetime cigarettes (TOBAC100), 'PACKSPER' - from regression
# 'Hand', 'DIABETES', 'Race', 'MARISTAT', 'TOBAC100' ?

print(features_new.shape)
indices3 = np.argsort(pv)
print(keys1[indices3])
indices3 = indices3[0:N1]
pv = sorted(pv)
print(len(pv), pv)
print(indices3)
keys3 = keys1[indices3]
print('keys3 is', keys3)
features_new = features[:, indices3]
x_axis = np.linspace(1, len(pv), len(pv))
features_new = features_new[:, 0:]

font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)
"""plt.figure(1)
plt.scatter(x_axis, pv, color='black')
plt.xlabel("Feature")
plt. ylabel("P-value")
#plt.xlim(-2, 20)
#plt.ylim(-2, 20)
plt.show()"""

# L1-based feature selection
from sklearn.feature_selection import SelectFromModel
N2 = 45
rf_estimator = RandomForestClassifier(bootstrap=True, max_samples=3000, max_features='auto', min_samples_split=2,
                                     n_estimators=200, verbose=0, random_state=42, n_jobs=6)
rf_fit = rf_estimator.fit(features_new, label_ohe)
feature_imp = rf_fit.feature_importances_
print(feature_imp)
indices4 = np.argsort(feature_imp)
keys4 = keys3[indices4]
print('keys4 is:', keys4)
print(sorted(feature_imp))
model = SelectFromModel(rf_fit, threshold=0.000001, prefit=True, max_features=N2)
features_new = model.transform(features_new)
print(features_new.shape)

#print(data_nor[0:10, :])

from sklearn.model_selection import train_test_split
X_train0, X_test, y_train0, y_test = train_test_split(features_new, label, test_size=0.15, random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.10, random_state=22)
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

if training_method == 'MLP':
    # hyper-parameters
    learning_rate = [0.0005, 0.001, 0.003, 0.005]  # The optimization learning rate
    epochs = [20, 40]  # Total number of training epochs, 25 might be best
    batch_size = [30, 70, 100, 150]  # Training batch size
    h1 = [20, 40, 60, 80]  # number of units in first hidden layer
    h2 = [5, 10, 15, 20]
    """learning_rate = [0.001, 0.003]  # The optimization learning rate
    epochs = [5, 10]  # Total number of training epochs, 25 might be best
    batch_size = [30, 50]  # Training batch size
    h1 = [40, 60]  # number of units in first hidden layer
    h2 = [10, 15]
    ncls = 1  # number of classes"""
    display_freq = 100  # Frequency of displaying the training results

    if opt_flag == 1:
        lro, epocho, batcho, h1o, h2o = MLP_train_opt(X_train, y_train, X_val, y_val, X_test, y_test,
                            display_freq, data_mean, data_std, learning_rate, epochs, batch_size, h1, h2)

        lr = lro  # The optimization learning rate
        epoch = epocho  # Total number of training epochs, 25 might be best
        batch_size = batcho  # Training batch size
        display_freq = 50  # Frequency of displaying the training results
        h1 = h1o  # number of units in first hidden layer
        h2 = h2o
        h3 = 1
        print('the optimal hyper parameter of is:', lro, epocho, batcho, h1o, h2o)
    else:
        print('Use the default hyper parameters:')
        lr = 0.0005 # The optimization learning rate
        epoch = 80  # Total number of training epochs, 25 might be best
        batch_size = 50  # Training batch size
        h1 = 30  # Number of units in first hidden layer
        h2 = 10
        h3 = 1
        display_freq = 50  # Frequency of displaying the training results

    wt1, b1, wt2, b2, wt3, b3, epoch_tmp = MLP_plot(X_train, y_train, X_val, y_val, X_test, y_test, h1,
                                                                  h2, h3, lr, epoch, batch_size, display_freq)

    epoch = epoch_tmp # After optimization
    #epoch = 3  # After optimization
    act_func = "relu"
    model_mlp = MLPR_v0(h1, h2, h3, batch_size=batch_size, max_epoch=epoch, lr0=lr, act_func=act_func)
    model_mlp.fit(X_train, y_train, X_val, y_val)
    train_pred = model_mlp.predict(X_train)
    val_pred = model_mlp.predict(X_val)
    test_pred = model_mlp.predict(X_test)
    #print('y_val is', val_pred.shape, y_val.shape)

    # Calculate the prediction for training data


    #print('Test set score is', test_score)
    model_mlp.lossPlot()
    model_mlp.lossPlotE()

if training_method == 'SVR':
    kel = 'rbf'
    c = 1.0
    eps = 0.3
    ga = 'auto'
    coe = 0.0
    deg = 3.0
    svr_rbf = SVR(kernel=kel, C=c, epsilon=eps, gamma=ga, coef0=coe, degree=deg)

    train_svr = svr_rbf.fit(X_train, y_train)
    train_pred_nor = train_svr.predict(X_train)
    val_pred_nor = train_svr.predict(X_val)
    test_pred_nor = train_svr.predict(X_test)

if training_method == 'RF':
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor

    # for illustration purposes only, don't use this code!
    param_grid = {'n_estimators': [10, 50, 100, 150, 200, 400, 600],
                  'min_samples_split': [2, 4, 6, 8, 10],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_leaf': [1, 3, 5],
                  'bootstrap': [True],
                  'max_samples': [50, 100, 200, 500, 1000]}

    """grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
    grid.fit(X_train0, y_train0)
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    par_opt = grid.best_params_
    print("Best parameters: ", grid.best_params_)
    print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))"""

    """Best cross-validation accuracy: 0.62
    'bootstrap': True, 'max_features': 'auto', 'max_samples': 1000, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 600
    Test set accuracy: 0.58"""

    rf_estimator = RandomForestClassifier(bootstrap=True, max_samples=3000, max_features='auto', min_samples_split=2, n_estimators=200, verbose=0, random_state=42, n_jobs=6)
    train_est = rf_estimator.fit(X_train, y_train)
    train_pred = train_est.predict(X_train)
    val_pred = train_est.predict(X_val)
    test_pred = train_est.predict(X_test)
    acc_train = train_est.score(X_train, y_train)
    acc_val = train_est.score(X_val, y_val)
    acc_test = train_est.score(X_test, y_test)
    print(train_pred.shape, test_pred.shape)
    cfs_train = confusion_matrix(y_train, train_pred, labels=[0, 1, 2, 3])
    cfs_val = confusion_matrix(y_val, val_pred, labels=[0, 1, 2, 3])
    cfs_test = confusion_matrix(y_test, test_pred, labels=[0, 1, 2, 3])

    print(acc_train, acc_val, acc_test)
    print(cfs_train, '\n', cfs_val, '\n', cfs_test)

    filename = './final models/' + training_method + '_finalized_model.sav'
    model = train_est
    pickle.dump(model, open(filename, 'wb'))

    # Calculate the prediction for training data


"""plt.figure(1)
plt.scatter(train_real, train_pred, color='black')
plt.xlabel("BHS")
plt. ylabel("BHS (Predicted)")
plt.xlim(-2, 20)
plt.ylim(-2, 20)

plt.figure(2)
plt.scatter(val_real, val_pred, color='black')
plt.xlabel("BHS")
plt. ylabel("BHS (Predicted)")
plt.xlim(-2, 20)
plt.ylim(-2, 20)

plt.figure(3)
plt.scatter(test_real, test_pred, color='black')
plt.xlabel("BHS")
plt. ylabel("BHS (Predicted)")
plt.xlim(-2, 20)
plt.ylim(-2, 20)

plt.show()"""



