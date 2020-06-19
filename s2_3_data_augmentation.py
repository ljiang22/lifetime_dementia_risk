import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from mllib.preprocess import normalize
from mllib.preprocess import unnormalize
from mllib.preprocess import unnormalize_v1
from mllib.networks import evaluate
from mllib.networks import MLP_train_opt
from mllib.networks import MLP_plot
from mllib.networks import MLP_REG_v1
from mllib.networks import MLPR_v0
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import sklearn
from sklearn.metrics import SCORERS
from sklearn import preprocessing
import pickle


set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
#set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

input_file = './raw data_edit/data_fd.csv'    # The well name of an input file
data_input_ori = pd.read_csv(input_file)
data_input_ori0 = data_input_ori
print(data_input_ori.head())

#training_method ='SVR'
#training_method ='MLP'
training_method ='RF'
opt_flag = 0
NT = 2
cdr = data_input_ori['sumbox']
label_min = 0
label_max = 18.0
data_input_ori = data_input_ori.drop(columns=['Subject'])
data_input_ori0 = data_input_ori0.drop(columns=['Subject'])
keys = data_input_ori.keys()
x = data_input_ori.values #returns a numpy array
#print(x.shape, x[0:50, :])
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
x_scaled = np.transpose(normalize(x))
# Normalization
data_mean = np.mean(cdr)
data_std = np.std(cdr)
#print(x_scaled.shape)
data_input_ori = pd.DataFrame(x_scaled)
data_input_ori.columns = keys
print(data_input_ori.shape, data_input_ori.keys())

keys = data_input_ori.keys()
print(keys)
M, N = data_input_ori.shape
print(M, N)

#print(data_input_ori.head())
#print(data_input_ori.describe())

# Data augmentation

data_cdr15 = data_input_ori.loc[0:1]
data_cdr8 = data_input_ori.loc[0:1]
data_cdr4 = data_input_ori.loc[0:1]
data_cdr1 = data_input_ori.loc[0:1]  # with cdr>0.5
data_cdr0 = data_input_ori.loc[0:1]
for i in range(M):
    if cdr[i] >= 15.0:
        data_cdr15 = data_cdr15.append(data_input_ori.loc[i], ignore_index=True)
        #print(i, data_cdr15.shape)
    elif cdr[i] >= 8 and cdr[i] < 15:
        data_cdr8 = data_cdr8.append(data_input_ori.loc[i], ignore_index=True)
    elif cdr[i] >= 4 and cdr[i] < 8:
        data_cdr4 = data_cdr4.append(data_input_ori.loc[i], ignore_index=True)
    elif cdr[i] >= 0.5 and cdr[i] < 4:
        data_cdr1 = data_cdr1.append(data_input_ori.loc[i], ignore_index=True)
    else:
        data_cdr0 = data_cdr0.append(data_input_ori.loc[i], ignore_index=True)
        #print(i, data_cdr0.shape)
        #print(i, label[i])

print(data_cdr15.shape, data_cdr8.shape, data_cdr4.shape, data_cdr1.shape, data_cdr0.shape)


len0 = len(data_cdr0)
idx= np.arange(len0)
idx = idx[1:]
print(idx)
idx = sklearn.utils.shuffle(idx, random_state=27)
print(idx)
idx_test = idx[0:300]  # 15%
idx_val = idx[300:640] #20% of training set
idx_train = idx[640:2000] # Only use 2000 data points, because there are too many data less than 0.5
data_cdr0_train = data_cdr0.loc[idx_train]
data_cdr0_val = data_cdr0.loc[idx_val]
data_cdr0_test = data_cdr0.loc[idx_test]
print(data_cdr0_train.shape, data_cdr0_val.shape, data_cdr0_test.shape)

len1 = len(data_cdr1)
idx1= np.arange(len1)
idx1 = idx1[1:]
#print(idx1)
idx = sklearn.utils.shuffle(idx1, random_state=27)
#print(idx1)
idx_test = idx1[0:128]  # 15%
idx_val = idx1[128:274] #20% of training set
idx_train = idx1[274:] # Only use 2000 data points, because there are too many data less than 0.5
data_cdr1_train = data_cdr1.loc[idx_train]
data_cdr1_val = data_cdr1.loc[idx_val]
data_cdr1_test = data_cdr1.loc[idx_test]
print(data_cdr1_train.shape, data_cdr1_val.shape, data_cdr1_test.shape)

len4 = len(data_cdr4)
idx1= np.arange(len4)
idx1 = idx1[1:]
#print(idx1)
idx = sklearn.utils.shuffle(idx1, random_state=27)
#print(idx1)
idx_test = idx1[0:47]  # 15%
idx_val = idx1[47:100] #20% of training set
idx_train = idx1[100:] # Only use 2000 data points, because there are too many data less than 0.5
data_cdr4_train = data_cdr4.loc[idx_train]
data_cdr4_val = data_cdr4.loc[idx_val]
data_cdr4_test = data_cdr4.loc[idx_test]
print(data_cdr4_train.shape, data_cdr4_val.shape, data_cdr4_test.shape)

len8 = len(data_cdr8)
idx1= np.arange(len8)
idx1 = idx1[1:]
#print(idx1)
idx = sklearn.utils.shuffle(idx1, random_state=27)
#print(idx1)
idx_test = idx1[0:17]  # 15%
idx_val = idx1[17:37] #20% of training set
idx_train = idx1[37:] # Only use 2000 data points, because there are too many data less than 0.5
data_cdr8_train = data_cdr8.loc[idx_train]
data_cdr8_val = data_cdr8.loc[idx_val]
data_cdr8_test = data_cdr8.loc[idx_test]
print(data_cdr8_train.shape, data_cdr8_val.shape, data_cdr8_test.shape)

len1 = len(data_cdr15)
idx1= np.arange(len1)
idx1 = idx1[1:]  # because the first sample does belong to this category.
#print(idx1)
idx = sklearn.utils.shuffle(idx1, random_state=27)
#print(idx1)
idx_test = idx1[0:2]  # 15%
idx_val = idx1[2:4] #20% of training set
idx_train = idx1[4:] # Only use 2000 data points, because there are too many data less than 0.5
data_cdr15_train = data_cdr15.loc[idx_train]
data_cdr15_val = data_cdr15.loc[idx_val]
data_cdr15_test = data_cdr15.loc[idx_test]
print(data_cdr15_train.shape, data_cdr15_val.shape, data_cdr15_test.shape)

data_val = pd.concat([data_cdr0_val, data_cdr1_val, data_cdr4_val, data_cdr8_val, data_cdr15_val], axis=0)
data_test = pd.concat([data_cdr0_test, data_cdr1_test, data_cdr4_test, data_cdr8_test, data_cdr15_test], axis=0)
print(data_val.shape, data_test.shape)


# Training data augment
data_cdr15_0 = data_cdr15_train
data_cdr15 = data_cdr15_train
for j in range(3):
    data_cdr15 = data_cdr15.append(data_cdr15_0)

print(data_cdr15.shape)
data_cdr8_0 = data_cdr8_train
data_cdr8 = data_cdr8_train
for j in range(2):
    data_cdr8 = data_cdr8.append(data_cdr8_0)

print(data_cdr8.shape)

data_cdr4_0 = data_cdr4_train
data_cdr4 = data_cdr4_train
for j in range(1):
    data_cdr4 = data_cdr4.append(data_cdr4_0)

print(data_cdr4.shape)

data_cdr1_0 = data_cdr1_train
data_cdr1 = data_cdr1_train
for j in range(1):
    data_cdr1 = data_cdr1.append(data_cdr1_0)

print(data_cdr1.shape)


data_aug0 = pd.concat([data_cdr0_train, data_cdr1, data_cdr4, data_cdr8, data_cdr15], axis=0)
#data_aug0 = pd.concat([data_cdr0_train, data_cdr1_0, data_cdr4_0, data_cdr8_0, data_cdr15_0], axis=0)
data_aug0.reset_index(drop=True, inplace=True) # Reindex the dataset in a continuous number
M1, N1 = data_aug0.shape
print(data_aug0.shape)
idx = np.arange(M1)
print(idx)
idx1 = sklearn.utils.shuffle(idx, random_state=27)
print(idx1)
print(len(idx1))
data_aug = data_aug0.loc[idx1]
data_aug.reset_index(drop=True, inplace=True) # Reindex the dataset in a continuous number

"""print(data_aug.shape, data_aug.keys())
M, N = data_aug.shape
x1 = len(data_cdr0_train) / M
x2 = len(data_cdr1) / M
x3 = len(data_cdr4) / M
x4 = len(data_cdr8) / M
x5 = len(data_cdr15) / M
print(x1)
data_pie = [x1, x2, x3, x4, x5]
explode = (0.1, 0.0, 0, 0, 0.0)
labels = ['Normal', 'Mild impairment', '', '', 'Dementia']
f, bx = plt.subplots(nrows=1, ncols=1)
#autopct='%1.1f%%', labels=labels,
bx.pie(data_pie, explode=explode, shadow=True, autopct='%1.1f%%', startangle=90)

plt.show()"""




# Feature selection
var = 0.0   # Remove features with low variance. 0.8 means remove all features that are either one or zero (on or off) in more than 80% of the samples
sel = VarianceThreshold(threshold=var)
data_tmp = sel.fit_transform(data_input_ori0)
indices = sel.get_support(indices=True)
print(indices)
keys1 = keys[indices]
print(keys1)
keys_list = list(keys1)
print(keys_list)
data_val = data_val[keys_list]
data_test = data_test[keys_list]
data_edit = data_aug[keys_list]
keys1 = keys1[2:]

#  'CVPACE', 'CVCHF','CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMCHR', 'NCOTHR', 'INCONTF', 'TOBAC30', 'ABUSOTHR', 'PSYCDIS'

print(keys1)

print(data_aug.shape, data_edit.shape, data_val.shape, data_test.shape, type(data_edit))


X_val = data_val.values[:, 2:]
y_val = data_val.values[:, 1]
X_test = data_test.values[:, 2:]
y_test = data_test.values[:, 1]

#print(data_edit[0:50])

data_input = data_edit
print(data_input.shape, type(data_input))
#print(max(data_input[:, 1]))

# Normalization
#data_mean = np.mean(data_input[:, 1])
#data_std = np.std(data_input[:, 1])
#print(data_mean, data_std)

#data_nor = np.transpose(normalize(data_input))
data_nor = data_input.values
print('data_nor shape is', data_nor.shape)

features = data_nor[:, 2:]
label_norm = data_nor[:, 1]
print(features.shape, label_norm.shape)

sltk = SelectKBest(f_regression, k=45)
fv, pv = f_regression(features, label_norm)
features_new = sltk.fit_transform(features, label_norm)
indices2 = sltk.get_support(indices=True)
keys2 = keys1[indices2]
print(keys2)
# 'Race', Atrial fibrillation (CVAFIB), 'DIABETES', Thyroid disease (THYROID), 100 lifetime cigarettes (TOBAC100), 'PACKSPER'
# 'CVHATT', 'TOBAC100', 'THYROID', 'dem_idx', 'CVANGIO', 'CVOTHR'
# 'Hand', 'DIABETES', 'THYROID', 'TOBAC100', 'Race'

print(features_new.shape)
indices3 = np.argsort(pv)
print(keys1[indices3])
indices3 = indices3[0:29]
#pv = sorted(pv)
#print(len(pv), pv)
keys3 = keys1[indices3]
print('keys3 is', keys3)
features_new = features[:, indices3]
X_val = X_val[:, indices3]
X_test = X_test[:, indices3]
x_axis = np.linspace(1, len(pv), len(pv))
features_new = features_new[:, 0:]
pv = sorted(pv)

font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)
plt.figure(1)
plt.scatter(x_axis, pv, color='black')
plt.xlabel("Feature")
plt. ylabel("P-value")
#plt.xlim(-2, 20)
#plt.ylim(-2, 20)
plt.show()

# Recursive feature elimination:
"""rf_estimator = RandomForestRegressor(bootstrap=True, max_samples=3000, max_features='auto', min_samples_split=2,
                                     n_estimators=40, verbose=0, random_state=42, n_jobs=6)

# The "accuracy" scoring is proportional to r2score for prediction
# print(sorted(sklearn.metrics.SCORERS.keys()))

rfecv = RFECV(estimator=rf_estimator, step=1, cv=5, scoring='r2')
rfecv.fit(features_new, label_norm)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()"""

# L1-based feature selection
from sklearn.feature_selection import SelectFromModel
rf_estimator = RandomForestRegressor(bootstrap=True, max_samples=2000, max_features='auto', min_samples_split=2,
                                     n_estimators=40, verbose=0, random_state=42, n_jobs=6)
rf_fit = rf_estimator.fit(features_new, label_norm)
feature_imp = rf_fit.feature_importances_
#print(feature_imp)
print(sorted(feature_imp))
indices4 = np.argsort(feature_imp)
keys4 = keys3[indices4]
print('keys4 is', keys4)
#features_new = features[:, indices3]
model = SelectFromModel(rf_fit, threshold=0.0000001, prefit=True, max_features=46)
features_new = model.transform(features_new)
print(features_new.shape)

# Plot number of features VS. cross-validation scores
x_axis = np.linspace(1, len(feature_imp), len(feature_imp))
plt.figure()
plt.scatter(x_axis, feature_imp)
plt.xlabel("Features")
plt.ylabel("Feature importance")
plt.show()


#print(data_nor[0:10, :])

from sklearn.model_selection import train_test_split
#X_train0, X_test, y_train0, y_test = train_test_split(features_new, label_norm, test_size=0.15, random_state=22)
#X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.10, random_state=22)
X_train = features_new
y_train = label_norm


print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

train_real = unnormalize(y_train, data_mean, data_std)
val_real = unnormalize(y_val, data_mean, data_std)
test_real = unnormalize(y_test, data_mean, data_std)

data_min = 0.0
data_max = 18.0
"""train_real = unnormalize_v1(y_train, data_max, data_min)
val_real = unnormalize_v1(y_val, data_max, data_min)
test_real = unnormalize_v1(y_test, data_max, data_min)"""

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
    model_mlp = MLPR_v0(data_mean, data_std, h1, h2, h3, batch_size=batch_size, max_epoch=epoch, lr0=lr, act_func=act_func)
    model_mlp.fit(X_train, y_train, X_val, y_val)
    train_pred = model_mlp.predict(X_train)
    val_pred = model_mlp.predict(X_val)
    test_pred = model_mlp.predict(X_test)
    #print('y_val is', val_pred.shape, y_val.shape)

    # Calculate the prediction for training data


    test_score = model_mlp.score(X_test, test_real)
    print('Test set score is', test_score)
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

    # Calculate the prediction for training data
    train_pred = unnormalize(train_pred_nor, data_mean, data_std)
    val_pred = unnormalize(val_pred_nor, data_mean, data_std)
    test_pred = unnormalize(test_pred_nor, data_mean, data_std)

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

    rf_estimator = RandomForestRegressor(bootstrap=True, max_samples=2000, max_features='auto', min_samples_split=2, n_estimators=100, verbose=0, random_state=42, n_jobs=6)
    train_est = rf_estimator.fit(X_train, y_train)
    train_pred_nor = train_est.predict(X_train)
    val_pred_nor = train_est.predict(X_val)
    test_pred_nor = train_est.predict(X_test)

    filename = './final models/' + training_method + '_finalized_model.sav'
    model = train_est
    pickle.dump(model, open(filename, 'wb'))

    # Calculate the prediction for training data
    train_pred = unnormalize(train_pred_nor, data_mean, data_std)
    val_pred = unnormalize(val_pred_nor, data_mean, data_std)
    test_pred = unnormalize(test_pred_nor, data_mean, data_std)

    """train_pred = unnormalize_v1(train_pred_nor, data_max, data_min)
    val_pred = unnormalize_v1(val_pred_nor, data_max, data_min)
    test_pred = unnormalize_v1(test_pred_nor, data_max, data_min)"""


acc_pred_train, r2_train, err_train, corr_train = evaluate(train_real, train_pred)
print(training_method + ' Train accuracy = {:0.2f}%'.format(acc_pred_train))
print(training_method + ' Train R2 score = {:0.2f}'.format(r2_train))
print(training_method + ' Train avg. error = {:0.3f}'.format(err_train))
print(training_method + ' Train correlation = {:0.2f}'.format(corr_train))

acc_pred_train, r2_train, err_train, corr_train = evaluate(val_real, val_pred)
print(training_method + ' Validation accuracy = {:0.2f}%'.format(acc_pred_train))
print(training_method + ' Validation R2 score = {:0.2f}'.format(r2_train))
print(training_method + ' Validation avg. error = {:0.3f}'.format(err_train))
print(training_method + ' Validation correlation = {:0.2f}'.format(corr_train))

acc_pred_train, r2_train, err_train, corr_train = evaluate(test_real, test_pred)
print(training_method + ' Test accuracy = {:0.2f}%'.format(acc_pred_train))
print(training_method + ' Test R2 score = {:0.2f}'.format(r2_train))
print(training_method + ' Test avg. error = {:0.3f}'.format(err_train))
print(training_method + ' Test correlation = {:0.2f}'.format(corr_train))

plt.figure(1)
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

plt.show()



