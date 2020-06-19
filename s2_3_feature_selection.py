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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import sklearn
from sklearn.metrics import SCORERS
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
keys0 = []  # The features with low variance
for i in range(len(keys)):
    if i not in indices:
        keys0.append(keys[i])
print('The features removed due to low variance', keys0)

print(indices)
keys1 = keys[indices]
keys1 = keys1[2:]
print(keys1)

print(data_input_ori.shape, data_edit.shape, type(data_edit))

#print(data_edit[0:50])

data_input = data_edit
print(data_input.shape, type(data_input))
#print(max(data_input[:, 1]))

# Normalization
data_mean = np.mean(data_input[:, 1])
data_std = np.std(data_input[:, 1])
#print(data_mean, data_std)

data_nor = np.transpose(normalize(data_input))
print('data_nor shape is', data_nor.shape)

features = data_nor[:, 2:]
label_norm = data_nor[:, 1]
print(features.shape, label_norm.shape)

N1 = 45
sltk = SelectKBest(f_regression, k=N1)  #
fv, pv = f_regression(features, label_norm)
features_new = sltk.fit_transform(features, label_norm)
indices2 = sltk.get_support(indices=True)
keys2 = keys1[indices2]
print(keys2)
# 'Race', Atrial fibrillation (CVAFIB), 'DIABETES', Thyroid disease (THYROID), 100 lifetime cigarettes (TOBAC100), 'PACKSPER'

print(features_new.shape)
indices3 = np.argsort(pv)
print('keys3 without removing feature', keys1[indices3])
indices3 = indices3[0:N1]
#pv = sorted(pv)
print(len(pv), pv)
print(indices3)
keys3 = keys1[indices3]
print('keys3 is', keys3)
features_new = features[:, indices3]
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
N2 = N1
from sklearn.feature_selection import SelectFromModel
rf_estimator = RandomForestRegressor(bootstrap=True, max_samples=3000, max_features='auto', min_samples_split=2,
                                     n_estimators=100, verbose=0, random_state=42, n_jobs=6)
rf_fit = rf_estimator.fit(features_new, label_norm)
feature_imp = rf_fit.feature_importances_
print(sorted(feature_imp))
indices4 = np.argsort(feature_imp)
keys4 = keys3[indices4]
print('keys4 is', keys4)

sdasds
#features_new = features[:, indices3]
model = SelectFromModel(rf_fit, threshold=0.0000001, prefit=True, max_features=N2)
features_new = model.transform(features_new)
print(features_new.shape)

#print(data_nor[0:10, :])

from sklearn.model_selection import train_test_split
X_train0, X_test, y_train0, y_test = train_test_split(features_new, label_norm, test_size=0.15, random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.10, random_state=22)
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
train_real = unnormalize(y_train, data_mean, data_std)
val_real = unnormalize(y_val, data_mean, data_std)
test_real = unnormalize(y_test, data_mean, data_std)


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

    rf_estimator = RandomForestRegressor(bootstrap=True, max_samples=3000, max_features='auto', min_samples_split=2, n_estimators=600, verbose=0, random_state=42, n_jobs=6)
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
plt.xlabel("CDR")
plt. ylabel("CDR (Predicted)")
plt.xlim(-2, 20)
plt.ylim(-2, 20)

plt.figure(2)
plt.scatter(val_real, val_pred, color='black')
plt.xlabel("CDR")
plt. ylabel("CDR (Predicted)")
plt.xlim(-2, 20)
plt.ylim(-2, 20)

plt.figure(3)
plt.scatter(test_real, test_pred, color='black')
plt.xlabel("CDR")
plt. ylabel("CDR (Predicted)")
plt.xlim(-2, 20)
plt.ylim(-2, 20)

plt.show()



