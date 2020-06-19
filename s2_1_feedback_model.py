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


set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
#set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

input_file = './raw data_edit/data_fd.csv'    # The well name of an input file
data_input_ori = pd.read_csv(input_file)
print(data_input_ori.head())

training_method ='SVR'
#training_method ='MLP'
opt_flag = 0
NT = 2

keys = data_input_ori.keys()
print(keys)


data_input = data_input_ori.values
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
#print(data_nor[0:10, :])

from sklearn.model_selection import train_test_split
X_train0, X_test, y_train0, y_test = train_test_split(features, label_norm, test_size=0.15, random_state=27)
X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.10, random_state=27)
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
    #svr_rbf = SVR(kernel=kel, C=c, epsilon=eps, gamma=ga, coef0=coe, degree=deg)
    svr_rbf = SVR(C=1.0)
    train_svr = svr_rbf.fit(X_train, y_train)
    train_pred_nor = train_svr.predict(X_train)
    val_pred_nor = train_svr.predict(X_val)
    test_pred_nor = train_svr.predict(X_test)

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

font = {'family': 'normal', 'size': 18}
plt.rc('font', **font)

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



