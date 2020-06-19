import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from mllib.preprocess import unnormalize_v1
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from mllib.preprocess import unnormalize
import time


def ctc_next_batch(training, label, train101_imag_num_examples, batch_size):
    """Return the next `batch_size` examples from this data set."""
      # Shuffle the data
    perm = np.arange(train101_imag_num_examples)
    np.random.shuffle(perm)
    training = training[perm]
    label = label[perm]
      # Start next epoch
    start = 0
    training_index_in_epoch = batch_size
    assert batch_size <= train101_imag_num_examples
    end = training_index_in_epoch
    #print('training shape is', training[start:end].shape, label.shape)
    return training[start:end], label[start:end]

# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01) #tf.constant_initializer
    #initer = tf.constant_initializer(0.001)
    #initer = tf.random_normal_initializer
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    #initial = tf.constant_initializer(0.0)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)


def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        print('x shape is', in_dim, x.shape)
        W = weight_variable(name, shape=[in_dim, num_units])
        b = bias_variable(name, [num_units])
        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            #layer = tf.nn.leaky_relu(layer, alpha=0.2)
            layer = tf.nn.relu(layer)

        #print(W.shape, b.shape, layer.shape)

        return layer, W, b

def fc_layer_edit(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        #print('x shape is', x.shape)  # ?x20x1
        Ns = x.shape[2]
        Nt = x.shape[1]
        W = weight_variable(name, shape=[Ns, num_units])
        b = bias_variable(name, [num_units])
        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)
        y_pred = []
        for nt in range(Nt):
            x_tmp = x[:, nt, :]
            #print(x_tmp.shape)
            #x_tmp = tf.reshape(x_tmp, (x_tmp.shape[0], 1))
            layer = tf.matmul(x_tmp, W)
            #print(layer.shape)
            layer += b
            #print('The layer shape is', layer.shape)
            if use_relu:
                layer = tf.nn.leaky_relu(layer, alpha=0.2)
            y_pred.append(layer)

        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
        #print('y shape is', y_pred.shape)  # ?x20x1
        return y_pred, W, b

def fc_layer_edit0(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        #print(x.shape)
        Ns = x.shape[2]
        Nt = x.shape[1]
        W = weight_variable(name, shape=[Ns, num_units])
        b = bias_variable(name, [num_units])
        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)
        y_pred = []
        zp0 = 1.56
        for nt in range(Nt):
            x_tmp = x[:, nt, :]
            layer = tf.matmul(x_tmp, W)
            layer += b

            if nt == 0:
                #out_tmp0 = layer * zp0
                out_tmp0 = layer
            else:
                #out_tmp0 = layer * y_pred[nt-1]
                out_tmp0 = layer

            if use_relu:
                out_tmp0 = tf.nn.leaky_relu(out_tmp0, alpha=0.2)
            y_pred.append(out_tmp0)

        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
        #print('y shape is', y_pred.shape)  # ?x20x1
        return y_pred, W, b


def fc_layer_lj(x, wt, b, use_relu=True):
    # wt = np.float32(wt)
    #print(x.shape, wt.shape)
    #x = np.float32(x)
    #x = np.transpose(x)
    wt = np.transpose(wt)
    #print(x.shape, wt.shape)
    #wt = wt.astype(float)
    #x = x.astype(float)
    layer = np.matmul(wt, x)
    layer += b
    layer = np.float32(layer)
    if use_relu:
        layer = tf.nn.leaky_relu(layer, alpha=0.2)
        sess = tf.Session()
        layer = sess.run(layer)
        #print(layer.shape, layer)
        sess.close()

    return layer


def encoded_label(y_input, nit, n_class):
    y_label = np.zeros((nit, n_class), dtype=float)

    for i in range(nit):
        label = np.zeros(n_class, dtype=float)
        label[int(y_input[i])-1] = 1.0
        y_label[i, :] = label
    return y_label

# for regression
def encoded_label_r(y_input, nit, n_class):
    y_label = np.zeros((nit, n_class), dtype=float)

    for i in range(nit):
        y_label[i, :] = y_input[i]
    return y_label

def soft_max(output_logit):
    N_lt, M_lt = np.shape(output_logit)
    e_out = np.exp(output_logit)
    sum_out = np.sum(e_out, axis=1)
    out_softmax = np.zeros((N_lt, M_lt), dtype=float)
    for i in range(N_lt):
        for k in range(M_lt):
            out_softmax[i, k] = e_out[i,k]/sum_out[i]
    return out_softmax

def evaluate(y_real, y_predict):
    #print(y_real.shape, y_predict.shape)
    errors = abs(y_predict - y_real)
    mse = np.mean(np.square(errors))
    err_avg = np.mean(errors)
    error0 = abs(np.divide(errors, y_real))
    error0 = error0[~np.isnan(error0)]
    error0 = error0[~np.isinf(error0)]
    index = np.where(error0 <= 1.0)
    error0 = error0[index]
    #print(error0)
    errorp = 100 * np.mean(error0)
    accuracy = 100 - errorp
    #print(accuracy)
    #print('Model Performance')
    r2 = r2_score(y_real, y_predict)
    corr = np.corrcoef(y_real, y_predict)
    #print(corr)
    #print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy, r2, err_avg, corr[0, 1]

def evaluate_v1(y_real, y_predict):
    #print(y_real.shape, y_predict.shape)
    errors = abs(y_predict - y_real)
    mse = np.mean(np.square(errors))
    err_avg = np.mean(errors)
    """error0 = abs(np.divide(errors, y_real))
    error0 = error0[~np.isnan(error0)]
    error0 = error0[~np.isinf(error0)]
    index = np.where(error0 <= 1.0)
    error0 = error0[index]
    #print(error0)
    errorp = 100 * np.mean(error0)
    accuracy = 100 - errorp"""
    #print(accuracy)
    #print('Model Performance')
    r2 = r2_score(y_real, y_predict)
    corr = np.corrcoef(y_real, y_predict)
    #print(corr)
    #print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #print('Accuracy = {:0.2f}%.'.format(accuracy))
    return r2, err_avg, corr[0, 1]

# Make sure the seismic attribute has the same size as porosity. The assumption is that they have the same start time.
def match_process(MSFT_tmp, Por_tmp):
    Ntr = len(MSFT_tmp)
    MSFT_edit = []
    for nr in range(Ntr):
        nt = len(Por_tmp[nr])
        ns = len(MSFT_tmp[nr])
        MSFT = np.asarray(MSFT_tmp[nr])
        if ns > nt:
            #print(MSFT.shape)
            MSFT = MSFT[0:nt, :, :]
        MSFT_edit.append(MSFT)
    return MSFT_edit

# Remove wells with bad quality. Make sure it is in correct order
# wells = ['19A', 'BT2', 'SR', 'F-11A','F-11T2', 'F-12', 'F-1', 'F-1A']
def data_select(MSFT_tmp, Por_tmp):
    Ntr = len(MSFT_tmp)
    MSFT_edit = []
    por_edit = []
    for nr in range(Ntr):
        if nr < 6 :
           MSFT = np.asarray(MSFT_tmp[nr])
           por = np.asarray(Por_tmp[nr])
           MSFT_edit.append(MSFT)
           por_edit.append(por)
    return MSFT_edit, por_edit

from sklearn.svm import SVR
def SVR_pred(X_train, y_train, X_blind, X_train_well, c=1000.0, gamma=0.1):
    kel = 'rbf'
    c=1.0
    eps = 0.3
    ga = 'auto'
    coe = 0.0
    deg = 3.0
    svr_rbf = SVR(kernel=kel, C=c, epsilon=eps, gamma=ga, coef0=coe, degree=deg)
    train_pred = svr_rbf.fit(X_train, y_train).predict(X_train_well)
    blind_pred = svr_rbf.fit(X_train, y_train).predict(X_blind)
    return train_pred, blind_pred

from sklearn.ensemble import RandomForestRegressor
def RF_pred(X_train, y_train, X_blind, X_train_well):
    estimator = RandomForestRegressor(bootstrap=True, max_features='auto', min_samples_leaf=4,
                                  min_samples_split=10, n_estimators=400, verbose=2, random_state=42, n_jobs=6)
    estimator.fit(X_train, y_train)
    train_pred = estimator.predict(X_train_well)
    blind_pred = estimator.predict(X_blind)
    return train_pred, blind_pred

def MLP_REG_v1(X_train, y_train, X_blind, y_blind, X_train_well, y_train_well, h1, h2, h3, lr, epochs, batch_size, display_freq=50):
    #start0 = time.time()

    #print(X_train.shape, X_blind.shape, y_train.shape, y_blind.shape, X_train_well.shape, y_train_well.shape)

    tf.reset_default_graph()  #reset the whole graph

    ncls = 1  # number of classes, used for regression, not classification

    Ntrn, Mtrn = X_train.shape
    Ntt, Mtt = X_train_well.shape
    Nbd, Mbd = X_blind.shape

    # create one-hot shot encoded label
    y_train_label = encoded_label_r(y_train, Ntrn, ncls)
    y_train_well_label = encoded_label_r(y_train_well, Ntt, ncls)
    y_blind_label = encoded_label_r(y_blind, Nbd, ncls)

    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, Mtrn], name='X')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

    fc1, Wt1, b1 = fc_layer(x, h1, 'Hidden_layer', use_relu=True)
    fc2, Wt2, b2 = fc_layer(fc1, h2, 'Hidden_layer1', use_relu=True)
    output_logits, Wt3, b3 = fc_layer(fc2, h3, 'Output_layer', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.squared_difference(output_logits, y))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    # print(Ntrn, batch_size)
    num_tr_iter = int(Ntrn / batch_size)

    with tf.Session() as sess:
        sess.run(init)
        global_step = 0

        for epoch in range(int(epochs)):
            """if epoch % 5 == 0:
                print('Training epoch: {}'.format(epoch))
                print("num_tr_iter", num_tr_iter)"""
            for iteration in range(num_tr_iter):
                # print(Ntrn, batch_size)
                batch_x, batch_y = ctc_next_batch(X_train, y_train_label, Ntrn, int(batch_size))
                global_step += 1

                # Run optimization op (backprop)
                feed_dict_batch = {x: batch_x, y: batch_y}
                _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)


                loss_batch = sess.run(loss, feed_dict=feed_dict_batch)

                #if iteration % display_freq == 0:
                    #print("iter= {0:3d}:\t Loss={1:.6f}".format(iteration, loss_batch))

        # Run validation after every epoch
        #feed_dict_train = {x: X_train, y: y_train}
        #loss_train = sess.run(loss, feed_dict=feed_dict_train)
        # print('---------------------------------------------------------')
        #print("Total train validation loss: {1:.6f}".format(loss_train))
        # print('---------------------------------------------------------')

        # Test the network after training
        feed_dict_train = {x: X_train_well, y: y_train_well_label}
        loss_test = sess.run(loss, feed_dict=feed_dict_train)
        #print("Train loss for train well: {0:.6f}".format(loss_test))

        # test the network after training with blind well data
        feed_dict_blind = {x: X_blind, y: y_blind_label}
        loss_blind = sess.run(loss, feed_dict=feed_dict_blind)
        #print("Blind loss: {0:.6f}".format(loss_blind))

        results = sess.run(output_logits, feed_dict=feed_dict_blind)  # results for blind well
        res_training = sess.run(output_logits, feed_dict=feed_dict_train)  # results for blind well

        # output the model
        wt1 = sess.run(Wt1)
        b1 = sess.run(b1)

        wt2 = sess.run(Wt2)
        b2 = sess.run(b2)

        wt3 = sess.run(Wt3)
        b3 = sess.run(b3)

    #end1 = (time.time() - start0)
    #print('the time for each iteration (mins):', end1 / 60)

    sess.close()

    return res_training, results, wt1, b1, wt2, b2, wt3, b3


def MLP_plot(X_train, y_train, X_val, y_val, X_test, y_test, h1, h2, h3, lr, epochs, batch_size, display_freq=50):
    #start0 = time.time()

    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    #print(y_blind)

    """f, bx = plt.subplots(nrows=1, ncols=1)
    bx.plot(y_train,  '-', label='Original', color='black')
    bx.legend(loc='upper right')
    bx.invert_yaxis()
    # bx.grid()
    bx.locator_params(axis='x', nbins=5)
    bx.set_xlabel("P-impedance(km/s*g/c$m^3$)")
    bx.set_ylabel("Time (ms)")
    #plt.show()

    f, bx = plt.subplots(nrows=1, ncols=1)
    bx.plot(y_blind,  '-', label='Original', color='black')
    bx.legend(loc='upper right')
    bx.invert_yaxis()
    # bx.grid()
    bx.locator_params(axis='x', nbins=5)
    bx.set_xlabel("P-impedance(km/s*g/c$m^3$)")
    bx.set_ylabel("Time (ms)")
    #plt.show()

    f, bx = plt.subplots(nrows=1, ncols=1)
    bx.plot(y_train_well,  '-', label='Original', color='black')
    bx.legend(loc='upper right')
    bx.invert_yaxis()
    # bx.grid()
    bx.locator_params(axis='x', nbins=5)
    bx.set_xlabel("P-impedance(km/s*g/c$m^3$)")
    bx.set_ylabel("Time (ms)")
    plt.show()"""

    tf.reset_default_graph()  #reset the whole graph

    ncls = 1  # number of classes, used for regression, not classification

    Ntrn, Mtrn = X_train.shape
    Ntc, Mtc = X_val.shape
    Nbd, Mbd = X_test.shape


    # create one-hot shot encoded label
    y_train_label = encoded_label_r(y_train, Ntrn, ncls)
    y_val_label = encoded_label_r(y_val, Ntc, ncls)
    #y_train_well_label = encoded_label_r(y_train_well, Ntt, ncls)
    y_test_label = encoded_label_r(y_test, Nbd, ncls)

    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, Mtrn], name='X')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

    fc1, Wt1, b1 = fc_layer(x, h1, 'Hidden_layer', use_relu=True)
    fc2, Wt2, b2 = fc_layer(fc1, h2, 'Hidden_layer1', use_relu=True)
    output_logits, Wt3, b3 = fc_layer(fc2, h3, 'Output_layer', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.squared_difference(output_logits, y))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    # print(Ntrn, batch_size)
    num_tr_iter = int(Ntrn / batch_size)
    loss_train_all = []
    loss_val_all = []
    loss_test_all = []

    with tf.Session() as sess:
        sess.run(init)
        global_step = 0

        for epoch in range(int(epochs)):
            if epoch % 5 == 0:
                print('Training epoch: {}'.format(epoch))
                print("num_tr_iter", num_tr_iter)
            for iteration in range(num_tr_iter):
                # print(Ntrn, batch_size)
                batch_x, batch_y = ctc_next_batch(X_train, y_train_label, Ntrn, int(batch_size))
                global_step += 1

                # Run optimization op (backprop)
                #print(batch_x.shape, batch_y.shape, type(batch_x))


                feed_dict_batch = {x: batch_x, y: batch_y}
                #y_pred =  sess.run(output_logits, feed_dict=feed_dict_batch)
                #print(batch_size, y_pred.shape)

                _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)


                loss_batch = sess.run(loss, feed_dict=feed_dict_batch)

            feed_dict_train = {x: X_train, y: y_train_label}
            loss_train_tmp = sess.run(loss, feed_dict=feed_dict_train)
            loss_train_all.append(loss_train_tmp)

            # feed_dict_test = {x: X_test, y: y_test_label}
            feed_dict_val = {x: X_val, y: y_val_label}
            loss_val_tmp = sess.run(loss, feed_dict=feed_dict_val)
            loss_val_all.append(loss_val_tmp)

            feed_dict_test = {x: X_test, y: y_test_label}
            loss_test_tmp = sess.run(loss, feed_dict=feed_dict_test)
            loss_test_all.append(loss_test_tmp)

        loss_min = np.min(loss_val_all)
        loss_min_idx = np.argmin(loss_val_all)
        print(loss_min, loss_min_idx)

        # Run validation after every epoch
        feed_dict_train = {x: X_train, y: y_train_label}
        loss_train = sess.run(loss, feed_dict=feed_dict_train)
        train_pred = sess.run(output_logits, feed_dict=feed_dict_train)

        # Test the network after training
        feed_val_test = {x: X_val, y: y_val_label}
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        test_pred = sess.run(output_logits, feed_dict=feed_dict_val)

        # Test the network after training with blind well data
        feed_dict_test = {x: X_test, y: y_test_label}
        loss_test = sess.run(loss, feed_dict=feed_dict_test)
        test_pred = sess.run(output_logits, feed_dict=feed_dict_test)

        results = [loss_train, loss_val, loss_test]

        #print(results)

        #print(results)
        ModelParMLP =[]
        #output the model
        wt1 = sess.run(Wt1)
        b1 = sess.run(b1)

        wt2 = sess.run(Wt2)
        b2 = sess.run(b2)

        wt3 = sess.run(Wt3)
        b3 = sess.run(b3)

        ModelParMLP.append(wt1)
        ModelParMLP.append(b1)
        ModelParMLP.append(wt2)
        ModelParMLP.append(b2)
        ModelParMLP.append(wt3)
        ModelParMLP.append(b3)

        font = {'family': 'normal', 'size': 18}
        plt.rc('font', **font)

        plt.figure(1)
        plt.plot(loss_train_all, c="black", label='Training')
        plt.plot(loss_val_all, c="g", label='Validation')
        plt.plot(loss_test_all, c="r", label='Test')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('MSE')

        Nmx = len(loss_train_all)
        x_axis = np.linspace(1, Nmx, num=Nmx)
        plt.figure(2)
        plt.scatter(x_axis, loss_train_all, c="black", label='Training')
        plt.scatter(x_axis, loss_val_all, c="g", label='Validation')
        plt.scatter(x_axis, loss_test_all, c="r", label='Test')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('MSE')

        plt.show()

    sess.close()

    return wt1, b1, wt2, b2, wt3, b3, loss_min_idx



def MLP_MPI(learning_rate, epochs, batch_size, h1, h2, idx, X_train, y_train,
            X_test, y_test, X_blind, y_blind, display_freq, data_mean, data_std):

    tf.reset_default_graph()

    ncls = 1  # regression
    Ntrn, Mtrn = X_train.shape
    Ntt, Mtt = X_test.shape
    Nbd, Mbd = X_blind.shape

    # create one-hot shot encoded label
    y_train_label = encoded_label_r(y_train, Ntrn, ncls)
    y_test_label = encoded_label_r(y_test, Ntt, ncls)
    y_blind_label = encoded_label_r(y_blind, Nbd, ncls)


    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, Mtrn], name='X')
        y = tf.placeholder(tf.float32, shape=[None, ncls], name='Y')

    fc1, Wt1, b1 = fc_layer(x, h1, 'Hidden_layer', use_relu=True)
    fc2, Wt2, b2 = fc_layer(fc1, h2, 'Hidden_layer1', use_relu=True)
    output_logits, Wt3, b3 = fc_layer(fc2, ncls, 'Output_layer', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.squared_difference(output_logits, y))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    #print(Ntrn, batch_size)
    num_tr_iter = int(Ntrn / batch_size)

    with tf.Session() as sess:
        sess.run(init)
        #train_writer = tf.summary.FileWriter(logdir="./logs/", graph=sess.graph)

        global_step = 0

        for epoch in range(int(epochs)):
            """if epoch % 2 == 0:
                print('Training epoch: {}'.format(epoch))
                print("num_tr_iter", num_tr_iter)"""
            for iteration in range(num_tr_iter):
                # print(Ntrn, batch_size)
                batch_x, batch_y = ctc_next_batch(X_train, y_train_label, Ntrn, int(batch_size))
                global_step += 1

                # Run optimization op (backprop)
                feed_dict_batch = {x: batch_x, y: batch_y}
                _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)


                loss_batch = sess.run(loss, feed_dict=feed_dict_batch)

                """if iteration % display_freq == 0:
                    print("iter= {0:3d}:\t Loss={1:.6f}".format(iteration, loss_batch))"""

        # Run validation after every epoch
        feed_dict_train = {x: X_train, y: y_train_label}
        loss_train = sess.run(loss, feed_dict=feed_dict_train)
        train_pred = sess.run(output_logits, feed_dict=feed_dict_train)


        train_real = unnormalize_v1(y_train, data_mean, data_std)
        train_pred = np.reshape(train_pred, len(train_pred))
        train_pred = unnormalize_v1(train_pred, data_mean, data_std)
        acc_pred_train, r2_train, err_train, corr_train = evaluate(train_real, train_pred)


        # Test the network after training
        feed_dict_test = {x: X_test, y: y_test_label}
        loss_test = sess.run(loss, feed_dict=feed_dict_test)
        test_pred = sess.run(output_logits, feed_dict=feed_dict_test)

        # recover to real value level from prediction
        test_real = unnormalize_v1(y_test, data_mean, data_std)
        test_pred = np.reshape(test_pred, len(test_pred))
        test_pred = unnormalize_v1(test_pred, data_mean, data_std)
        acc_pred_test, r2_test, err_test, corr_test = evaluate(test_real, test_pred)

        # test the network after training with blind well data
        feed_dict_blind = {x: X_blind, y: y_blind_label}
        loss_blind = sess.run(loss, feed_dict=feed_dict_blind)
        blind_pred = sess.run(output_logits, feed_dict=feed_dict_blind)

        # recover to real value level from prediction
        blind_real = unnormalize_v1(y_blind, data_mean, data_std)
        blind_pred = np.reshape(blind_pred, len(blind_pred))
        blind_pred = unnormalize_v1(blind_pred, data_mean, data_std)
        acc_pred_blind, r2_blind, err_blind, corr_blind = evaluate(blind_real, blind_pred)

        results = [idx, loss_train, loss_test, loss_blind, acc_pred_train, r2_train, err_train, corr_train,
                   acc_pred_test, r2_test, err_test, corr_test, acc_pred_blind, r2_blind, err_blind, corr_blind,
                   learning_rate, epochs, batch_size, h1, h2]
        #print(results)
        ModelParMLP =[]
        #output the model
        wt1 = sess.run(Wt1)
        b1 = sess.run(b1)

        wt2 = sess.run(Wt2)
        b2 = sess.run(b2)

        wt3 = sess.run(Wt3)
        b3 = sess.run(b3)

        ModelParMLP.append(wt1)
        ModelParMLP.append(b1)
        ModelParMLP.append(wt2)
        ModelParMLP.append(b2)
        ModelParMLP.append(wt3)
        ModelParMLP.append(b3)

    database_path = 'C:\Study2017\study and work2018_syn_amazon\Ph.D dissertation\phd-research-2020\data base\MLP model parameters'
    if idx < 10:
        FileName1 = database_path + '\ModelResult_000' + str(idx)
        FileName2 = database_path + '\ModelPar_000' + str(idx)
    elif idx < 100:
        FileName1 = database_path + '\ModelResult_00' + str(idx)
        FileName2 = database_path + '\ModelPar_00' + str(idx)
    elif idx < 1000:
        FileName1 = database_path + '\ModelResult_0' + str(idx)
        FileName2 = database_path + '\ModelPar_0' + str(idx)
    else:
        FileName1 = database_path + '\ModelResult_' + str(idx)
        FileName2 = database_path + '\ModelPar_' + str(idx)
        # print(FileName1)
    np.save(FileName1, results)
    np.save(FileName2, ModelParMLP)

    sess.close()


def MLP_train_opt(X_train, y_train, X_test, y_test, X_blind, y_blind, display_freq, data_mean, data_std,learning_rate,
                  epochs, batch_size, h1, h2):
    N_lr = len(learning_rate)
    N_eph = len(epochs)
    N_bz = len(batch_size)
    N_h1 = len(h1)
    N_h2 = len(h2)

    Ns = N_lr * N_eph * N_bz * N_h1 * N_h2
    PAR = np.zeros((Ns, 5))

    start0 = time.time()

    ni = 0
    for nlr in range(N_lr):
        for neph in range(N_eph):
            for nbz in range(N_bz):
                for nh1 in range(N_h1):
                    for nh2 in range(N_h2):
                        PAR[ni, 0] = learning_rate[nlr]
                        PAR[ni, 1] = epochs[neph]
                        PAR[ni, 2] = batch_size[nbz]
                        PAR[ni, 3] = h1[nh1]
                        PAR[ni, 4] = h2[nh2]
                        ni += 1
                        # print(ni)
                        # print(PAR[ni-1, :])
    nfg = int(Ns // 10)
    for ns in range(0, Ns):
        MLP_MPI(PAR[ns, 0], PAR[ns, 1], PAR[ns, 2], PAR[ns, 3], PAR[ns, 4], ns, X_train, y_train, X_test, y_test,
                X_blind, y_blind, display_freq, data_mean, data_std)

        if ns % nfg == 0.0:
            print('Finished', ns / Ns * 100, '%')

        if ns == 0:
            end1 = (time.time() - start0)
            print('The total time is (m):', end1 / 60 * Ns)

    Ntr = Ns
    ModelResult = []
    result_path = 'C:\Study2017\study and work2018_syn_amazon\Ph.D dissertation\phd-research-2020\data base\MLP model parameters'
    for ns in range(0, Ntr):
        if ns < 10:
            FileName = result_path + '\ModelResult_000' + str(ns) + ".npy"
        elif ns < 100:
            FileName = result_path + '\ModelResult_00' + str(ns) + ".npy"
        elif ns < 1000:
            FileName = result_path + '\ModelResult_0' + str(ns) + ".npy"
        else:
            FileName = result_path + '\ModelResult_' + str(ns) + ".npy"

        result = np.load(FileName)
        ModelResult.append(result)

    ModelResult = np.asarray(ModelResult)
    print(ModelResult.shape)

    # results = [idx, loss_train, loss_test, loss_blind, acc_pred_train, r2_train, err_train, corr_train,
    #           acc_pred_test, r2_test, err_test, corr_test, acc_pred_blind, r2_blind, err_blind, corr_blind,
    #           learning_rate, epochs, batch_size, h1, h2]
    loss_opt_test = [np.min(ModelResult[:, 1]),
                     np.min(ModelResult[:, 2])]  # the smallest loss for train and test data in this result
    accuracy_opt_test = np.max(ModelResult[:, 9])  # the highest r2 for test data in this result
    nbest_test1 = np.argmin(ModelResult[:, 3])   # the index for best of r2 score for test result
    # nbest_test2 = np.argmin(ModelResult[:, 2])  # the index for best of loss  for test result

    print("The smallest loss for training data , test data are:", loss_opt_test)
    print("The highest r2 score for test data is:", accuracy_opt_test)
    # print(nbest_test1, nbest_test2)
    print("The best hyper-parameter for highest R2 score in test results is:", ModelResult[nbest_test1, :])

    return ModelResult[nbest_test1, 16], ModelResult[nbest_test1, 17], ModelResult[nbest_test1, 18], \
           ModelResult[nbest_test1, 19], ModelResult[nbest_test1, 20]

# Create training and test data for RNN
def rnn_data(data_feature, data_label, batch_size, dtt=50):
    Nt, Ns = np.shape(data_feature)
    print(dtt)
    Ntt = int((Nt // dtt * dtt))  # the number used to cut the original data set, make it be the multiples of ns=50
    data_feature = data_feature[0:Ntt, :]
    data_feature = np.split(data_feature, dtt, axis=0)
    data_feature = np.array(data_feature)
    data_feature = np.transpose(data_feature, (1, 0, 2))
    print('data feature shape is', data_feature.shape)
    data_label = data_label[0:Ntt]
    data_label = np.split(data_label, dtt, axis=0)
    data_label = np.array(data_label)
    M1, N1 = data_label.shape
    data_label = np.reshape(data_label, (M1, 1, N1))
    data_label = np.transpose(data_label, (2, 1, 0))
    print('the label shape is', data_label.shape)

    data_feature = torch.tensor(data_feature).float()
    data_label = torch.tensor(data_label).float()

    print("data_feature shape", data_feature.shape)
    train_data = torch.utils.data.TensorDataset(data_feature, data_label)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    return train_loader

def rnn_data_v1(data_feature, data_label, batch_size, dtt=50):
    Nt, Nat = np.shape(data_feature)
    print(dtt)
    Ntt = int((Nt // dtt * dtt))  # the number used to cut the original data set, make it be the multiples of ns=50
    data_feature = data_feature[0:Ntt, :]
    data_feature = np.split(data_feature, dtt, axis=0)
    data_feature = np.array(data_feature)
    N1, M1, K1 = np.shape(data_feature)
    data_feature = np.reshape(data_feature, (N1, M1, K1))
    print('data feature shape is', data_feature.shape)
    data_feature = np.transpose(data_feature, (1, 0, 2))
    print('data feature shape is', data_feature.shape)
    data_label = data_label[0:Ntt]
    data_label = np.split(data_label, dtt, axis=0)
    data_label = np.array(data_label)
    #M1, N1 = data_label.shape
    #data_label = np.reshape(data_label, (M1, N1))
    data_label = np.transpose(data_label, (1, 0))
    print('the label shape is', data_label.shape)

    data_feature = torch.tensor(data_feature, requires_grad=True).float()
    data_label = torch.tensor(data_label, requires_grad=True).float()

    print("data_feature shape", data_feature.shape)
    train_data = torch.utils.data.TensorDataset(data_feature, data_label)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    return train_loader

class RnnGruModel_v1(nn.Module):
    def __init__(self, act_func="tanh"):
        super(RnnGruModel_v1, self).__init__()
        self.activation = nn.ReLU() if act_func=="relu" else nn.Tanh()

        self.gru1 = nn.RNN(input_size=1,
                     hidden_size=30,
                     num_layers=2,
                     zp0=11.0,
                     batch_first=True,
                     bidirectional=False)

        """self.gru2 = nn.RNN(input_size=60,
                        hidden_size=15,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False)"""

        #self.out = nn.Linear(in_features=30, out_features=1)

    def rnnModel(self, X):
        #print(X.shape)
        tmp_x = torch.transpose(X, 0, 1)
        #print(tmp_x.shape)
        #tmp_x = X
        #mlp_out = self.mlp(tmp_x)
        #print(mlp_out.shape)
        rnn_out1 = self.gru1(tmp_x)
        #print(rnn_out1.shape)
        #rnn_out2, _ = self.gru2(rnn_out1)
        #tmp_x = rnn_out1.transpose(0, 1)
        #print(tmp_x.shape)
        #y = self.out(tmp_x)
        #y = tmp_x
        #M, N, K = x.shape
        #x = x.detach().numpy()  detach() cannot be used, otherwise, the loss function will not iterable!!!
        #x = np.reshape(x, (M, 1, N))
        #x = torch.tensor(x).float()
        y = rnn_out1
        return y

class RnnGruModel_v2(nn.Module):
    def __init__(self, act_func="tanh"):
        super(RnnGruModel_v2, self).__init__()
        self.activation = nn.ReLU() if act_func=="relu" else nn.Tanh()

        self.gru1 = nn.RNN(input_size=1,
                     hidden_size=30,
                     num_layers=1,
                     batch_first=True,
                     bidirectional=False)

        """self.gru2 = nn.RNN(input_size=60,
                        hidden_size=15,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False)"""

        self.out = nn.Linear(in_features=30, out_features=1)

    def rnnModel(self, X):
        #print('X shape is', X.shape)
        tmp_x = torch.transpose(X, 0, 1)
        #print(tmp_x.shape)
        #tmp_x = X
        #mlp_out = self.mlp(tmp_x)
        #print(mlp_out.shape)
        rnn_out1, _ = self.gru1(tmp_x)
        #print('rnn_out1 is', rnn_out1.shape)
        #rnn_out2, _ = self.gru2(rnn_out1)
        #tmp_x = rnn_out1.transpose(0, 1)
        #print(tmp_x.shape)
        y = self.out(rnn_out1)
        #y = tmp_x
        #M, N, K = x.shape
        #x = x.detach().numpy()  detach() cannot be used, otherwise, the loss function will not iterable!!!
        #x = np.reshape(x, (M, 1, N))
        #x = torch.tensor(x).float()
        #y = tmp_x
        #print('y shape is', y.shape)

        y = torch.transpose(y, 1, 0)

        return y

class RnnGruModel(nn.Module):
    def __init__(self, act_func="tanh"):
        super(RnnGruModel, self).__init__()
        self.activation = nn.ReLU() if act_func == "relu" else nn.Tanh()

        self.mlp = nn.Sequential(nn.Linear(in_features=1600, out_features=60),
                                 self.activation,
                                 nn.Linear(in_features=60, out_features=15),
                                 self.activation,
                                 nn.Linear(in_features=15, out_features=10))

        self.gru1 = nn.GRU(input_size=10,
                           hidden_size=5,
                           num_layers=3,
                           batch_first=True,
                           bidirectional=False)

        """self.gru2 = nn.RNN(input_size=60,
                        hidden_size=15,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False)"""

        self.out = nn.Linear(in_features=5, out_features=1)

    def rnnModel(self, X):
        # print(X.shape)
        tmp_x = np.transpose(X, (1, 0, 2))
        # print(tmp_x.shape)
        # tmp_x = X
        mlp_out = self.mlp(tmp_x)
        # print(mlp_out.shape)
        rnn_out1, _ = self.gru1(mlp_out)
        # print(rnn_out1.shape)
        # rnn_out2, _ = self.gru2(rnn_out1)
        tmp_x = rnn_out1.transpose(0, 1)
        # print(tmp_x.shape)
        y = self.out(tmp_x)
        # y = tmp_x
        # M, N, K = x.shape
        # x = x.detach().numpy()  detach() cannot be used, otherwise, the loss function will not iterable!!!
        # x = np.reshape(x, (M, 1, N))
        # x = torch.tensor(x).float()
        y = y.transpose(-1, -2)
        return y

def training(model, train_loader, X_val, y_val, max_epoch, optimizer, criterion):
    train_loss = []
    val_loss = []
    y_val = np.reshape(y_val, (1, len(y_val)))
    y_val = torch.tensor(y_val).float()
    #print(list(model.parameters()))


    #count = 0
    for epoch in range(max_epoch):
        #optimizer = optim.Adam(list(model.parameters()), amsgrad=True)
        #model_par = list(model.parameters())
        #print('the length of model par', len(model_par), model_par[5])
        print(list(model.parameters())[0].grad)
        for x, y in train_loader:
            # print(x.shape)
            optimizer.zero_grad()
            #print('x shape is', x.shape)
            #print(model)
            y_pred = model.rnnModel(x)
            y_val_pred = model.rnnModel(X_val)
            #y_val_pred = np.reshape(y_val_pred.detach().numpy(), (len(y_val)))
            #y_val_pred = torch.tensor(y_val_pred).float()
            #y = torch.reshape(y, (y.shape[0], y.shape[1], 1))
            #y_val = torch.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
            #y = torch.reshape(y, (y.shape[1], 1, y.shape[0]))
            #print(y_val_pred.shape, y_val.shape, y_pred.shape, y.shape)

            loss = criterion(y_pred, y)
            #loss = torch.reshape(loss, (1, 1))
            #print(loss.shape, loss)

            val_loss_tmp = criterion(y_val_pred, y_val)
            train_loss.append(loss)
            #print(loss)
            val_loss.append(val_loss_tmp)
            #loss.requres_grad = True
            loss.backward()
            optimizer.step()
            #print(list(model.parameters()))
            #print(loss, val_loss_tmp)
            #print(count)
            #count = count +1
        print('epoch:', epoch, 'Training loss:', loss, 'Validatiaon loss:', val_loss_tmp)
    return model, train_loss, val_loss

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
class BaseDnnRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        super(BaseDnnRegressor, self).__init__()

    def drawLossPlot(self, train_loss, val_loss):
        font = {'family': 'normal', 'size': 18}
        plt.rc('font', **font)
        plt.figure(1)
        plt.plot(train_loss, c="black", label='Training')
        plt.plot(val_loss, c="r", label='Validation')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.show()

    def drawLossPlotEpoch(self, train_loss, val_loss, nit):
        train_loss1 = []
        val_loss1 = []
        Ns = len(train_loss)
        print(Ns, nit)
        for ni in range(Ns):
            if (ni+1) % nit == 0:
                #print(Ns, ni, train_loss[ni], val_loss[ni])
                train_loss1.append(train_loss[ni])
                val_loss1.append(val_loss[ni])
        if len(train_loss1) > 3:
            train_loss1 = train_loss1[3:]
            val_loss1 = val_loss1[3:]

        font = {'family': 'normal', 'size': 18}
        plt.rc('font', **font)
        plt.figure(1)
        plt.plot(train_loss1, c="black", label='Training')
        plt.plot(val_loss1, c="r", label='Validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()


from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
class RnnGruRegressor(BaseDnnRegressor):
    def __init__(self, data_mean, data_std, batch_size=50, dtt=50, max_epoch=50, lr0=0.001, act_func="relu"):
        super(RnnGruRegressor, self).__init__()
        self.batch_size = batch_size
        self.dtt = dtt
        self.max_epoch = max_epoch
        self.lr = lr0
        self.act_func = act_func
        self.mean = data_mean
        self.std = data_std
        self.model = None
        self.train_loss = []
        self.val_loss = []
        self.nit = None

    def fit(self, X, y, X_val, y_val):
        #X, y = check_X_y(X, y)
        self.nit = int(len(y) // self.dtt / self.batch_size)
        criterion = nn.MSELoss()
        train_loader = rnn_data_v1(X, y, self.batch_size, self.dtt)
        #print(train_loader)
        self.model = RnnGruModel_v1(act_func=self.act_func)
        optimizer = optim.Adam(list(self.model.parameters()), amsgrad=True, lr=self.lr)
        self.model, self.train_loss, self.val_loss = training(self.model, train_loader, X_val, y_val, self.max_epoch, optimizer, criterion)
        return self

    def predict(self, X):
        """Predict using the RNN GRU model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        #check_is_fitted(self)
        print(X.shape)
        N1, M1, K1 = X.shape
        y_pred = self.model.rnnModel(X)
        y_pred = y_pred.detach().numpy()
        #print(y_pred.shape)
        y_pred = np.reshape(y_pred, (M1))
        y_pred = unnormalize(y_pred, self.mean, self.std)
        return y_pred

    def lossPlot(self):
        self.drawLossPlot(self.train_loss, self.val_loss)

    def lossPlotE(self):
        self.drawLossPlotEpoch(self.train_loss, self.val_loss, self.nit)

def batch_out(X, y):
    X = X.detach().numpy()
    y = y.detach().numpy()
    batch_x = []
    batch_y = []
    Nh, Nt, Ns = X.shape
    for nh in range(Nh):
        batch_x.append(X[nh, :, :])
        y_tmp = y[nh, :]
        y_tmp= np.reshape(y_tmp, (len(y_tmp), 1))
        y_tmp = tf.convert_to_tensor(y_tmp)
        batch_y.append(y_tmp)
    #batch_x = tf.convert_to_tensor(batch_x)
    #batch_y = tf.convert_to_tensor(batch_y)

    return batch_x, batch_y

def MLP_training_v0(X_train, y_train, X_val, y_val, h1, h2, h3, lr, epochs, batch_size, display_freq=50):
    #start0 = time.time()

    tf.reset_default_graph()  #reset the whole graph
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    # print(y_blind)

    tf.reset_default_graph()  # reset the whole graph

    ncls = 1  # number of classes, used for regression, not classification

    Ntrn, Mtrn = X_train.shape
    Ntc, Mtc = X_val.shape

    # create one-hot shot encoded label
    y_train_label = encoded_label_r(y_train, Ntrn, ncls)
    y_val_label = encoded_label_r(y_val, Ntc, ncls)

    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, Mtrn], name='X')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

    fc1, Wt1, b1 = fc_layer(x, h1, 'Hidden_layer', use_relu=True)
    fc2, Wt2, b2 = fc_layer(fc1, h2, 'Hidden_layer1', use_relu=True)
    output_logits, Wt3, b3 = fc_layer(fc2, h3, 'Output_layer', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.squared_difference(output_logits, y))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    # print(Ntrn, batch_size)
    num_tr_iter = int(Ntrn / batch_size)
    loss_train_all = []
    loss_val_all = []
    loss_test_all = []

    with tf.Session() as sess:
        sess.run(init)
        global_step = 0

        for epoch in range(int(epochs)):
            if epoch % 5 == 0:
                print('Training epoch: {}'.format(epoch))
                print("num_tr_iter", num_tr_iter)
            for iteration in range(num_tr_iter):
                # print(Ntrn, batch_size)
                batch_x, batch_y = ctc_next_batch(X_train, y_train_label, Ntrn, int(batch_size))
                global_step += 1

                # Run optimization op (backprop)
                # print(batch_x.shape, batch_y.shape, type(batch_x))

                feed_dict_batch = {x: batch_x, y: batch_y}
                # y_pred =  sess.run(output_logits, feed_dict=feed_dict_batch)
                # print(batch_size, y_pred.shape)

                _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)

                #loss_batch = sess.run(loss, feed_dict=feed_dict_batch)

                feed_dict_train = {x: X_train, y: y_train_label}
                loss_train_tmp = sess.run(loss, feed_dict=feed_dict_train)
                loss_train_all.append(loss_train_tmp)

                feed_dict_val = {x: X_val, y: y_val_label}
                loss_val_tmp = sess.run(loss, feed_dict=feed_dict_val)
                loss_val_all.append(loss_val_tmp)

        ModelParMLP = []
        # output the model
        wt1 = sess.run(Wt1)
        b1 = sess.run(b1)

        wt2 = sess.run(Wt2)
        b2 = sess.run(b2)

        wt3 = sess.run(Wt3)
        b3 = sess.run(b3)

        ModelParMLP.append(wt1)
        ModelParMLP.append(b1)
        ModelParMLP.append(wt2)
        ModelParMLP.append(b2)
        ModelParMLP.append(wt3)
        ModelParMLP.append(b3)
    sess.close()

    return ModelParMLP, loss_train_all, loss_val_all

def MLP_training(input, dtt, X_val, y_val, h1, h2, h3, lr, epochs, batch_size, display_freq=50):
    #start0 = time.time()

    tf.reset_default_graph()  #reset the whole graph
    Mtrn = dtt
    Ntrn = 2
    X_val = X_val[:, 0:Mtrn, :]
    y_val = y_val[0:Mtrn]
    y_val = np.reshape(y_val, (1, len(y_val), 1))

    """y_train = np.ones(20)
    print(y_train)
    y_train_label = encoded_label_r(y_train, Mtrn, ncls)
    print(y_train_label.shape, y_train_label)"""


    """Ntrn, Mtrn = X_train.shape
    Ntt, Mtt = X_train_well.shape
    Nbd, Mbd = X_blind.shape
    Ntc, Mtc = X_test.shape

    # create one-hot shot encoded label
    y_train_label = encoded_label_r(y_train, Ntrn, ncls)
    y_test_label = encoded_label_r(y_test, Ntc, ncls)
    y_train_well_label = encoded_label_r(y_train_well, Ntt, ncls)
    y_blind_label = encoded_label_r(y_blind, Nbd, ncls)"""

    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, Mtrn, Ntrn], name='X')
        y = tf.placeholder(tf.float32, shape=[None, Mtrn, 1], name='Y')

    fc1, Wt1, b1 = fc_layer_edit(x, h1, 'Hidden_layer', use_relu=True)
    fc2, Wt2, b2 = fc_layer_edit(fc1, h2, 'Hidden_layer1', use_relu=True)
    output_logits, Wt3, b3 = fc_layer_edit0(fc2, h3, 'Output_layer', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.squared_difference(output_logits, y))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    # print(Ntrn, batch_size)
    #num_tr_iter = int(Ntrn / batch_size)
    loss_train_all = []
    loss_val_all = []


    with tf.Session() as sess:
        sess.run(init)
        global_step = 0

        for epoch in range(int(epochs)):
            if epoch % 5 == 0:
                print('Training epoch: {}'.format(epoch))
                #print("num_tr_iter", num_tr_iter)
            for x_tmp, y_tmp in input:
                # print(Ntrn, batch_size)
                #print(X.shape)
                x_tmp = x_tmp.detach().numpy()
                y_tmp = y_tmp.detach().numpy()
                batch_x = x_tmp

                batch_y = np.reshape(y_tmp, (y_tmp.shape[0], y_tmp.shape[1], 1))
                #batch_x, batch_y = batch_out(X, y)
                #print(batch_y[0].shape, type(batch_y[0]))
                global_step += 1
                #print(batch_x.shape, batch_y.shape, type(batch_x))
                # Run optimization op (backprop)
                feed_dict_batch = {x: batch_x, y: batch_y}
                #print(batch_x.shape, batch_y.shape, type(batch_x))

                #y_pred =  sess.run(output_logits, feed_dict=feed_dict_batch)
                #print('y_pred', y_pred.shape)

                #loss_tmp = sess.run(loss, feed_dict=feed_dict_batch)
                #print(loss_tmp)
                sess.run(optimizer, feed_dict=feed_dict_batch)

                loss_train_tmp = sess.run(loss, feed_dict=feed_dict_batch)
                loss_train_all.append(loss_train_tmp)

                # feed_dict_test = {x: X_test, y: y_test_label}

                feed_dict_test = {x: X_val, y: y_val}
                loss_val_tmp = sess.run(loss, feed_dict=feed_dict_test)
                loss_val_all.append(loss_val_tmp)

            print(loss_train_tmp, loss_val_tmp)

                #loss_batch = sess.run(loss, feed_dict=feed_dict_batch)



        #output the model
        wt1 = sess.run(Wt1)
        b1 = sess.run(b1)

        wt2 = sess.run(Wt2)
        b2 = sess.run(b2)

        wt3 = sess.run(Wt3)
        b3 = sess.run(b3)
        ModelParMLP = []
        ModelParMLP.append(wt1)
        ModelParMLP.append(b1)
        ModelParMLP.append(wt2)
        ModelParMLP.append(b2)
        ModelParMLP.append(wt3)
        ModelParMLP.append(b3)

    sess.close()

    return ModelParMLP, loss_train_all, loss_val_all

# MLP prediction model for basic version
def model_predict_v0(X, weights):
    Nt, Ns= X.shape
    X = np.asarray(X, dtype='float32')
    w1 = weights[0]
    b1 = weights[1]
    w2 = weights[2]
    b2 = weights[3]
    w3 = weights[4]
    b3 = weights[5]

    y_pred = []
    for nt in range(Nt):
        x_tmp = X[nt, :]
        x_tmp = np.reshape(x_tmp, (1, X.shape[1]))
        h1 = tf.matmul(x_tmp, w1)
        h1 += b1
        #h1 = tf.nn.leaky_relu(h1, alpha=0.2)
        h1 = tf.nn.relu(h1)
        h2 = tf.matmul(h1, w2)
        h2 += b2
        #h2 = tf.nn.leaky_relu(h2, alpha=0.2)
        h2 = tf.nn.relu(h2)
        h3 = tf.matmul(h2, w3)
        h3 += b3
        #h3 = tf.nn.relu(h3)

        #h3 = tf.Session().run(h3)
        #print(h3)
        y_pred.append(h3)
    with tf.Session() as sess:
        y_pred = sess.run(y_pred)
    sess.close()
    y_pred = np.asarray(y_pred)
    y_pred = np.reshape(y_pred, (y_pred.shape[0]))
    #print(y_pred.shape)

    return y_pred

def model_predict(X, weights):
    Nw, Nt, Ns = X.shape
    X = np.reshape(X, (Nt, Ns))
    X = np.asarray(X, dtype='float32')
    w1 = weights[0]
    b1 = weights[1]
    w2 = weights[2]
    b2 = weights[3]
    w3 = weights[4]
    b3 = weights[5]

    y_pred = []
    zp0 = 1.56
    for nt in range(Nt):
        x_tmp = X[nt, :]
        x_tmp = np.reshape(x_tmp, (1, X.shape[1]))
        h1 = tf.matmul(x_tmp, w1)
        h1 += b1
        h1 = tf.nn.leaky_relu(h1, alpha=0.2)
        h2 = tf.matmul(h1, w2)
        h2 += b2
        h2 = tf.nn.leaky_relu(h2, alpha=0.2)
        h3 = tf.matmul(h2, w3)
        h3 += b3

        if nt == 0:
            #out_tmp0 = h3 * zp0
            out_tmp0 = h3
        else:
            #out_tmp0 = h3 * y_pred[nt - 1]
            out_tmp0 = h3

        #h3 = tf.Session().run(h3)
        #print(h3)
        y_pred.append(out_tmp0)
    with tf.Session() as sess:
        y_pred = sess.run(y_pred)
    sess.close()
    y_pred = np.asarray(y_pred)

    return y_pred

class MLPR(BaseDnnRegressor):
    def __init__(self, data_mean, data_std, h1, h2, h3, batch_size=50, dtt=50, max_epoch=50, lr0=0.001, act_func="relu"):
        super(MLPR, self).__init__()
        self.batch_size = batch_size
        self.dtt = dtt
        self.max_epoch = max_epoch
        self.lr = lr0
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.act_func = act_func
        self.mean = data_mean
        self.std = data_std
        self.model = None
        self.train_loss = []
        self.val_loss = []
        self.nit = None

    def fit(self, X, y, X_val, y_val):
        #X, y = check_X_y(X, y)
        self.nit = int(len(y) // self.dtt / self.batch_size)
        criterion = nn.MSELoss()
        train_loader = rnn_data_v1(X, y, self.batch_size, self.dtt)
        #print(train_loader)
        self.weight, self.train_loss, self.val_loss = MLP_training(train_loader, self.dtt, X_val, y_val, self.h1, self.h2, self.h3,
                                                                   self.lr, self.max_epoch, self.batch_size)
        return self

    def predict(self, X):
        """Predict using the RNN GRU model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        #check_is_fitted(self)
        print(X.shape)
        N1, M1, K1 = X.shape

        y_pred = model_predict(X, self.weight)
        #y_pred = np.asarray(y_pred)
        #print(y_pred.shape)
        y_pred = np.reshape(y_pred, (M1))
        #print(y_pred)

        y_pred = unnormalize(y_pred, self.mean, self.std)
        #print(y_pred)
        #y_pred = np.asarray(y_pred, dtype='float64')

        return y_pred

    def lossPlot(self):
        self.drawLossPlot(self.train_loss, self.val_loss)

    def lossPlotE(self):
        self.drawLossPlotEpoch(self.train_loss, self.val_loss, self.nit)

# This version of MLP is the basic algorithm,, no modification
class MLPR_v0(BaseDnnRegressor):
    def __init__(self, data_mean, data_std, h1, h2, h3, batch_size=50, dtt=50, max_epoch=50, lr0=0.001, act_func="relu"):
        super(MLPR_v0, self).__init__()
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.lr = lr0
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.act_func = act_func
        self.mean = data_mean
        self.std = data_std
        self.model = None
        self.train_loss = []
        self.val_loss = []
        self.nit = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.nit = int(len(y_train) / self.batch_size)
        self.weight, self.train_loss, self.val_loss = MLP_training_v0(X_train, y_train, X_val, y_val, self.h1, self.h2, self.h3,
                                                                   self.lr, self.max_epoch, self.batch_size)
        return self

    def predict(self, X):
        """Predict using the MLP model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        #check_is_fitted(self)
        #print(X.shape)
        y_pred = model_predict_v0(X, self.weight)
        #y_pred = np.asarray(y_pred)
        #print(y_pred.shape)
        #y_pred = np.reshape(y_pred, (M1))
        #print(y_pred)

        y_pred = unnormalize(y_pred, self.mean, self.std)
        #print(y_pred)
        #y_pred = np.asarray(y_pred, dtype='float64')

        return y_pred

    def lossPlot(self):
        self.drawLossPlot(self.train_loss, self.val_loss)

    def lossPlotE(self):
        self.drawLossPlotEpoch(self.train_loss, self.val_loss, self.nit)



