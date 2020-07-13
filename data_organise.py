import numpy as np
import tensorflow as tf
import math


'''Module to organise the data set. We are keeping examples by row i.e. m x n where m is the number of examples'''


def balance_data(raw_X, raw_Y):
    '''Givens an unevenly distributed data set returns an evenly distributed data set (i.e. removes entries from the abundant classes)'''
    positive = []
    negative = []
    neutral = []

    for i in range(len(raw_X)):
        if raw_Y[i] == 'positive':
            positive.append([raw_X[i], raw_Y[i]])
        elif raw_Y[i] == 'negative':
            negative.append([raw_X[i], raw_Y[i]])
        elif raw_Y[i] == 'neutral':
            neutral.append([raw_X[i], raw_Y[i]])

    length = min(len(positive), len(negative), len(neutral))
    lst = [positive, negative, neutral]
    for i in range(len(lst)):
        np.random.shuffle(lst[i])

    data_set = positive[0:length] + negative[0:length] + neutral[0:length]

    balanced_X = [element[0] for element in data_set]
    balanced_Y = [element[1] for element in data_set]

    return balanced_X, balanced_Y


def balance_data_binary(raw_X, raw_Y):
    '''Givens an unevenly distributed binary data set returns an evenly distributed data set (i.e. removes entries from the abundant classes)'''
    positive = []
    negative = []

    for i in range(len(raw_X)):
        if raw_Y[i] == 1:
            positive.append([raw_X[i], raw_Y[i]])
        elif raw_Y[i] == 0:
            negative.append([raw_X[i], raw_Y[i]])

    length = min(len(positive), len(negative))

    # shuffle both positive and negative lists
    lst = [positive, negative]
    for i in range(len(lst)):
        np.random.shuffle(lst[i])

    data_set = positive[0:length] + negative[0:length]

    balanced_X = [element[0] for element in data_set]
    balanced_Y = [element[1] for element in data_set]

    return balanced_X, balanced_Y


def convert_to_one_hot(labels, C):
    '''Takes an m x 1 list of categories and returns a one_hot encoded numpy array Y of size m x C where the m is the number of examples and C is the number of classes'''
    D = tf.constant(C, name='D')  # 'D' for 'depth'
    one_hot_matrix = tf.one_hot(labels, axis=1, depth=D)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


def split_train_test(X, Y, split=0.7):
    '''Takes a data set X, Y where samples are row-wise (np.shape[0]) and returns a training set, X_train, Y_train and a test set X_test, Y_test divided according to 'split' ratio (defaults to 0.7)'''
    p = np.random.permutation(len(X))
    X_1 = np.array(X)
    Y_1 = np.array(Y)
    X_shuff, Y_shuff = X_1[p], Y_1[p]
    X_train, Y_train = X_shuff[0:int(len(X_shuff) * split)], Y_shuff[0:int(len(Y_shuff) * split)]
    X_test, Y_test = X_shuff[len(X_train):], Y_shuff[len(Y_train):]
    return X_train, Y_train, X_test, Y_test


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    # completely ripped from DeepLearning AI
    np.random.seed(seed)
    m = X.shape[1]
    num_classes = Y.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((num_classes, m))
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:(num_complete_minibatches * mini_batch_size + m % mini_batch_size)]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:(num_complete_minibatches * mini_batch_size + m % mini_batch_size)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def remove_empty_rows(X, Y):
    '''removes rows where there is no text entry in X i.e. no CV to correspond to the person

    params:
            X (m x n)
            Y (m x 1)
    '''
    rows_to_remove = [i for i in range(len(X)) if X[i] == None]
    X_return = [X[j] for j in range(len(X)) if j not in rows_to_remove]
    Y_return = [Y[k] for k in range(len(Y)) if k not in rows_to_remove]
    return X_return, Y_return
