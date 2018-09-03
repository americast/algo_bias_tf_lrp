from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import keras


import sys
sys.path.append("..")
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.avgpool import AvgPool
from modules.convolution import Convolution
import modules.render as render
from modules.utils import Utils, Summaries, plot_relevances
import input_data
import pudb

EPOCHS = 100
BATCH_SIZE = 1000
LEARNING_RATE = 0.0001

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


# def init_weights(shape):
#     """ Weight initialization """
#     weights = tf.random_normal(shape, stddev=0.1)
#     return tf.Variable(weights)

# def forwardprop(X, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9):
#     """
#     Forward-propagation.
#     IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
#     """
#     h_1 = tf.nn.selu((tf.matmul(X, w_1)))  # The \sigma function
#     h_2 = tf.nn.selu((tf.matmul(h_1, w_2)))  # The \sigma function
#     h_3 = tf.nn.selu((tf.matmul(h_2, w_3)))  # The \sigma function
#     h_4 = tf.nn.selu((tf.matmul(h_3, w_4)))  # The \sigma function
#     h_5 = tf.nn.selu((tf.matmul(h_4, w_5)))  # The \sigma function
#     h_6 = tf.nn.selu((tf.matmul(h_5, w_6)))  # The \sigma function
#     h_7 = tf.nn.selu((tf.matmul(h_6, w_7)))  # The \sigma function
#     h_8 = tf.nn.selu((tf.matmul(h_7, w_8)))  # The \sigma function
#     yhat = tf.nn.selu((tf.matmul(h_8, w_9)))  # The \sigma function
    
#     return yhat

# def get_iris_data():
#     """ Read the iris data set and split them into training and test sets """
#     iris   = datasets.load_iris()
#     data   = iris["data"]
#     target = iris["target"]

#     # Prepend the column of 1s for bias
#     N, M  = data.shape
#     all_X = np.ones((N, M + 1))
#     all_X[:, 1:] = data

#     # Convert into one-hot vectors
#     num_labels = len(np.unique(target))
#     all_Y = np.eye(num_labels)[target]  # One liner trick!
#     return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    global EPOCHS, BATCH_SIZE, LEARNING_RATE
    # train_X, test_X, train_y, test_y = get_iris_data()

    # Saver
    name = ""

    print("Train? (y for train, n for test)")
    choice = input()
    train_flag = True
    if (choice =='n' or choice=='N'):
        df = pd.read_csv("data/out-test.csv")
        BATCH_SIZE = df.shape[0]
        EPOCHS = 1
        train_flag = False
        name = input("Enter model file name: ")
    else:
         df = pd.read_csv("data/out-train.csv")



    cols = df.columns.values
    cols = np.delete(cols, [1])
    train_X = df.loc[:,cols].values

    train_y = df["decile_score"].values
    y_train_ = train_y
    train_y = keras.utils.np_utils.to_categorical(train_y)



    print(train_X.shape)
    print(train_y.shape)
    # exit()
    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    # h_size_1 = 256                                # Number of hidden nodes
    # h_size_2 = 256                                # Number of hidden nodes
    # h_size_3 = 128                                # Number of hidden nodes
    # h_size_4 = 64                                  # Number of hidden nodes
    # h_size_5 = 64                                  # Number of hidden nodes
    # h_size_6 = 32                                  # Number of hidden nodes
    # h_size_7 = 16                                  # Number of hidden nodes
    # h_size_8 = 8                                  # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    net = Sequential([Linear(input_dim=166, output_dim=256, act ='relu', batch_size=BATCH_SIZE),
                 Linear(256, act ='relu'), 
                 Linear(128, act ='relu'), 
                 Linear(64, act ='relu'), 
                 Linear(64, act ='relu'), 
                 Linear(32, act ='relu'), 
                 Linear(16, act ='relu'),
                 Linear(8, act ='relu'),
                 Linear(3, act ='relu'),
                 Softmax()])

    output = net.forward(tf.convert_to_tensor(X))

    trainer = net.fit(output, y, loss='softmax_crossentropy', optimizer='adam', opt_params=[LEARNING_RATE])

if __name__ == '__main__':
    main()
