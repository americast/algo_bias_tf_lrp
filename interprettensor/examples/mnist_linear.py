'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin 
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import pandas as pd
import keras



import tensorflow as tf
import numpy as np
import pdb
import pudb

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 2,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 1000,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 1,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_linear_logs','Summaries directory')
flags.DEFINE_boolean("relevance", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/epsilon/ww/flat/alphabeta')
flags.DEFINE_boolean("save_model", True,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_linear_model','Checkpoint dir')


FLAGS = flags.FLAGS

def nn():
    return Sequential([Linear(input_dim=166, output_dim=256, act ='relu', batch_size=FLAGS.batch_size),
                 Linear(256, act ='relu'), 
                 Linear(128, act ='relu'), 
                 Linear(64, act ='relu'), 
                 Linear(64, act ='relu'), 
                 Linear(32, act ='relu'), 
                 Linear(16, act ='relu'),
                 Linear(8, act ='relu'),
                 Linear(3, act ='relu'),
                 Softmax()])


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# input dict creation as per tensorflow source code
def feed_dict(final_X, final_Y, train):    
    if train:
        xs, ys = next_batch(FLAGS.batch_size, final_X, final_Y)
        k = FLAGS.dropout
    else:
        xs = final_X
        ys = final_Y
        k = 1.0
    return xs, ys, k

def train(train_X, train_y):
  # Import data
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # pu.db
  # config = tf.ConfigProto(
  #         device_count = {'GPU': 0}
  #     )
  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 166], name='x-input')
        y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, 3], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    # Model definition along with training and relevances
    with tf.variable_scope('model'):
        net = nn()
        # pu.db
        y = net.forward(x)
        
    saver = tf.train.Saver()
        
    with tf.variable_scope('relevance'):    
        if FLAGS.relevance:
            LRP = net.lrp(y,FLAGS.relevance_method, 1)
            
            # LRP layerwise 
            relevance_layerwise = []
            R = y
            for layer in net.modules[::-1]:
                print("layer here: ", layer)
                R = net.lrp_layerwise(layer, R, 'epsilon',1)
                relevance_layerwise.append(R)
        else:
            LRP = []
            relevance_layerwise = []
            
    # Accuracy computation
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out 
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()
    
    utils = Utils(sess, FLAGS.checkpoint_dir)
    if FLAGS.reload_model:
        utils.reload_model()

    print("y.shape: "+str(y.shape)+" y_.shape: "+str(y_.shape))
    trainer = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])

    uninit_vars = set(tf.global_variables()) - set(tf.trainable_variables())
    tf.variables_initializer(uninit_vars).run()
            
    # iterate over train and test data
    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:
            #pdb.set_trace()
            d = feed_dict(train_X, train_y, False)
            final_X = tf.data.Dataset.from_tensor_slices(d[0])
            final_Y = tf.data.Dataset.from_tensor_slices(d[1])
            test_inp = {x:final_X, y_:final_Y, keep_prob:d[2]}
            summary, acc , relevance_test, op, rel_layer= sess.run([merged, accuracy, LRP,y, relevance_layerwise], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
            print([rel for rel in rel_layer])
            print(np.sum(relevance_test))
            saver.save(sess, str(FLAGS.checkpoint_dir)+"/model_epoch_"+str(i)+".ckpt")

            
        else:
            d = feed_dict(final_X, final_Y, True)
            inp = {x:d[0], y_:d[1], keep_prob:d[2]}
            summary, _ , relevance_train,op, rel_layer= sess.run([merged, trainer.train, LRP,y, relevance_layerwise], feed_dict=inp)
            train_writer.add_summary(summary, i)
            
            
    # relevances plotted with visually pleasing color schemes
    # if FLAGS.relevance:
    #     # plot test images with relevances overlaid
    #     images = d[0].reshape([FLAGS.batch_size,28,28,1])
    #     plot_relevances(relevance_test.reshape([FLAGS.batch_size,28,28,1]), images, test_writer )
    # plot train images with relevances overlaid
    # images = inp[inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
    # images = (images + 1)/2.0
    # plot_relevances(relevance_train.reshape([FLAGS.batch_size,28,28,1]), images, train_writer )

    train_writer.close()
    test_writer.close()


def main(_):
    global EPOCHS
    # train_X, test_X, train_y, test_y = get_iris_data()

    # Saver
    name = ""

    print("Train? (y for train, n for test)")
    choice = input()
    train_flag = True
    if (choice =='n' or choice=='N'):
          df = pd.read_csv("data/out-test.csv")
          FLAGS.batch_size = df.shape[0]
          FLAGS.max_steps = 1
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

    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train(train_X, train_y)


if __name__ == '__main__':
    tf.app.run()
