from __future__ import division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import numpy.random as rnd
import os

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

data_path = os.path.expanduser("~") + "/Data/Mnist"
mnist = input_data.read_data_sets(data_path)


def inception_module(prev_layer, one,
                     reduce_one_to_three, three,
                     reduce_one_to_five, five,
                     pool):

    conv_1x1 = tf.layers.conv2d(prev_layer,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                filters=one,
                                padding="SAME",
                                kernel_size=1,
                                strides=1,
                                activation=tf.nn.elu)

    conv_1x1_reduce_to_3x3 = tf.layers.conv2d(prev_layer,
                                              filters=reduce_one_to_three,
                                              padding="SAME",
                                              kernel_size=1,
                                              strides=1,
                                              activation=tf.nn.elu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    conv_3x3 = tf.layers.conv2d(conv_1x1_reduce_to_3x3,
                                filters=three,
                                padding="SAME",
                                kernel_size=3,
                                strides=1,
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    conv_1x1_reduce_to_5x5 = tf.layers.conv2d(prev_layer,
                                              filters=reduce_one_to_five,
                                              padding="SAME",
                                              kernel_size=1,
                                              strides=1,
                                              activation=tf.nn.elu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    conv_5x5 = tf.layers.conv2d(conv_1x1_reduce_to_5x5,
                                filters=five,
                                padding="SAME",
                                kernel_size=5,
                                strides=1,
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    pool_3x3 = tf.layers.max_pooling2d(prev_layer,
                                       pool_size=pool,
                                       strides=1,
                                       padding="SAME")
    conv_1x1_from_pool = tf.layers.conv2d(pool_3x3,
                                          filters=pool,
                                          kernel_size=1,
                                          strides=1,
                                          activation=tf.nn.elu,
                                          padding="SAME",
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    layer_concat = tf.concat(
        [conv_1x1, conv_3x3, conv_5x5, conv_1x1_from_pool], axis=3)
    return layer_concat


height = 28
width = 28
channels = 1
n_inputs = height * width
learning_rate = 0.001

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
        y = tf.placeholder(tf.int32, shape=[None], name="y")
        is_training = tf.placeholder_with_default(
            False, shape=[], name="is_training")

    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(X_reshaped,
                                 filters=64,
                                 kernel_size=7,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # with tf.name_scope("max_pool_1"):
        # max_poo1_1 = tf.layers.max_pooling2d(conv1,
        #                                     pool_size=3,
        #                                     strides=2,
        #                                     padding="SAME")

    with tf.name_scope("LRN_1"):
        LRN_1 = tf.nn.local_response_normalization(conv1,
                                                   depth_radius=2,
                                                   alpha=0.00002,
                                                   beta=0.75,
                                                   bias=1)

    with tf.name_scope("conv2"):
        conv2_1 = tf.layers.conv2d(LRN_1,
                                   filters=64,
                                   kernel_size=1,
                                   strides=1,
                                   padding="SAME",
                                   activation=tf.nn.elu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        conv2_2 = tf.layers.conv2d(conv2_1,
                                   filters=192,
                                   kernel_size=3,
                                   strides=1,
                                   padding="SAME",
                                   activation=tf.nn.elu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    with tf.name_scope("LRN_2"):
        LRN_2 = tf.nn.local_response_normalization(conv2_2,
                                                   depth_radius=2,
                                                   alpha=0.00002,
                                                   beta=0.75,
                                                   bias=1)

    # with tf.name_scope("max_pool_2"):
    #    max_pool_2 = tf.layers.max_pooling2d(LRN_2,
    #                                         pool_size=3,
    #                                         strides=2,
    #                                         padding="SAME")

    with tf.name_scope("inception_1"):
        inception_1 = inception_module(LRN_2,
                                       one=64,
                                       reduce_one_to_three=96,
                                       reduce_one_to_five=16,
                                       three=128,
                                       five=32,
                                       pool=32)

    with tf.name_scope("inception_2"):
        inception_2 = inception_module(inception_1,
                                       one=128,
                                       reduce_one_to_three=128,
                                       reduce_one_to_five=32,
                                       three=192,
                                       five=96,
                                       pool=64)

    # with tf.name_scope("max_pool_3"):
        # max_pool_3 = tf.layers.max_pooling2d(inception_2,
        #                                     pool_size=3,
        #                                     strides=2,
        #                                     padding="SAME")

    with tf.name_scope("inception_3"):
        inception_3 = inception_module(inception_2,
                                       one=192,
                                       reduce_one_to_three=96,
                                       reduce_one_to_five=16,
                                       three=208,
                                       five=48,
                                       pool=64)

    with tf.name_scope("inception_4"):
        inception_4 = inception_module(inception_3,
                                       one=160,
                                       reduce_one_to_three=112,
                                       reduce_one_to_five=24,
                                       three=224,
                                       five=64,
                                       pool=64)

    with tf.name_scope("inception_5"):
        inception_5 = inception_module(inception_4,
                                       one=128,
                                       reduce_one_to_three=128,
                                       reduce_one_to_five=24,
                                       three=256,
                                       five=64,
                                       pool=64)

    with tf.name_scope("inception_6"):
        inception_6 = inception_module(inception_5,
                                       one=112,
                                       reduce_one_to_three=144,
                                       reduce_one_to_five=32,
                                       three=288,
                                       five=64,
                                       pool=64)

    with tf.name_scope("inception_7"):
        inception_7 = inception_module(inception_6,
                                       one=256,
                                       reduce_one_to_three=160,
                                       reduce_one_to_five=32,
                                       three=320,
                                       five=128,
                                       pool=128)

    # with tf.name_scope("max_pool_4"):
    #    max_poo1_4 = tf.layers.max_pooling2d(inception_7,
    #                                         pool_size=3,
    #                                         strides=2,
    #                                         padding="SAME")

    with tf.name_scope("inception_8"):
        inception_8 = inception_module(inception_7,
                                       one=256,
                                       reduce_one_to_three=160,
                                       reduce_one_to_five=32,
                                       three=320,
                                       five=128,
                                       pool=128)

    with tf.name_scope("inception_9"):
        inception_9 = inception_module(inception_8,
                                       one=384,
                                       reduce_one_to_three=192,
                                       reduce_one_to_five=48,
                                       three=384,
                                       five=128,
                                       pool=128)

    with tf.name_scope("avg_pool_1"):
        avg_pool_1 = tf.layers.average_pooling2d(inception_9,
                                                 pool_size=7,
                                                 strides=1,
                                                 padding="VALID")

    with tf.name_scope("dropout"):
        dropout = tf.layers.dropout(avg_pool_1,
                                    rate=0.4,
                                    training=is_training)

    with tf.name_scope("FCN"):
        FCN_1 = tf.layers.dense(dropout,
                                units=1,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        reshaped_FCN_1 = tf.reshape(FCN_1, [-1, 22 * 22])
        FCN_2 = tf.layers.dense(reshaped_FCN_1,
                                units=10,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=FCN_2)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(FCN_2, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 5
batch_size = 100

with tf.Session(graph=graph) as sess:
    init.run(session=sess)
    for epoch in range(n_epochs):

        for iteration in range(len(mnist.train.labels) // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        for iteration in range(len(mnist.test.labels) // batch_size):
            X_batch_test, Y_batch_test = mnist.test.next_batch(batch_size)
            acc_test = accuracy.eval(
                feed_dict={X: X_batch_test, y: Y_batch_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
