# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html

import tensorflow as tf
import numpy as np

from import_mnist_data import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1001):
        # train in batches of 100 of a possible 55k images & labels
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            x: batch_xs,
            y_: batch_ys    
        })

        # test against different set of 10k images and labesl
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        if i % 200 == 0:
            print('Accuracy after {} iterations = {}'.format(
                i, 
                sess.run(accuracy, feed_dict={
                    x: mnist.test.images,
                    y_: mnist.test.labels
                })
            ))
