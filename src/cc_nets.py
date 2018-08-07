# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def crowd_counting_cnn(inputs, scope='crowd_counting_cnn'):
    with tf.variable_scope(scope) as sc:
        net = slim.conv2d(inputs, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME', scope='conv1')
        net = slim.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME', scope='conv2')

        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 128, [5, 5], activation_fn=tf.nn.relu, padding='SAME', scope='conv3')
        net = slim.conv2d(net, 128, [5, 5], activation_fn=tf.nn.relu, padding='SAME', scope='conv4')

        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME', scope='conv5')
        net = slim.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME', scope='conv6')

        net = slim.max_pool2d(net, [2, 2], 1, padding='SAME', scope='pool3')
        net = slim.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME', scope='conv7')
        net = slim.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME', scope='conv8')

        net = slim.max_pool2d(net, [2, 2], 1, padding='SAME', scope='pool4')
        net = slim.conv2d(net, 1024, [3, 3], activation_fn=tf.nn.relu, padding='SAME', scope='conv9')
        net = slim.conv2d(net, 1024, [3, 3], activation_fn=tf.nn.relu, padding='SAME', scope='conv10')

        net = slim.conv2d(net, 1, [1, 1], activation_fn=tf.nn.relu, padding='SAME', scope='conv11')
        head_crowd = tf.reduce_sum(net, axis=[1, 2, 3])

        return net, head_crowd


if __name__ == '__main__':
    print('run m_model')
