from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khai bao hang
STDDEV_INDEX = 0.1


# Khoi tao bien chua cac tham so ve weight voi kich thuoc cua kernel
def weight_variable(weight_shape):
    weight_init = tf.truncated_normal(weight_shape, stddev=STDDEV_INDEX)
    return tf.Variable(weight_init)


# Khoi tao bien chua cac tham so va bias tuong ung
def bias_variable(bias_shape):
    bias_init = tf.constant(0.1, shape=bias_shape)
    return tf.Variable(bias_init)


# Khai bao convolution layer voi day du cac buoc
def conv1d(x, weight):
    return tf.nn.conv1d(x, weight, stride=2, padding='SAME')

# Khai bao pooling/subsampling 2 feate
