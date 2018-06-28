#!/usr/bin/env python

import tensorflow as tf
from skfuzzy import gaussmf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khai bao hang so
WINDOW_SIZE = 10
RULE_NUMBER = 20
p_para_size = [WINDOW_SIZE, RULE_NUMBER]
# Create a Tensorflow constant
const = tf.constant(2.0, name="const")

# Create Tensorflow variables
b = tf.Variable(2.0, name="b")
c = tf.Variable(3.0, name="c")
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

premise_parameters = {
    'h1': tf.Variable(tf.random_normal(p_para_size)),
    'out': tf.Variable(tf.random_normal(p_para_size))
}

x = tf.placeholder([WINDOW_SIZE, ])
x_test = tf.convert_to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
init_op = tf.global_variables_initializer()


def models(input_series):
    
    return 0


def training():
    pass


def summary():
    pass


def figures():
    pass


with tf.Session() as sess:
    sess.run(init_op)
    k = sess.run(tf.exp(premise_parameters['h1'], name='k'))
    print(k)



