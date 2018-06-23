#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
from skfuzzy import gaussmf
import tensorflow as tf
import numpy as np
# Just disables the warning, doesn't enable AVX/FMA
import os
import anf
WINDOW_SIZE = 2
RULE_NUMBER = 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# b = tf.placeholder(tf.float32, (2,))
# x = tf.placeholder(tf.float32, [None, WINDOW_SIZE])
# y = tf.placeholder(tf.float32, (1,))


def anfis_model(z: np.ndarray):
    """
    Return predict value of ANFIS model
    """
    mean1, mean2, sigma1, sigma2 = 25.0, 40.0, 15.0, 20.0
    x = np.ones((2, 2))
    y = x
    a = anf.ANFIS(x, y, 'gauss', RULE_NUMBER, epoch=10)
    a.fix_p_para(mean1, mean2, sigma1, sigma2)
    return tf.convert_to_tensor(a.output_single(z))


def train():
    pass


def summary():
    pass

print(anfis_model(np.asarray([0.5, 2])))
tf.distributions