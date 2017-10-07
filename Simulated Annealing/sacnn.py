from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from scipy.constants import Boltzmann
import random
from numpy import exp
from numpy import abs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# So class trong bo du lieu MNIST, dai dien cho cac so tu 0 den 9
NUM_CLASSES = 10

# Kich co anh trong bo du lieu MNIST
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
KERNEL_SIZE = 5
NUM_FEATURE_MAPS = 6
STDDEV_INDEX = 0.1
NUM_FEATURE_MAPS_2 = 12
NUM_FEATURE = 92
BATCH_SIZE = 50
FLAGS = None
EPOCH_NUMBER = 1
TRAINING_SIZE = 60000

# Cac tham so lien quan den giai thuat SA
NEIGHBOR_NUMBER = 10
REDUCE_FACTOR = 0.9
TEMPERATURE_INIT = 100
BOLTZMANN_CONSTANT = Boltzmann

# bien toan cuc
w_conv1 = None
b_conv1 = None
w_conv2 = None
b_conv2 = None
w_fc1 = None
b_fc1 = None
w_fc2 = None
b_fc2 = None


# Khai bao bien chua cac tham so ve weight voi kich thuoc cua kernel va so feature map
def weight_variable(weight_shape):
    weight_init = tf.truncated_normal(weight_shape, stddev=STDDEV_INDEX)
    return tf.Variable(weight_init)


# Khai bao bien chua cac tham so ve bias tuon

def bias_variable(bias_shape):
    bias_init = tf.constant(0.1, shape=bias_shape)
    return tf.Variable(bias_init)


# Khai bao convolution layer 2 chieu voi day du cac buoc
def conv2d(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')


# Khai bao subsample 2x2 feature map (max_pool_2x2)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Xay dung khoi tao mang Neural Network
def deep_network(x):
    global w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2
    # Reshape du lieu de sung dung ben trong mang neuron
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, IMAGE_SIZE,
                                 IMAGE_SIZE, 1])

    # Layer dau tien - map mot image ra 6 feature maps
    with tf.name_scope('conv1'):
        # Weight
        w_conv1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE,
                                   1, NUM_FEATURE_MAPS])
        # Bias
        b_conv1 = bias_variable([NUM_FEATURE_MAPS])
        # Activation function duoc su dung la ham ReLU
        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

    # Layer pool/subsampling
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Layer convolution thu 2 -- noi 6 feature maps thanh 12
    with tf.name_scope('conv2'):
        # Weight
        w_conv2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE,
                                   NUM_FEATURE_MAPS, NUM_FEATURE_MAPS_2])
        # Bias
        b_conv2 = bias_variable([NUM_FEATURE_MAPS_2])
        # Activation function su dung ham ReLU
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    # Layer pooling/subsampling thu 2
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connect layer 1 sau 2 lan pool
    # Anh 28x28 tro thanh 7x7x12 feature maps - ket noi toi 92 feature
    with tf.name_scope('fc1'):
        # Weight fc_1
        w_fc1 = weight_variable([7 * 7 * NUM_FEATURE_MAPS_2, NUM_FEATURE])
        b_fc1 = bias_variable([92])

        h_pool2_flatten = tf.reshape(h_pool2, [-1, 7 * 7 * 12])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, w_fc1) + b_fc1)

    # Su dung dropout de kiem soat do phuc tap cua mo hinh
    with tf.name_scope('dropout'):
        dropper = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, dropper)

    # Ghep 92 feature vao 10 class, tuong duong voi cac so
    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([92, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    return y_conv, dropper


# Lay mot lan can cua ma tran chua tham so
def neighbor(x):
    delta = tf.random_normal(shape=x.get_shape(), mean=0.0, stddev=0.001*tf.reduce_mean(x))
    x = x + delta
    return x


def main(_):
    # Nhap du lieu
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Tao mo hinh
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])

    # Ham mat mat
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Khoi tao do thi deep net
    y_conv, keep_prob = deep_network(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    # Toi uu theo giai thuat co san tren tensorflow
    with tf.name_scope('momentum'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    # Gia tri duoc dung lam ham toi thieu
    accuracy = tf.reduce_mean(correct_prediction)

    # Luu temp graph
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        global w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2
        sess.run(tf.global_variables_initializer())
        while 1:
            for epoch in range(EPOCH_NUMBER):
                for i in range(1200):
                    batch = mnist.train.next_batch(BATCH_SIZE)
                    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                test_accuracy = accuracy.eval(feed_dict={
                       x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
                })

            # Tien hanh thuat toan SA
            # Back up tra lai gia tri cac tham so neu khong co thay doi
            print("Tham so truoc SA")
            print(sess.run(b_conv1))
            print(test_accuracy)
            back_up = w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2
            # Khoi tao gia tri toi uu ban dau
            f = test_accuracy
            x0 = back_up
            temperature = TEMPERATURE_INIT
            for n in range(NEIGHBOR_NUMBER):
                # w_conv1, b_conv1, w_conv2, b_conv2 = neighbor(w_conv1), neighbor(b_conv1), \
                #                                      neighbor(w_conv2), neighbor(b_conv2)
                # w_fc1, b_fc1, w_fc2, b_fc2 = neighbor(w_fc1), neighbor(b_fc1), neighbor(w_fc2), neighbor(b_fc2)
                sess.run(w_conv1.assign(neighbor(w_conv1))), sess.run(b_conv1.assign(neighbor(b_conv1)))
                sess.run(w_conv2.assign(neighbor(w_conv2))), sess.run(b_conv2.assign(neighbor(b_conv2)))
                sess.run(w_fc1.assign(neighbor(w_fc1))), sess.run(b_fc1.assign(neighbor(b_fc1)))
                sess.run(w_fc2.assign(neighbor(w_fc2))), sess.run(b_fc2.assign(neighbor(b_fc2)))

                w_fc1.eval(), b_fc1.eval(), w_fc2.eval(), b_fc2.eval()
                # Gan cac tham so cho cac gia tri lan can
                # Gia tri ham dem xet
                print("Tham so duoc xet:")
                print(sess.run(b_conv1))
                f_delta = accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
                })
                print(f_delta)
                if f_delta > f:
                    f_new = f_delta
                else:
                    df = f - f_delta
                    r = random.uniform(0, 1)
                    # Dieu kien phan bo boltzmann
                    if r > exp(-df/Boltzmann/temperature):
                        f_new = f_delta
                    else:
                        f_new = f
                        # Tra lai tham so ban dau
                        w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2 = x0
                f = f_new
                temperature = REDUCE_FACTOR * temperature
                termination_criterion = abs(test_accuracy/f - 1)
                # Dieu kien dung cua SA
            if (termination_criterion > -0.02) and (termination_criterion < 0.02):
                print(" SA Test accuracy    : " + str(accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
                })))
                break
            print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
