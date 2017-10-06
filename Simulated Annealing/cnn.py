from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

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
EPOCH_NUMBER = 10
TRAINING_SIZE = 60000


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

    return y_conv, dropper, w_conv1, w_conv2, b_conv1, b_conv2,


def main(_):
    # Nhap du lieu
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Tao mo hinh
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Khoi tao do thi deep net
    y_conv, keep_prob = deep_network(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCH_NUMBER):
            for i in range(1200):
                batch = mnist.train.next_batch(BATCH_SIZE)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            test_accuracy = accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
            })
            print("Epoch "+str(epoch+1)+" : Test accuracy: "+str(test_accuracy))
        print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
