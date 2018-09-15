from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

plt.switch_backend('agg')


# Save figure for 1-dimension data
def save_figures(data, label,
                 x_label, y_label,
                 path):
    array = np.asarray(data)
    x = np.arange(1, array.shape[0] + 1)
    plt.plot(x, array, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(path)


# Initialize variables as Premise Parameter
def premise_parameter(para_shape,
                      min_mu=10.0, max_mu=15.0,
                      min_sigma=5.0, max_sigma=10.0):
    """

    :param para_shape:
    :param min_mu:
    :param max_mu:
    :param min_sigma:
    :param max_sigma:
    :return:
    # """
    para_init = {
        'mu': tf.Variable(tf.random_uniform(para_shape, minval=min_mu, maxval=max_mu)),
        'sigma': tf.Variable(tf.random_uniform(para_shape, minval=min_sigma, maxval=max_sigma))
    }
    return para_init


# Adding timestamp for writer
def writer(context):
    print(f"{datetime.now()}: {context}")


# Initialize variables as Consequence Parameter
def consequence_parameter(para_shape):
    para_init = tf.random_normal(para_shape)
    return tf.Variable(para_init)


class ANFIS:
    def __init__(self,
                 rule_number=5, window_size=20):
        """
        :param rule_number: So luat trong mo hinh mang Takagi-Sugeno
        :param window_size: Kich thuoc lay mau cho input dau vao
        """
        self.rule_number = rule_number
        self.window_size = window_size
        self.premise_shape = [rule_number, window_size]
        self.consequence_shape_weights = [window_size, rule_number]
        self.consequence_shape_bias = [1, rule_number]
        self.w_fuzz = premise_parameter(self.premise_shape)
        self.weights = consequence_parameter(self.consequence_shape_weights)
        self.bias = consequence_parameter(self.consequence_shape_bias)
    
    def output(self, x: np.ndarray):
        """
        Show list of outputs from list of inputs
        :param x:
        :return output:
        """
        # Reshape
        with tf.name_scope("reshape"):
            x_input = tf.tile(x, [1, self.rule_number, 1])
        
        # Fuzzification Layer
        with tf.name_scope('layer_1'):
            fuzzy_sets = tf.exp(- tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
                                            tf.square(self.w_fuzz['sigma'])))
        # Rule-set Layer
        with tf.name_scope('layer_2'):
            fuzzy_rules = tf.reduce_prod(fuzzy_sets, axis=2)
        
        # Normalization Layer
        with tf.name_scope('layer_3'):
            sum_fuzzy_rules = tf.reduce_sum(fuzzy_rules, axis=1)
            normalized_fuzzy_rules = tf.divide(fuzzy_rules, tf.reshape(sum_fuzzy_rules, (-1, 1)))
        
        # Defuzzification Layer and Output Layer
        with tf.name_scope('layer_4_5'):
            f = tf.add(tf.matmul(tf.reshape(x, (-1, self.window_size)), self.weights), self.bias)
            output = tf.reduce_sum(tf.multiply(normalized_fuzzy_rules, f), axis=1)
        
        return tf.reshape(output, (-1, 1))
    
    # Traning ANFIS model
    def train(self,
              x_train, y_train,
              epoch=10000, rate=1e-2,
              tracking_loss=False,
              load_path=None, save_path=None
              ):
        """

        :param x_train: Inputs to train
        :param y_train: Labels to train
        :param epoch: Number epoch to train
        :param rate: Learning rate to train
        :param tracking_loss: Flags if you wanna track loss values
        :param load_path: Flags if you wanna load model from specified path
        :param save_path: Cpecified path you wanna save model
        :return:
        """
        # Creating Placeholder
        writer("Creating Placeholder ... ")
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        
        # Creating cost and optimizer
        writer("Creating cost and optimizer")
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)
        
        saver = tf.train.Saver()
        
        # Check tracking_loss flags
        track_list = np.empty((0,))
        
        # Initializing session
        with tf.Session() as sess:
            
            # Check Model path Loading
            if load_path is not None:
                saver.restore(sess, load_path)
            
            # Start training
            sess.run(tf.global_variables_initializer())
            writer("Starting train ... ")
            for e in range(1, epoch + 1):
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})
                writer(f"{e}: {c}")
                # Appened new loss value to track_list
                if tracking_loss:
                    track_list = np.append(track_list, c)
            
            # Check save_path
            if save_path is not None:
                saver.save(sess, save_path)
        writer(track_list)
        # Saving figures
        fig_path = f"{save_path}_tracking.png"
        writer(f"Saving tracking figures to {fig_path} ")
        save_figures(data=track_list, label="Loss_function",
                     x_label='epoch', y_label='loss value',
                     path=fig_path)
    
    # Compute loss from input data and compare output with labels
    def loss(self,
             x_test, y_test,
             load_path=None):
        """

        :param x: Input to execute
        :param y: label to compare
        :param load_path: Load model from this path
        :return: loss function
        """
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            # Check Model path Loading
            sess.run(tf.global_variables_initializer())
            if load_path is not None:
                saver.restore(sess, load_path)
            # op = sess.run(self.output(x_), feed_dict={x_: x})
            mse = sess.run(cost, feed_dict={x: x_test, y: y_test})
            writer(f"mse: {mse}")
        return mse
