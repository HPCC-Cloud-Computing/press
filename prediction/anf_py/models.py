from datetime import datetime

import numpy as np
import tensorflow as tf


# Initialize variables as Premise Parameter
def premise_parameter(para_shape, min_mu=10.0, max_mu=15.0, min_sigma=5.0, max_sigma=10.0):
    """

    :param para_shape:
    :param min_mu:
    :param max_mu:
    :param min_sigma:
    :param max_sigma:
    :return:
    """
    para_init = {
        'mu': tf.Variable(tf.random_uniform(para_shape, minval=min_mu, maxval=max_mu)),
        'sigma': tf.Variable(tf.random_uniform(para_shape, minval=min_sigma, maxval=max_sigma))
    }
    return para_init


# Initialize variables as Consequence Parameter
def consequence_parameter(para_shape):
    para_init = tf.random_normal(para_shape)
    return tf.Variable(para_init)


# Create a number of neighbors of value x
# for Simulated Annealing
def neighbor(x):
    delta = tf.random_normal(shape=x.get_shape(), mean=0.0, stddev=0.001 * tf.reduce_mean(x))
    x = x + delta
    return x


class ANFIS:

    def __init__(self, rule_number, window_size):
        """
        :param rule_number:
        :param window_size:
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

    def train(self, x_train, y_train,
              epoch, rate,
              x_test=None, y_test=None,
              save_path=None,
              tracking=None):
        """
        Kieu train anfis thuan tuy, su dung duy nhat GD
        Training theo kich ban 1
        :param x_train:
        :param y_train:
        :param epoch:
        :param rate:
        :param x_test:
        :param y_test:
        :param save_path:
        :return:
        """
        # Placeholder
        print(f'{datetime.now()}: \t Creating Placeholders ... ')
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Cost function
        print(f'{datetime.now()}: \t Creating Cost function ... ')
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))

        # Optimizer
        print(f'{datetime.now()}: \t Creating Optimizer ... ')
        optimizer = tf.train.AdamOptimizer(rate).minimize(cost)

        # Loader and Saver()
        saver = tf.train.Saver(max_to_keep=None)

        min_points = 9999
        # Start training

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Load model from path
            for e in range(epoch):
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})

                point = sess.run(cost, feed_dict={x: x_test, y: y_test})
                loss_value = sess.run(cost, feed_dict={x: x_train, y: y_train})
                if point < min_points:
                    min_points = point
                print(
                    f"{datetime.now()}: Epoch: {e} - Loss function: "
                    f"{loss_value} - point: {point} - minimum: {min_points}")
            if save_path is not None:
                print(f"Saving model to directory {save_path} ...")
                saver.save(sess, save_path)
        return

    def sa1_train(self):
        """
        Training theo kich ban 2
        Trong 1 epoch, thuc hien GD truoc, sau do thuc hien SA
        """
        pass

    def sa2_train(self):
        """
        Training theo kich ban 3
        :return:
        """
        pass

    def sa3_train(self):
        """
        Training theo kich ban 4
        """
        pass

    def sa4_train(self):
        """
        Training theo kich ban 5
        """
        pass

    def predict(self, input_data, load_path=None):
        # Initialize
        with tf.name_scope("Initialize"):
            # Initialize Placeholder
            print("Initializing Placeholder ..")
            # tf.reset_default_graph()
            x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])

            loader = tf.train.Saver()

            predict = self.output(x)
            with tf.Session() as sess:
                if load_path is not None:
                    print(f"Restoring models from path: {load_path}")
                    loader.restore(sess, load_path)
                # Restore model from path and computing result from input x_test

                print("Computing result ...")
                result = sess.run(predict, feed_dict={x: input_data})
                print("Done")
        return result

    def mse(self, x, y, load_path):
        # Initialize
        with tf.name_scope("Initialize"):
            # Initialize Placeholder
            print("Initializing Placeholder ..")
            # tf.reset_default_graph()
            x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])

            loader = tf.train.Saver()

            mse = tf.reduce_mean(tf.squared_difference(self.output(x_placeholder), y))

            with tf.Session() as sess:
                if load_path is not None:
                    print(f"Restoring models from path: {load_path}")
                    loader.restore(sess, load_path)
                # Restore model from path and computing result from input x_test

                print("Computing result ...")
                result = sess.run(mse, feed_dict={x_placeholder: x})
                print(result)
                print("Done")
        return result

    def compare_images(self, x, y, load_path):
        # predicted = self.predict(x, load_path=load_path)
        pass
