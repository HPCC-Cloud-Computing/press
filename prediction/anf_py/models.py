import tensorflow as tf
import numpy as np


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

    def predict(self, x, input_size):
        """

        :param input_size:
        :param x:
        :return:
        """
        with tf.name_scope("reshape"):
            x_input = tf.tile(x, [1, self.rule_number, 1])

        with tf.name_scope('layer_1'):  # Fuzzification Layer
            # Premise parameters tmp = - tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
            # tf.square(self.w_fuzz['sigma']))
            fuzzy_sets = tf.exp(- tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
                                            tf.square(self.w_fuzz['sigma'])))
        with tf.name_scope('layer_2'):  # Rule-set Layer
            fuzzy_rules = tf.reduce_prod(fuzzy_sets, axis=2)

        with tf.name_scope('layer_3'):  # Normalization Layer
            sum_fuzzy_rules = tf.expand_dims(tf.reduce_sum(fuzzy_rules, axis=1), axis=1)
            normalized_fuzzy_rules = tf.expand_dims(tf.divide(fuzzy_rules[0], sum_fuzzy_rules[0]), axis=0)
            for i in np.arange(1, input_size):
                tmp_norm_rules = tf.expand_dims(tf.divide(fuzzy_rules[i], sum_fuzzy_rules[i]), axis=0)
                normalized_fuzzy_rules = tf.concat([normalized_fuzzy_rules, tmp_norm_rules], 0)
            normalized_fuzzy_rules = tf.expand_dims(normalized_fuzzy_rules, axis=1)

        with tf.name_scope('layer_4_5'):  # Defuzzification Layer and Output Layer
            f = tf.squeeze(tf.expand_dims(tf.transpose(tf.add(tf.matmul(x[0], self.weights), self.bias)), axis=0), [0])
            output = tf.matmul(normalized_fuzzy_rules[0], f)
            for i in np.arange(1, input_size):
                tmp_f = tf.squeeze(tf.expand_dims(
                    tf.transpose(tf.add(tf.matmul(x[i], self.weights), self.bias)), axis=0), [0])
                tmp_output = tf.matmul(normalized_fuzzy_rules[i], tmp_f)
                output = tf.concat([output, tmp_output], 0)
        return output

    def train(self, x_train, y_train, x_test, y_test, batch_size, epoch, rate):
        """

        :rtype: object
        """
        # Session
        net = tf.InteractiveSession()
        # Placeholder
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Cost function
        cost = tf.reduce_mean(tf.squared_difference(self.predict(x, batch_size), y))

        # Entire train loss function
        # lost = tf.reduce_mean(tf.squared_difference(self.predict(x, x_train.shape[0]), y))
        # lost_list = []

        # Optimizer
        optimizer = tf.train.AdamOptimizer(rate).minimize(cost)
        # Test loss
        acc = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.predict(x, x_test.shape[0]), y)))
        # Init session
        net.run(tf.global_variables_initializer())
        # Start training
        for e in range(epoch):
            # Shuffle training data
            shuffle = np.random.permutation(np.arange(len(y_train)))
            x_train = x_train[shuffle]
            y_train = y_train[shuffle]

            # Based-batch training
            for i in np.arange(0, len(y_train) // batch_size):
                start = i * batch_size
                batch_x = x_train[start:start + batch_size]
                batch_y = y_train[start:start + batch_size]

                # Optimizing
                net.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # lost_value = net.run(lost, feed_dict={x: x_train, y: y_train})
            test = net.run(acc, feed_dict={x: x_test, y: y_test})
            print(e, 'Test: ', test)
            # lost_list.append(lost_value)
        net.close()
        # duration = time.time() - start_time
        # print(duration)

    def summary(self):
        pass

    def save(self):
        pass
