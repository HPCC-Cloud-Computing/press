import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVER'] = '2'


def premise_parameter(para_shape):
    para_init = {
        'mu': tf.Variable(tf.random_uniform(para_shape, minval=20.0, maxval=30.0)),
        'sigma': tf.Variable(tf.random_uniform(para_shape, minval=15.0, maxval=20.0))
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

    def predict(self, x):
        with tf.name_scope("reshape"):
            x_input = tf.tile(x, [self.rule_number, 1])

        with tf.name_scope('layer_1'):  # Fuzzification Layer
            # Premise parameters tmp = - tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
            # tf.square(self.w_fuzz['sigma']))
            fuzzy_sets = tf.exp(- tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
                                            tf.square(self.w_fuzz['sigma'])))

        with tf.name_scope('layer_2'):  # Rule-set Layer
            fuzzy_rules = tf.reduce_prod(fuzzy_sets, axis=1)

        with tf.name_scope('layer_3'):  # Normalization Layer
            normalized_fuzzy_rules = tf.reshape(fuzzy_rules / tf.reduce_sum(fuzzy_rules), [1, self.rule_number])

        with tf.name_scope('layer_4'):  # Defuzzification Layer
            f = tf.transpose(tf.add(tf.matmul(x, self.weights), self.bias))

        with tf.name_scope('layer_5'):  # Output Layer
            output = tf.matmul(normalized_fuzzy_rules, f)

        return output
