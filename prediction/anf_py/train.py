import tensorflow as tf
import models


def train(rule_number, window_size):
    model = models.ANFIS(rule_number, window_size)

    with tf.name_scope("Save model"):
        pass


