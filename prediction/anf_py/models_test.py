import tensorflow as tf
import models

WINDOW_SIZE = 2


def test(x):
    anf_model = models.ANFIS(window_size=WINDOW_SIZE, rule_number=5)
    return anf_model.predict(x)


def main():
    x = tf.placeholder(tf.float32, [1, WINDOW_SIZE])
    pred = test(x)
    x_test = [[1, 2]]
    print('ss')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z = sess.run(pred(x), feed_dict={x: x_test})
        print(z)
        print('1')
