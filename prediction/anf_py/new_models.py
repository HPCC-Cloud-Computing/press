import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from random import uniform as random
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt
import seaborn as sb

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

        return output

    def train(self, x_train, y_train, batch_size, epoch, rate, load_path=None, save_path=None):
        """

        :rtype: object
        """
        # Session
        net = tf.InteractiveSession()
        # Placeholder
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Cost function
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))

        # Entire train loss function

        # Optimizer
        optimizer = tf.train.AdamOptimizer(rate).minimize(cost)

        # Loader and Saver()
        load_and_save = tf.train.Saver()

        # Init session
        net.run(tf.global_variables_initializer())

        if load_path is not None:
            print(f"Loading model to directory {load_path} ...")
            load_and_save.restore(net, load_path)

        # Start training

        for e in range(epoch):
            # Optimizing
            net.run(optimizer, feed_dict={x: x_train, y: y_train})
            # # Shuffle training data
            # shuffle = np.random.permutation(np.arange(len(y_train)))
            # x_train = x_train[shuffle]
            # y_train = y_train[shuffle]
            #
            # # Based-batch training
            # for i in np.arange(0, len(y_train) // batch_size):
            #     start = i * batch_size
            #     batch_x = x_train[start:start + batch_size]
            #     batch_y = y_train[start:start + batch_size]
            #
            #     # Optimizing
            #     net.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            #
            # # lost_value = net.run(lost, feed_dict={x: x_train, y: y_train})
            loss_value = net.run(cost, feed_dict={x: x_train, y: y_train})
            print(f"Epoch: {e} - Loss function: {loss_value}")

        # Save model
        if save_path is not None:
            print(f"Saving model to directory {save_path} ...")
            load_and_save.save(net, save_path)

        net.close()
        # duration = time.time() - start_time
        # print(duration)

    def hybrid_sa_training(self, x_train, y_train, x_test, y_test, batch_size, epoch,
                           rate, temp_init, neighbor_number, reduce_factor):
        # Session
        print("Initializing Session ...")
        net = tf.InteractiveSession()

        # Placeholder
        print("Initializing Placeholder ...")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])

        # Cost function
        print("Initializing Loss Function and optimizer ...")
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))

        # Entire train loss function
        # Phan code nay dung de lay cac gia tri cua train loss, su dung de ve~ do thi hoi tu loss
        # lost = tf.reduce_mean(tf.squared_difference(self.predict(x, x_train.shape[0]), y))
        # lost_list = []
        sa_loss = tf.reduce_mean(tf.squared_difference(self.output(x), y))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(rate).minimize(cost)
        # Test loss
        # acc la bien dua ra gia tri ve RMSE cua tap test so voi gia tri du doan duoc
        acc = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.output(x), y)))
        # Su dung pred nay de dua ra duoc MAE vi MAE khong co san trong tensorflow
        # Viec nay khong anh huong nhieu den code performance nen khong can dua het vao tensor cung duoc
        pred = self.output(x, x_test.shape[0])

        # Init session
        net.run(tf.global_variables_initializer())
        print("Start training ...")
        # Start training
        for e in range(epoch):
            # Shuffle training data
            shuffle = np.random.permutation(np.arange(len(y_train)))
            x_train = x_train[shuffle]
            y_train = y_train[shuffle]
            print("Epoch: ", e)

            # Based-batch training
            print("Gradient Descent ...")
            for i in np.arange(0, len(y_train) // batch_size):
                start = i * batch_size
                batch_x = x_train[start:start + batch_size]
                batch_y = y_train[start:start + batch_size]

                # Optimizing
                net.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # SA training ngay sau do
            previous_parameters = self.w_fuzz, self.weights, self.bias
            temp = temp_init
            f0 = net.run(sa_loss, feed_dict={x: x_train, y: y_train})
            print("Simulated Annealing ...")
            for n in range(neighbor_number):
                net.run(self.w_fuzz['mu'].assign(neighbor(self.w_fuzz['mu'])))
                net.run(self.w_fuzz['sigma'].assign(neighbor(self.w_fuzz['sigma'])))
                net.run(self.weights.assign(neighbor(self.weights)))
                net.run(self.bias.assign(neighbor(self.bias)))
                f = net.run(sa_loss, feed_dict={x: x_train, y: y_train})
                if f < f0:
                    f_new = f
                    previous_parameters = self.w_fuzz, self.weights, self.bias
                else:
                    df = f - f0
                    r = random(0, 1)
                    if r > np.exp(-df / Boltzmann / temp):
                        f_new = f
                        previous_parameters = self.w_fuzz, self.weights, self.bias
                    else:
                        f_new = f0
                        self.w_fuzz, self.weights, self.bias = previous_parameters
                f0 = f_new
                temp = reduce_factor * temp
            test_rmse = net.run(acc, feed_dict={x: x_test, y: y_test})  # RMSE test
            pred_value = net.run(pred, feed_dict={x: x_test})
            test_mae = mean_absolute_error(pred_value, y_test)  # MAE test
            print(e, 'Test-RMSE: ', test_rmse, 'MAE:', test_mae)
            tf.train.Saver().save('tmp/model.cpkt')

        net.close()

    def predict(self, x_test: np.ndarray, load_path=None):
        # Initialize
        with tf.name_scope("Initialize"):
            # Initialize Session
            print("Initializing session ... ")
            net = tf.InteractiveSession()

            # Initialize Placeholder
            print("Initializing Placeholder ..")
            x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])

            # Restore model from path and computing result from input x_test
            print(f"Restoring models from path: {load_path}")
            # Load model
            try:
                tf.train.Saver().restore(net, load_path)
            except ValueError:
                print("Invalid directory")
                return None

            predict = self.output(x)

        # Start session
        with tf.name_scope("output_test"):
            net.run(tf.global_variables_initializer())
            print("Computing result ...")
            result = net.run(predict, feed_dict={x: x_test})
        print("Done")
        return result

    def figure(self, x_test: np.ndarray, y_test: np.ndarray,
               load_path: str):
        predicted = self.predict(x_test=x_test, load_path=load_path)
        actual = y_test
        sb.set()
        plt.title("Result")
        plt.plot(actual, label="Actual")
        plt.plot(predicted, label="Predict")
        plt.legend()
        plt.show()

    def summary(self):
        pass

    def save(self):
        pass