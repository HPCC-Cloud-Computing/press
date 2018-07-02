import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time
# import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khai bao hang so
WINDOW_SIZE = 5
RULE_NUMBER = 30
ATTRIBUTE = 'meanCPUUsage'
p_para_shape = [WINDOW_SIZE, RULE_NUMBER]
TRAIN_PERCENTAGE = 0.8

fname = "google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv"
# Cac Header trong file
header = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
          "meanCPUUsage", "canonical memory usage", "AssignMem",
          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
          "max_disk_io_time", "cpi", "mai",
          "sampling_portion", "agg_type", "sampled_cpu_usage"]

# Lay du lieu tu file csv
df = pd.read_csv(fname, names=header)
mean_cpu_usage = df[ATTRIBUTE]

premise_parameters = {
    'mu': tf.Variable(tf.random_normal(p_para_shape)),
    'sigma': tf.Variable(tf.random_normal(p_para_shape))
}


# Initialize variables as Premise Parameter
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


# Ham generate du lieu tu file ra data ma ANFIS co the train duoc
def gen_to_data(ss, window_size, attribute):
    window_size += 1
    d = np.asarray(ss[attribute])
    temp_data = []
    for i in np.arange(d.shape[0] - window_size):
        temp = []
        for j in np.arange(window_size):
            temp.append(d[i+j])
        temp_data.append(temp)
    return temp_data


# Initialize ANFIS network
class ANFIS:
    def __init__(self):
        self.w_fuzz = premise_parameter([RULE_NUMBER, WINDOW_SIZE])
        self.weights = consequence_parameter([WINDOW_SIZE, RULE_NUMBER])
        self.bias = consequence_parameter([1, RULE_NUMBER])

    def predict(self, x):
        with tf.name_scope("reshape"):
            x_input = tf.tile(x, [RULE_NUMBER, 1])

        with tf.name_scope('layer_1'):  # Fuzzification Layer
            # Premise parameters tmp = - tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
            # tf.square(self.w_fuzz['sigma']))
            fuzzy_sets = tf.exp(- tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
                                            tf.square(self.w_fuzz['sigma'])))

        with tf.name_scope('layer_2'):  # Rule-set Layer
            fuzzy_rules = tf.reduce_prod(fuzzy_sets, axis=1)

        with tf.name_scope('layer_3'):  # Normalization Layer
            normalized_fuzzy_rules = tf.reshape(fuzzy_rules / tf.reduce_sum(fuzzy_rules), [1, RULE_NUMBER])

        with tf.name_scope('layer_4'):  # Defuzzification Layer
            f = tf.transpose(tf.add(tf.matmul(x, self.weights), self.bias))

        with tf.name_scope('layer_5'):  # Output Layer
            output = tf.matmul(normalized_fuzzy_rules, f)
        return output


def main():
    # data
    model = ANFIS()
    data = np.asarray(gen_to_data(df, WINDOW_SIZE, 'meanCPUUsage'))[:100]
    train_size = int(data.shape[0]*TRAIN_PERCENTAGE)
    data_size = data.shape[0]
    test_size = data.shape[0] - train_size

    # Training data
    y_train = data[:train_size, -1]

    # Test data
    x_data = data[:, :-1]
    y_test = data[train_size:, -1]

    # Save model
    saver = tf.train.Saver()

    # Tao dau vao dau ra cua network
    z = time.time()
    with tf.name_scope('loss'):
        x = tf.placeholder(tf.float32, [data_size, 1, WINDOW_SIZE])
        y_data_predict = tf.convert_to_tensor([model.predict(x[i]) for i in np.arange(data_size)])
        y_train_predict = y_data_predict[:train_size]
        y_test_predict = y_data_predict[train_size:]
        y_test = tf.reshape(y_test, [test_size, 1, 1])
        y_train = tf.reshape(y_train, [train_size, 1, 1])
        loss = tf.losses.mean_squared_error(y_train_predict, y_train)
        accuracy = tf.sqrt(tf.losses.mean_squared_error(y_test_predict, y_test))
        print(time.time() - z)

    with tf.name_scope('train'): # Phan nay ton nhieu chi phi khoi tao
        z = time.time()
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss=loss)
        print(time.time() - z)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_data = np.reshape(x_data, [data_size, 1, WINDOW_SIZE]).astype(np.float32)
        for i in range(100000):
            train_step.run(feed_dict={x: x_data})
            point = sess.run(loss, feed_dict={x: x_data})
            print("Loop", i, " .Loss: ", point)
        print(sess.run(accuracy, feed_dict={x: x_data}))
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
        # act = sess.run(y_test_predict, feed_dict={x: x_data})[:, 0, 0]
        # y_test = sess.run(y_test)[:, 0, 0]
        # x_axis = np.arange(0, test_size, 1)
        # plt.title('Google cluter timeseries: ' + str(ATTRIBUTE))
        # plt.plot(x_axis, act, label='predict')
        # plt.plot(x_axis, y_test, label='actual')
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    main()
