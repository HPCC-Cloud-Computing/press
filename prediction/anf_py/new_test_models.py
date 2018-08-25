import new_models
import pandas as pd
from utils import extract_data
import tensorflow as tf

# Network features
WINDOW_SIZE = 20
RULE_NUMBER = 3
ATTRIBUTE = 'meanCPUUsage'
p_para_shape = [WINDOW_SIZE, RULE_NUMBER]
TRAIN_PERCENTAGE = 0.8
BATCH_SIZE = 50
EPOCH = 500
LEARNING_RATE = 1e-4

# Ten file duoc dua vao ANFIS network
fname = "google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv"

# Cac Header trong file
header = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
          "meanCPUUsage", "canonical memory usage", "AssignMem",
          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
          "max_disk_io_time", "cpi", "mai",
          "sampling_portion", "agg_type", "sampled_cpu_usage"]


def main():
    # Khai bao thong tin ve ten file dua vao anfis
    df = pd.read_csv(fname, names=header)
    # Trich xuat va chia tach cac tap tu file dau vao
    x_train, y_train, x_test, y_test = extract_data(df, window_size=WINDOW_SIZE,
                                                    attribute=ATTRIBUTE, train_percentage=TRAIN_PERCENTAGE)
    # Khai bao ANFIS network
    anf_model = new_models.ANFIS(window_size=WINDOW_SIZE, rule_number=RULE_NUMBER)

    # Bat dau huan luyen mang anfis
    # anf_model.train(x_train=x_train, y_train=y_train,
    #                 batch_size=x_train.shape[0], epoch=EPOCH, rate=LEARNING_RATE, save_path='tmp/test1.cpkt')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1, WINDOW_SIZE])
    z = anf_model.output(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        op = sess.run(z, feed_dict={x: x_train})
    print(f"output: {op}")
    # anf_model.figure(x_test=x_test, y_test=y_test, load_path='tmp/test1.cpkt')
    # pred = anf_model.predict(x_test=x_test, load_path='tmp/test1.cpkt')
    # print(pred)


if __name__ == '__main__':
    main()