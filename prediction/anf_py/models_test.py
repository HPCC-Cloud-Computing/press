import numpy as np
import models
import pandas as pd

# Network features
WINDOW_SIZE = 5
RULE_NUMBER = 50
ATTRIBUTE = 'meanCPUUsage'
p_para_shape = [WINDOW_SIZE, RULE_NUMBER]
TRAIN_PERCENTAGE = 0.8
BATCH_SIZE = 50
EPOCH = 100
LEARNING_RATE = 1e-4

fname = "google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv"
# Cac Header trong file
header = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
          "meanCPUUsage", "canonical memory usage", "AssignMem",
          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
          "max_disk_io_time", "cpi", "mai",
          "sampling_portion", "agg_type", "sampled_cpu_usage"]


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


def extract_data(raw_data, window_size, attribute):
    """

    :rtype: object
    """
    # data
    data = np.asarray(gen_to_data(raw_data, window_size, attribute))
    train_size = int(data.shape[0] * TRAIN_PERCENTAGE)

    # Training data
    tmp_x_train = np.asarray(data[:train_size, :-1])
    x_train_ = np.reshape(tmp_x_train, [tmp_x_train.shape[0], 1, tmp_x_train.shape[1]])

    tmp_y_train = np.asarray(data[:train_size, -1])
    y_train_ = np.reshape(tmp_y_train, [tmp_y_train.shape[0], 1])

    # Test data
    tmp_x_test =np.asarray(data[train_size:, :-1])
    tmp_y_test = np.asarray(data[train_size:, -1])

    x_test_ = np.reshape(tmp_x_test, [tmp_x_test.shape[0], 1, tmp_x_test.shape[1]])
    y_test_ = np.reshape(tmp_y_test, [tmp_y_test.shape[0], 1])
    return x_train_, y_train_, x_test_, y_test_


def main():
    df = pd.read_csv(fname, names=header)
    anf_model = models.ANFIS(window_size=WINDOW_SIZE, rule_number=RULE_NUMBER)
    x_train, y_train, x_test, y_test = extract_data(df, window_size=WINDOW_SIZE, attribute=ATTRIBUTE)
    anf_model.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                    batch_size=BATCH_SIZE, epoch=EPOCH, rate=LEARNING_RATE)


if __name__ == '__main__':
    main()
