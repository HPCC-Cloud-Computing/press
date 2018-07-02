
import numpy as np
import models
import pandas as pd
WINDOW_SIZE = 20
RULE_NUMBER = 100
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
#

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

anf_model = models.ANFIS(window_size=WINDOW_SIZE, rule_number=RULE_NUMBER)
# x = np.asarray([[[1, 1]], [[4, 3]]], dtype=np.float32)
# y = np.asarray([[[1]], [[2]]])
# anf_model.train(x_train=x, y_train=y, epoch=10000)

# Data
data = np.asarray(gen_to_data(df, WINDOW_SIZE, 'meanCPUUsage'))
train_size = int(data.shape[0]*TRAIN_PERCENTAGE)
data_size = data.shape[0]
# test_size = data.shape[0] - train_size

# Training data
y_train = np.asarray(data[:train_size, -1])
y_train = np.reshape(y_train, [y_train.shape[0], 1])
# Test data
x_train = np.asarray(data[:train_size, :-1])
x_train = np.reshape(x_train, [x_train.shape[0], 1, x_train.shape[1]])
x_test = np.asarray(data[train_size:, :-1])
y_test = np.asarray(data[train_size:, -1])
x_test = np.reshape(x_test, [x_test.shape[0], 1, x_test.shape[1]])
y_test = np.reshape(y_test, [y_test.shape[0], 1])
anf_model.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, batch_size=200, epoch=5000)


