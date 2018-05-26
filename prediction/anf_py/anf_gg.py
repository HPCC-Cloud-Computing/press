import numpy as np
import pandas as pd
import anf as anfis
import matplotlib.pyplot as plt

fname = "google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv"
header = [  "time_stamp", "numberOfTaskIndex", "numberOfMachineId",
            "meanCPUUsage", "canonical memory usage", "AssignMem",
            "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
            "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
            "max_disk_io_time", "cpi", "mai",
            "sampling_portion", "agg_type", "sampled_cpu_usage"]

df = pd.read_csv(fname, names=header)
mean_cpu_usage = df['meanCPUUsage']

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

data = np.asarray(gen_to_data(df, 20, 'meanCPUUsage'))

train_size = int(data.shape[0]*0.8)
test_size = data.shape[0] - train_size
print(train_size)

# Training data
x = data[:train_size,:-1]
y = data[:train_size,-1]

# Test data
x_test = data[train_size:,:-1]
y_test = data[train_size:,-1]

print(x_test.shape[0])
print(test_size)
a = anfis.ANFIS(x, y, 'gauss', 2)
a.hybridTraining()
print('test RMSE: ', np.sqrt(anfis.loss_function(a.predict(x_test), y_test)))
x_axis = np.arange(0, test_size, 1)
pred = plt.plot(x_axis, a.predict(x_test), label='predict')
act = plt.plot(x_axis, y_test, label='actual')
plt.legend()
plt.show()

