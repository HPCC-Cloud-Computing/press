import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

if __name__ == '__main__':
    # file contain data
    filename = "data/dataset/dataset/data_resource_usage_10Minutes_6176858948.csv"

    # load data set: "mean_cpu_usage", "canonical_memory_usage"
    header = ["time_stamp", "number_of_task_index", "number_of_machine_id",
              "mean_cpu_usage", "canonical_memory_usage", "assign_mem",
              "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
              "mean_disk_io_time", "mean_local_disk_space", "max_cpu_usage",
              "max_disk_io_time", "cpi", "mai",
              "sampling_portion", "agg_type", "sampled_cpu_usage"]
    df = pd.read_csv(filename, names=header)
    dataset = np.array(df[["mean_cpu_usage", "canonical_memory_usage"]])
    dataset = dataset.astype('float32')

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scale = scaler.fit_transform(dataset)

    # split data for training and testing
    # train: cpu and mem
    # test: cpu
    train_size = int(dataset.shape[0] * 0.8)
    x_train = data_scale[:train_size - 1, :]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    y_train = data_scale[1:train_size, 0]
    x_test = data_scale[train_size:len(data_scale) - 1, :]
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    y_test = data_scale[train_size + 1:, 0]
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # create model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 2)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # learn
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.125)

    # plot history
    plt.figure('History', figsize=(16, 9))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # predict
    y_predict = model.predict(x_test)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
    y_predict = np.concatenate((y_predict.reshape(len(y_test), 1), x_test[:, 1:]), axis=1)
    y_predict = scaler.inverse_transform(y_predict)[:, 0]
    y_test = np.concatenate((y_test.reshape(len(y_test), 1), x_test[:, 1:]), axis=1)
    y_test = scaler.inverse_transform(y_test)[:, 0]
    print(y_predict)
    print(y_test)
    print(mean_squared_error(y_test, y_predict))
    plt.figure('Result', figsize=(16, 9))
    plt.plot(y_predict, label='predict')
    plt.plot(y_test, label='actual')
    plt.legend()
    plt.show()
