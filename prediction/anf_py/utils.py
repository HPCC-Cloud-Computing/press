#!/usr/bin/env python
import numpy as np
from skfuzzy import gaussmf, gbellmf, sigmf
import random
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
"""
    Cac ham phu tro cua ANFIS duoc de tai day
    ...
"""
func_dict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}


# random uniform
def rd(x, y):
    return random.uniform(x, y)


# premise parameter
def frame_parameter(rule_number: int, window_size: int,
                    mean1=20000, mean2=25000, sigma1=10000, sigma2=15000):
    x = [[['gaussmf', {'mean': rd(mean1, mean2), 'sigma': rd(sigma1, sigma2)}]
          for _ in np.arange(window_size)] for _ in np.arange(rule_number)]
    return np.asarray(x)


def consequence_parameter(rule_number: int, window_size: int):
    return np.ones((rule_number, window_size+1), dtype=float)


# Dau ra cua lop dau tien
# Lop thuc hien tinh toan do mo qua cac tap mo cho truoc
def first_layer(x: np.ndarray, fp: np.ndarray):
    ws, rn = fp.shape[1], fp.shape[0]
    temp = [[func_dict[fp[i][j][0]](x[j], **fp[i][j][1])
            for j in np.arange(ws)]
            for i in np.arange(rn)]
    return np.asarray(temp)


def loss_function(x, y):
    return mse(x, y)


# Dau ra cua lop thu 2
# Lop thuc hien tinh toan cac luat tu cac tap mo
def second_layer(ofl: np.ndarray):
    ws, rn = ofl.shape[1], ofl.shape[0]
    temp = np.ones(rn, dtype=float)
    for i in np.arange(rn):
        for j in np.arange(ws):
            temp[i] *= ofl[i][j]
    return temp


# Dau ra cua lop thu 3
def third_layer(osl: np.ndarray):
    return osl / osl.sum()


# Dau ra cua lop thu 4
def fouth_layer(otl: np.ndarray, x: np.ndarray, cp: np.ndarray):
    mat = np.append(x, 1)
    temp = [otl[i]*np.dot(mat, cp[i]) for i in np.arange(cp.shape[0])]
    return np.asarray(temp)


# Dau ra cuoi
def fifth_layer(ofl: np.ndarray):
    return sum(ofl)


def show_image(input_list: list):
    plt.plot(np.arange(1, len(input_list) + 1), input_list)
    plt.title('Training loss by epoch')
    plt.ylabel('Train loss')
    plt.xlabel('epoch')
    plt.axis([0, (len(input_list) + 1), 0, (max(input_list) + 1)])
    plt.show()


# Ham generate du lieu tu file ra data ma ANFIS co the train duoc
def gen_to_data(ss, window_size, attribute):
    window_size += 1
    d = np.asarray(ss[attribute])
    temp_data = []
    for i in np.arange(d.shape[0] - window_size):
        temp = []
        for j in np.arange(window_size):
            temp.append(d[i + j])
        temp_data.append(temp)
    return temp_data


def extract_data(raw_data, window_size, attribute, train_percentage):
    """

    :rtype: object
    """
    # data
    data = np.asarray(gen_to_data(raw_data, window_size, attribute))
    train_size = int(data.shape[0] * train_percentage)

    # Training data
    tmp_x_train = np.asarray(data[:train_size, :-1])
    x_train_ = np.reshape(tmp_x_train, [tmp_x_train.shape[0], 1, tmp_x_train.shape[1]])

    tmp_y_train = np.asarray(data[:train_size, -1])

    y_train_ = np.reshape(tmp_y_train, [tmp_y_train.shape[0], 1])
    # Test data
    tmp_x_test = np.asarray(data[train_size:, :-1])
    tmp_y_test = np.asarray(data[train_size:, -1])

    x_test_ = np.reshape(tmp_x_test, [tmp_x_test.shape[0], 1, tmp_x_test.shape[1]])
    y_test_ = np.reshape(tmp_y_test, [tmp_y_test.shape[0], 1])
    return x_train_, y_train_, x_test_, y_test_