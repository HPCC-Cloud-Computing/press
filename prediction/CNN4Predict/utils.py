from datetime import datetime
import codecs
import math
import numpy as np
import matplotlib.pyplot as plt

INTERVAL_BY_SECOND = 600


# Month converter function
def month_converter(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
              'Oct', 'Nov', 'Dec']
    return months.index(month) + 1


# Xu ly du lieu thoi gian tung ban ghi log thanh dang so
class ExecuteLine(object):
    def __init__(self, reg='', interval=60):
        self.reg = reg
        self.interval = interval

    def set_interval(self, interval):
        self.interval = interval

    def get_interval(self):
        return self.interval

    def time(self):
        self.reg.index('1')

    def detect_time(self):
        return self.reg[self.reg.find('[') + 1: self.reg.find(']')]

    def convert_time(self):
        try:
            temp_time = self.detect_time().split()
            hms = temp_time[0][temp_time[0].index(':') + 1:]
            hms = hms.split(':')
            dt = temp_time[0][:temp_time[0].index(':')].split('/')
            since = datetime(1970, 8, 15, 0, 0, 0)
            time_sample = datetime(int(dt[2]), month_converter(dt[1]),
                                   int(dt[0]), int(hms[0]), int(hms[1]),
                                   int(hms[2]))
        except ValueError:
            return 0
        return int((time_sample - since).total_seconds())


# Dem so request trong cac khoang thoi gian cho truoc
class CountRequest(object):
    def __init__(self, file, interval=600):
        self.file = file
        self.interval = interval

    def request_list(self):
        min_value = 0
        count_line = 0
        temp_request_list = []

        with codecs.open(self.file, "r", encoding='utf-8',
                         errors='ignore') as f:
            for line in f:
                time_element = ExecuteLine(line).convert_time()
                condition_append_list = math.floor(
                    (int(time_element) - min_value) / self.interval) >= len(
                    temp_request_list)
                if time_element == 0:
                    continue
                if count_line == 0:
                    min_value = time_element
                    count_line += 1
                if condition_append_list:
                    temp_request_list.append(1)
                else:
                    temp_request_list[-1] += 1
        return temp_request_list

    def output(self, output_file):
        with codecs.open(output_file, "wb+", encoding='utf-8',
                         errors='ignore') as f:
            for element in self.request_list():
                f.write(str(element) + '\n')


# Xu ly tren tung file mot
def get_single_data(targeted_file):
    temp_list = []
    with codecs.open(targeted_file, "r", encoding='utf-8',
                     errors='ignore') as f:
        for e in f:
            temp_list.append(int(e))
    return temp_list


# Nhan du lieu tren cac tap cho truoc
def get_data(start, end, interval):
    s = np.arange(start, end + 1, 1)
    temp_output = []
    for index in s:
        if index < 10:
            targeted_file = 'dataset_folder/wc_day' + '0' + str(
                index) + '_1_' + str(interval) + '.out'
        else:
            targeted_file = 'dataset_folder/wc_day' + str(index) + '_1_' + str(
                interval) + '.out'
        temp_output += get_single_data(targeted_file)
    return temp_output


# RMSE Function
def rmse(a, b):
    n = len(a)
    if n == len(b):
        return np.linalg.norm(a - b) / np.sqrt(n)
    else:
        print("Value Error!")


# Diagram between predicted time series and actual time series
def compared_diagram(predicted_series, actual_series, window_size,
                     is_matrix=True):
    x_axis = np.arange(0, len(predicted_series)) / 3600 * INTERVAL_BY_SECOND
    plt.plot(x_axis, predicted_series)
    plt.plot(x_axis, actual_series)
    plt.xlabel('Hours')
    plt.ylabel('Requests number by interval')
    plt.xlim([0, 24])
    plt.title('Worldcup 98 data: Day 10')
    plt.legend(
        ['Predicted time series ( window size = ' + str(window_size) + ')',
         'Actual time series'], loc='upper left')
    if is_matrix:
        plt.savefig('figure/matrix/ws' + str(window_size) + '.png')
    else:
        plt.savefig('figure/ws' + str(window_size) + '.png')


def compared_shift_diagram(predicted_series, actual_series, shift_index):
    x_axis = np.arange(0, len(predicted_series)) / 3600 * INTERVAL_BY_SECOND
    plt.plot(x_axis, predicted_series)
    plt.plot(x_axis, actual_series)
    plt.xlabel('Hours')
    plt.ylabel('Requests number by interval')
    plt.xlim([0, 24])
    plt.title('Worldcup 98 data: Day 10')
    plt.legend(
        ['Predicted time series ( shift_index = ' + str(shift_index) + ')',
         'Actual time series'], loc='upper left')
    if shift_index < 10:
        plt.savefig('figure/shift-' + '0' + str(shift_index) + '.png')
    else:
        plt.savefig('figure/shift-' + str(shift_index) + '.png')
