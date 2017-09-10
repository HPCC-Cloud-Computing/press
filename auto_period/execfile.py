from datetime import datetime
import codecs
import math
import numpy as np
def month_converter(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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
            tempTime = self.detect_time().split()
            time_zone = tempTime[-1]

            hms = tempTime[0][tempTime[0].index(':') + 1:]
            hms = hms.split(':')
            dt = tempTime[0][:tempTime[0].index(':')].split('/')
            since = datetime(1970, 8, 15, 0, 0, 0)
            time_sample = datetime(int(dt[2]), month_converter(dt[1]), int(dt[0]), int(hms[0]), int(hms[1]),
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
        min = 0
        max = 0
        count_line = 0
        temp_request_list = []

        with codecs.open(self.file, "r", encoding='utf-8', errors='ignore') as f:
            for line in f:
                timeElement = ExecuteLine(line).convert_time()
                condition_append_list = math.floor((int(timeElement) - min) / self.interval) >= len(temp_request_list)
                if timeElement == 0:
                    continue
                if count_line == 0:
                    min = timeElement
                    count_line += 1
                if condition_append_list:
                    temp_request_list.append(1)
                else:
                    temp_request_list[-1] += 1
        return temp_request_list

    def output(self, output_file):
        with codecs.open(output_file, "wb+", encoding='utf-8', errors='ignore') as f:
            for element in self.request_list():
                f.write(str(element) + '\n')

def get_single_data(targeted_file):
    temp_list = []
    with codecs.open(targeted_file, "r", encoding='utf-8', errors='ignore') as f:
        for e in f:
            temp_list.append(int(e))
    return temp_list

def get_data( start, end, interval):
    s = np.arange( start, end + 1, 1)
    temp_output = []
    for index in s:

        targeted_file = 'output_folder/wc_day'+str(index)+'_1_'+str(interval)+'.out'
        temp_output += get_single_data(targeted_file)
    return temp_output