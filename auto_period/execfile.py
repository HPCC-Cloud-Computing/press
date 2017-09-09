from datetime import datetime
import codecs


def month_converter(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month) + 1


class Execute_line(object):
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


class Count_request(object):
    def __init__(self, file, interval=60):
        self.file = file
        self.interval = interval
    def request_list(self):
        min = 0
        max = 0
        countLine = 0
        lRequest = []
        with codecs.open(self.file, "r", encoding='utf-8', errors='ignore') as f:
            for line in f:
                timeElement = Execute_line(line)
    ##            print(timeElement.convertTime(), int((timeElement.convertTime() - min) / self.interval))
                if timeElement.convert_time() == 0:
                    continue
                if countLine == 0:
                    min = timeElement.convert_time()
                    countLine += 1

                else:
                    max = timeElement.convert_time()
                    countLine += 1
                if (int(timeElement.convert_time() - min)) / int(self.interval) >= len(lRequest):
                    lRequest.append(1)
                else:
                    lRequest[int(int(timeElement.convert_time() - min) / int(self.interval)) - 1] += 1
            if lRequest[-1] <= 1:
                lRequest.pop(-1)
        return lRequest

    def output(self, output_file):
        with codecs.open(output_file, "wb+", encoding='utf-8', errors='ignore') as f:
            for element in self.request_list():
                f.write(str(element)+'\n')

def executed_list(target_file):
    temp_list = []
    with codecs.open(target_file, "r", encoding='utf-8', errors='ignore') as f:
        for e in f:
            temp_list.append(int(e))
    return temp_list

# if __name__ == '__main__':
#     t = Count_request('data_folder/wc_day8_1.out', 60)
#     t.output('output_folder/wc_day8_1_'+str(t.interval)+'.out')
def output_result(list, interval):
    with codecs.open('result/interval'+str(interval)+'/result.txt', "wb+", encoding='utf-8', errors='ignore') as f:
            for element in list:
                f.write(str(element)+'\n')