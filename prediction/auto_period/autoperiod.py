import numpy as np
import matplotlib.pyplot as plt
import math

SEQUENCE = 0
PERIODGRAM = 1
ACF = 2

DEFAULT_INTERVAL_VALUE = 600
DEFAULT_PEAK_PERCENT = 99


class Autoperiod(object):

    def __init__(self, time_series=[], interval=DEFAULT_INTERVAL_VALUE,
                 peak_percent=DEFAULT_PEAK_PERCENT):

        self.time_series = time_series
        self.interval = interval
        self.peak_percent = peak_percent

    def fft(self):
        a = self.time_series
        return np.fft.rfft(a)

    def length_series(self):
        return len(self.time_series)

    def get_ESD(self):
        a = self.fft()
        return np.abs(a ** 2)

    def ACF_list(self):
        s = []
        for i in range(self.length_series()):
            s.append(self.get_ACF(i))
        return s

    # In ra bieu do cua tung dang cu the
    def diagram(self, n):

        if n == PERIODGRAM:
            x_axis = np.arange(3, len(self.get_ESD())) / float(
                2 * len(self.get_ESD()))
            a = self.get_ESD()[3:]
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            plt.title('Periodgram')
            plt.plot(x_axis, a)
            plt.show()

        if n == SEQUENCE:
            x_axis = np.arange(0, self.length_series()) / 3600 * self.interval
            plt.plot(x_axis, self.time_series)
            plt.xlabel('Hours')
            plt.ylabel('Requests number by interval')
            plt.title('Sequence diagram')
            plt.show()

        if n == ACF:
            plt.xlabel('Tau')
            plt.ylabel('ACF_value')
            plt.title('Circular Autocorrection')
            plt.plot(self.ACF_list())
            plt.show()

    # Lay ACF theo tung phan tu
    def get_ACF(self, tau):
        tempACF = 0
        for i in range(self.length_series()):
            if i + tau == self.length_series():
                break
            tempACF += self.time_series[tau] * self.time_series[i + tau]
        return float(tempACF) / float(self.length_series())

    # Lay cac chu ki trien vong dua vao kiem tra
    def period_hints(self):
        threshold = math.floor(
            self.length_series() * (100 - self.peak_percent) / 100.0)
        # Lay danh sach cac phan tu co nang luong lon nhat

        period_temp_list = []
        index_hint_list = []
        temp_ESD = []

        for t in np.arange(len(self.get_ESD())):
            temp_ESD.append(self.get_ESD()[t])
        hints_list = sorted(temp_ESD, reverse=True)[:threshold]
        # Chuyen cac phan tu sang dang chi so va loc cac phan tu
        for element in hints_list:

            hint_index = temp_ESD.index(element)
            # Bo qua cac phan khong co chu ki ( k=0 ) va phan lay chu ki la so phan tu cua mau ( k=1 )
            if (hint_index < 2):
                continue
            else:
                # Kiem tra dieu kien ACF
                check_acf = (self.ACF_list()[hint_index]
                             >= (self.ACF_list()[hint_index - 1]
                                 + self.ACF_list()[hint_index + 1]) / 2)

                if check_acf:
                    index_hint_list.append(hint_index)
                    temp_period_element = math.floor(
                        float(self.length_series()) / hint_index)
                    period_temp_list.append(temp_period_element)
        return period_temp_list
