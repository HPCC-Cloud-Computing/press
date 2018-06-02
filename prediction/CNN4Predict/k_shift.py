# K-shift Value Algorithm
from utils import get_data
from sklearn.metrics import mean_squared_error
import numpy as np
from utils import compared_shift_diagram

START_DAY = 6
END_DAY = 10
SHIFT_INDEX = 1


class ShiftPredict(object):
    def __init__(self, time_series=None, shift_index=1):
        self.time_series = time_series
        self.shift_index = shift_index

    def predicted_series(self, start=1, length=1):
        return self.time_series[
               -self.shift_index + start:-self.shift_index + start + length]


a = get_data(START_DAY, END_DAY, 600)
b = ShiftPredict(a, SHIFT_INDEX)
k_shift_time_series = b.predicted_series(720 - 144, 144)
rmse_k_shift = np.sqrt(
    mean_squared_error(k_shift_time_series, a[720 - 144:720]))
if SHIFT_INDEX < 10:
    f = open("evaluate_result/shift-" + '0' + str(SHIFT_INDEX) + ".txt", "w")
else:
    f = open("evaluate_result/shift-" + str(SHIFT_INDEX) + ".txt", "w")
f.write("Shift index " + str(SHIFT_INDEX) + " : " + str(rmse_k_shift) + "\n")
f.close()
compared_shift_diagram(k_shift_time_series, a[720 - 144:720], SHIFT_INDEX)
