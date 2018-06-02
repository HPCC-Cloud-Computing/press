import numpy as np


# Test
class GAF(object):
    def __init__(self, time_series):
        self.time_series = time_series

    # Give length time series
    def length_time_series(self):
        return len(self.time_series)

    # Piecewise Aggregation Approximation
    def paa(self, paa_size):
        any_series = self.time_series
        length = len(any_series)
        if length == paa_size:
            return any_series
        elif length < paa_size:
            print("Error!")
            pass
        else:
            paa_series = np.zeros(paa_size)
            for i in np.arange(paa_size * length):
                index_paa = int(i / length)
                index_series = int(i / paa_size)
                paa_series[index_paa] += any_series[index_series]
            for i in np.arange(paa_size):
                paa_series[i] /= length
            return paa_series

    # Normalizing time series
    def normalize_time_series(self, reduced_length=length_time_series):
        max_time_series = np.amax(self.time_series)
        min_time_series = np.amin(self.time_series)
        normalize_series = []
        for i in range(len(self.time_series)):
            x = self.time_series[i]
            temp_element = ((x - max_time_series) + (x - min_time_series)) / \
                           (max_time_series - min_time_series)
            normalize_series.append(temp_element)
        return normalize_series

    # Gramian angular field
    def gaf(self):
        n = self.length_time_series()
        initial_matrix = np.zeros((n, n))
        x_ = self.normalize_time_series()
        for i in np.arange(n):
            for j in np.arange(n):
                initial_matrix[i][j] = x_[i] * x_[j] - np.sqrt(
                    1 - x_[i] * x_[i]) * np.sqrt(1 - x_[j] * x_[j])
        return initial_matrix.reshape(n, n, 1)
