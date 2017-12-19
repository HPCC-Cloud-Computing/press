import numpy as np

test_series = np.asarray([2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34])


def paa(any_series, paa_size):
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


print(test_series)
print(paa(test_series, 9))
