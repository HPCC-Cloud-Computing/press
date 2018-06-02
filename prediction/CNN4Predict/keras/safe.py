import numpy as np

"""Prepare input data, build model, evaluate."""
np.set_printoptions(threshold=25)
ts_length = 10
window_size = 50

print('\nSimple single timeseries vector prediction')
timeseries = np.arange(ts_length)  # The timeseries f(t) = t

print('\nMultiple-input, multiple-output prediction')
timeseries = np.array([np.arange(ts_length),
                       -np.arange(ts_length)])  # The timeseries f(t) = [t, -t]
print(timeseries)
