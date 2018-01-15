from __future__ import print_function, division

import csv
import numpy as np
from utils import get_data, compared_diagram
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

# Tham so lien quan den neural network
NUMBER_TIME_SERIES = 1
NUMBER_OUTPUTS = 1
NUMBER_FEATURE_MAPS = 4
WINDOW_SIZE = 10
NUMBER_NEURAL_PER_LAYER = 2
BATCH_SIZE = 10
NUMBER_EPOCH = 200

# Tham so lien quan den du lieu dau vao
INTERVAL_BY_SECOND = 600
PEAK_PERCENT = 99
START_DAY = 6
END_DAY = 10


# Thiet lap neural network
def neural_network(window_size, filter_length, nb_input_series=NUMBER_TIME_SERIES,
                   nb_outputs=NUMBER_OUTPUTS, nb_filter=NUMBER_FEATURE_MAPS):
    model = Sequential((
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length,
                      activation='relu', input_shape=(window_size, nb_input_series)),
        MaxPooling1D(),

        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
        MaxPooling1D(),

        Flatten(),
        Dense(nb_outputs, activation='linear'),
    ))
    print(window_size, nb_input_series)
    # Su dung toi uu Adam de toi thieu MSE
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


# Tao input va output tu bo du lieu ban dau
def make_timeseries_instances():
    x = []
    y = []
    with open('day6_10.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            int_row = [int(x) for x in row]
            x.append(int_row[:-1])
            y.append(int_row[-1])
    x = np.atleast_3d(x)
    q = np.atleast_3d(y)
    print(np.shape(x))
    return x, y, q


# Ham train du lieu
def evaluate_timeseries(window_size):
    filter_length = NUMBER_NEURAL_PER_LAYER
    nb_filter = NUMBER_FEATURE_MAPS
    model = neural_network(window_size=window_size,
                           filter_length=filter_length, nb_input_series=1,
                           nb_outputs=1, nb_filter=nb_filter)
    model.summary()
    x, y, q= make_timeseries_instances()
    test_size = int(0.2 * len(y))
    x_train, x_test, y_train, y_test = x[:-test_size], x[-test_size:], y[:-test_size], y[-test_size:]
    model.fit(x_train, y_train, nb_epoch=NUMBER_EPOCH, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
    pred = model.predict(x_test)

def main():
    # Khai bao cac tham so trong CNN
    np.set_printoptions(threshold=25)
    evaluate_timeseries(10)
    # Bieu do
    # compared_diagram(predicted_time_series, actual_time_series, WINDOW_SIZE)


if __name__ == '__main__':
    main()
