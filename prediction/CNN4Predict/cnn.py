from __future__ import print_function, division

import numpy as np
from utils import get_data, compared_diagram
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

# Tham so lien quan den neural network
NUMBER_TIME_SERIES = 1
NUMBER_OUTPUTS = 1
NUMBER_FEATURE_MAPS = 4
WINDOW_SIZE = 30
NUMBER_NEURAL_PER_LAYER = 5
BATCH_SIZE = 10
NUMBER_EPOCH = 200

# Tham so lien quan den du lieu dau vao
INTERVAL_BY_SECOND = 600
PEAK_PERCENT = 99
START_DAY = 6
END_DAY = 10

# Tham so du lieu K-shift
SHIFT_INDEX = 1


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
    # Su dung toi uu Adam de toi thieu MSE
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


# Tao input va output tu bo du lieu ban dau
def make_timeseries_instances(time_series, window_size):
    time_series = np.asarray(time_series)
    assert 0 < window_size < time_series.shape[0]
    x = np.atleast_3d(np.array([time_series[start:start + window_size]
                                for start in range(0, time_series.shape[0] - window_size)]))
    y = time_series[window_size+SHIFT_INDEX:]
    q = np.atleast_3d([time_series[-window_size:]])
    return x, y, q


# Ham train du lieu
def evaluate_timeseries(time_series, window_size):
    filter_length = NUMBER_NEURAL_PER_LAYER
    nb_filter = NUMBER_FEATURE_MAPS
    time_series = np.atleast_2d(time_series)
    if time_series.shape[0] == 1:
        time_series = time_series.T

    nb_samples, nb_series = time_series.shape
    model = neural_network(window_size=window_size,
                           filter_length=filter_length, nb_input_series=nb_series,
                           nb_outputs=nb_series, nb_filter=nb_filter)
    model.summary()

    x, y, q = make_timeseries_instances(time_series, window_size)
    test_size = int(0.2 * nb_samples)
    x_train, x_test, y_train, y_test = x[:-test_size], x[-test_size:], y[:-test_size], y[-test_size:]
    model.fit(x_train, y_train, nb_epoch=NUMBER_EPOCH, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
    predicted_time_series = []
    pred = model.predict(x_test)
    print('\n\nactual', 'predicted', sep='\t')
    for actual, predicted in zip(y_test, pred.squeeze()):
        print(actual.squeeze(), predicted, sep='\t')
        predicted_time_series.append(predicted)
    print('next', model.predict(q).squeeze(), sep='\t')
    return predicted_time_series


def main():
    # Khai bao cac tham so trong CNN
    np.set_printoptions(threshold=25)

    # Khai bao du lieu
    time_series = get_data(START_DAY, END_DAY, INTERVAL_BY_SECOND)
    predicted_time_series = evaluate_timeseries(time_series, WINDOW_SIZE)
    actual_time_series = get_data(END_DAY, END_DAY, INTERVAL_BY_SECOND)

    # Ghi gia tri RMSE ra file (CNN)
    rmse = np.sqrt(mean_squared_error(predicted_time_series, actual_time_series))
    f = open("evaluate_result/mse"+str(WINDOW_SIZE)+".txt", "w")
    f.write("Window size "+str(WINDOW_SIZE)+" : "+str(rmse)+"\n")
    f.close()

    # Bieu do
    compared_diagram(predicted_time_series, actual_time_series, WINDOW_SIZE)


if __name__ == '__main__':
    main()
