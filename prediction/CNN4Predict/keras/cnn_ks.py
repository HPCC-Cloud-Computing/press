#!/usr/bin/env python
"""
Example of using Keras to implement a 1D convolutional neural network (CNN) for timeseries prediction.
"""

from __future__ import print_function, division

import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential

NUMBER_TIME_SERIES = 1
NUMBER_OUTPUTS = 1
NUMBER_FEATURE_MAPS = 4


def neural_network(window_size, filter_length,
                   nb_input_series=NUMBER_TIME_SERIES,
                   nb_outputs=NUMBER_OUTPUTS, nb_filter=NUMBER_FEATURE_MAPS):
    model = Sequential((
        # Convolution Layer 1 su
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length,
                      activation='relu',
                      input_shape=(window_size, nb_input_series)),
        MaxPooling1D(),
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length,
                      activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='linear'),
        # For binary classification, change the activation to 'sigmoid'
    ))

    # Su dung toi uu Adam de toi thieu MSE
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def make_timeseries_instances(time_series, window_size):
    time_series = np.asarray(time_series)
    assert 0 < window_size < time_series.shape[0]
    X = np.atleast_3d(np.array([time_series[start:start + window_size]
                                for start in
                                range(0, time_series.shape[0] - window_size)]))
    y = time_series[window_size:]
    q = np.atleast_3d([time_series[-window_size:]])
    return X, y, q


def evaluate_timeseries(time_series, window_size):
    filter_length = 5
    nb_filter = 4
    time_series = np.atleast_2d(time_series)
    if time_series.shape[0] == 1:
        time_series = time_series.T  # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = time_series.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples,
                                                               nb_series),
          time_series)
    model = neural_network(window_size=window_size, filter_length=filter_length,
                           nb_input_series=nb_series, nb_outputs=nb_series,
                           nb_filter=nb_filter)
    print(
        '\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(
            model.input_shape, model.output_shape, nb_filter, filter_length))
    model.summary()

    X, y, q = make_timeseries_instances(time_series, window_size)
    print('\n\nInput features:', X, '\n\nOutput labels:', y,
          '\n\nQuery vector:', q, sep='\n')
    test_size = int(
        0.01 * nb_samples)  # In real life you'd want to use 0.2 - 0.5
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[
                                                                       :-test_size], y[
                                                                                     -test_size:]
    model.fit(X_train, y_train, nb_epoch=25, batch_size=2,
              validation_data=(X_test, y_test))

    pred = model.predict(X_test)
    print('\n\nactual', 'predicted', sep='\t')
    for actual, predicted in zip(y_test, pred.squeeze()):
        print(actual.squeeze(), predicted, sep='\t')
    print('next', model.predict(q).squeeze(), sep='\t')


def main():
    """Prepare input data, build model, evaluate."""
    np.set_printoptions(threshold=25)
    ts_length = 1000
    window_size = 50

    print('\nSimple single timeseries vector prediction')
    timeseries = np.sin(np.arange(ts_length))
    evaluate_timeseries(timeseries, window_size)


if __name__ == '__main__':
    main()
