import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feedforward import FeedForwardNN

window_size = 4

df = pd.read_csv('data-10min_workload.csv', names=['request'])
data = df['request'].values

data_train = data[40 * 144:47 * 144]
data_test = data[46 * 144:48 * 144]

x_train = []
for i in range(len(data_train) - window_size):
    x_train.append(data_train[i:i + window_size])
x_train = np.array(x_train)

y_train = data_train[window_size:]

x_test = []
for i in range(len(data_test) - window_size):
    x_test.append(data_test[i:i + window_size])
x_test = np.array(x_test)

y_test = data_test[window_size:]

# Preproccessing
min_value = min(data_train)
max_value = max(data_train)
x_train = (x_train - min_value) / (max_value - min_value)
y_train = (y_train - min_value) / (max_value - min_value)
x_test = (x_test - min_value) / (max_value - min_value)

nn = FeedForwardNN(x_train, y_train, loss_func='mean_absolute_error')
nn.fit()
y_predict = nn.predict(x_test) * (max_value - min_value) + min_value
y_predict = np.array(list(map(int, y_predict)))

plt.figure()
plt.plot(y_test, label='Actual')
plt.plot(y_predict, 'r-', label='Predict')
plt.legend()
plt.show()
