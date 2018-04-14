import csv
import numpy as np
from skfuzzy import gaussmf
from copy import deepcopy
import random
import matplotlib.pyplot as plt
DEFAULT_MEAN = 10000
DEFAULT_SIGMA = 10000
DEFAULT_NUMBERMF_PER_NODE = 5
WINDOW_SIZE = 20
# bo du lieu de test
X = []
Y = []


def loss_function(x, y):
    temp = 0
    for i in np.arange(x.shape[0]):
        temp += 0.5*((x[i]-y[i])**2)
    return np.sqrt(temp/x.shape[0])


with open('w20.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        int_row = [int(x) for x in row]
        X.append(int_row[:-1])
        Y.append(int_row[-1])
X = np.asarray(X)
Y = np.asarray(Y)
# kich thuoc cua so cung chinh la so input dau vao
window_size = X.shape[1]


# Xay dung layer 1:
# Cau truc layer 1:
# - So luong membershipfunction trong 1 nut = k ( thuong mac dinh se la 2)
# - Loai membership function trong 1 nut: Gauss, Bell, Sigmoid
# - Ma tran bieu dien tham so ham: kxl: l la so tham so co trong loai membershipfunction
# - De thuan tien trong viec bieu dien tham so ham, kien truc se su dung dong nhat 1 membershipfunction
class FirstLayer:
    def __init__(self, input_data, numbmf_per_node):
        self.data = input_data
        self.numbmf_per_node = numbmf_per_node
        self.train_size = input_data.shape[0]
        self.window_size = input_data.shape[1]
        self.premise_parameter = np.ones((self.data.shape[1], self.numbmf_per_node, 2), dtype=float)
        for i in np.arange(self.data.shape[1]):
            for j in np.arange(self.numbmf_per_node):
                self.premise_parameter[i][j][0] = random.uniform(5000, 15000)
                self.premise_parameter[i][j][1] = random.uniform(4000, 10000)

    def output(self):
        temp_output = np.zeros((self.train_size, self.window_size, self.numbmf_per_node), dtype=float)
        p = self.premise_parameter
        for i in np.arange(self.train_size):
            for j in np.arange(self.window_size):
                for k in np.arange(self.numbmf_per_node):
                    temp_output[i][j][k] = gaussmf(self.data[i][j],
                                                   p[j][k][0], p[j][k][1])
        return temp_output


def second_layer(output_1st_layer):
    train_size = output_1st_layer.shape[0]
    layer_size = output_1st_layer.shape[2]
    temp_output = np.zeros((train_size, layer_size), dtype=float)
    for i in np.arange(train_size):
        for j in np.arange(layer_size):
            temp_output[i][j] = 1
            for k in np.arange(output_1st_layer.shape[1]):
                temp_output[i][j] *= output_1st_layer[i][k][j]
    return temp_output


def third_layer(output_2nd_layer):
    train_size = output_2nd_layer.shape[0]
    layer_size = output_2nd_layer.shape[1]
    temp_output = np.zeros((train_size, layer_size), dtype=float)
    for i in np.arange(train_size):
        total = 0.0
        for j in np.arange(layer_size):
            total += output_2nd_layer[i][j]
        for j in np.arange(layer_size):
            temp_output[i][j] = output_2nd_layer[i][j] / total

    return temp_output


class FourthLayer:
    def __init__(self, numbermf_per_node):
        self.consequent_parameter = np.ones((numbermf_per_node, WINDOW_SIZE + 1), dtype=float) / 100
        self.node_number = numbermf_per_node

    def output(self, input_data, output_3rd_layer):
        train_size = output_3rd_layer.shape[0]
        p = self.consequent_parameter
        temp_output = np.zeros((train_size, self.node_number), dtype=float)
        for i in np.arange(train_size):
            for j in np.arange(self.node_number):
                for k in np.arange(WINDOW_SIZE):
                    temp_output[i][j] += p[j][k]*input_data[i][k]*output_3rd_layer[i][j]
            temp_output[i][j] += p[j][self.node_number]
        return temp_output


def fifth_layer(output_4th_layer):
    train_size = output_4th_layer.shape[0]
    temp_output = np.zeros(train_size, dtype=float)
    for i in np.arange(train_size):
        for k in np.arange(output_4th_layer.shape[1]):
            temp_output[i] += output_4th_layer[i][k]
    return temp_output


class AnfisModel:
    def __init__(self, input_data, output_data, numbmf_per_node):
        self.X = input_data
        self.Y = output_data
        self.train_size = self.X.shape[0]
        self.numb_per_node = numbmf_per_node
        self.first_layer = FirstLayer(X, DEFAULT_NUMBERMF_PER_NODE)
        self.fourth_layer = FourthLayer(self.numb_per_node)
        self.premise_parameter = deepcopy(self.first_layer.premise_parameter)
        self.consequent_parameter = deepcopy(self.fourth_layer.consequent_parameter)

    def predict(self, ):
        output_1st_layer = self.first_layer.output()
        output_2nd_layer = second_layer(output_1st_layer)
        output_3rd_layer = third_layer(output_2nd_layer)
        output_4th_layer = self.fourth_layer.output(X, output_3rd_layer)
        output_5th_layer = fifth_layer(output_4th_layer)

        return output_5th_layer

    def second_output(self):
        return second_layer(self.first_layer.output())

    def first_output(self):
        return self.first_layer.output()

    def half_predict(self):
        output_1st_layer = self.first_layer.output()
        output_2nd_layer = second_layer(output_1st_layer)
        output_3rd_layer = third_layer(output_2nd_layer)
        output_4th_layer = self.fourth_layer.output(X, output_3rd_layer)
        return output_4th_layer

    def backprogagation(self, eta = 0.85):
        for i in np.arange(10):
            combined = list(zip(self.X, self.Y))
            random.shuffle(combined)
            # feedfoward
            z = self.predict()
            # print loss after 1000 iterations
            loss = loss_function(Y, z)
            print("iter %d, rmse: %f" % (i, loss))
            f = self.half_predict()
            muy = self.second_output()
            a = self.first_output()
            total_muy = np.zeros(muy.shape[0], dtype=float)
            for i in np.arange(muy.shape[0]):
                for j in np.arange(muy.shape[1]):
                    total_muy[i] += muy[i][j]

            # backprogagation
            delta_mean = np.zeros((WINDOW_SIZE, self.numb_per_node), dtype=float)
            delta_sigma = np.zeros((WINDOW_SIZE, self.numb_per_node), dtype=float)
            for k in np.arange(self.train_size):
                for i in np.arange(WINDOW_SIZE):
                    for j in np.arange(self.numb_per_node):
                        delta_mean[i][j] = (z[k]-Y[k])*(f[k][j]-Y[k])*(X[k][i]-a[k][i][j]*X[k][i])*muy[k][j]\
                                            / (self.first_layer.premise_parameter[i][j][1]*X[k][i]*X[k][i]*total_muy[k])
                        delta_sigma[i][j] = (z[k]-Y[k])*(f[k][j]-Y[k])*((X[k][i]-a[k][i][j]*X[k][i])**2)*muy[k][j]\
                                            / (self.first_layer.premise_parameter[i][j][1]*(X[k][i]**3)*total_muy[k])
                for i in np.arange(WINDOW_SIZE):
                    for j in np.arange(self.numb_per_node):
                        self.first_layer.premise_parameter[i][j][0] -= eta*delta_mean[i][j]
                        self.first_layer.premise_parameter[i][j][1] -= eta*delta_sigma[i][j]
        return self.first_layer.premise_parameter

    def rlse(self):
        pass

    def train_hybrid_jang(self):
        self.backprogagation(eta=0.85)

    def evaluate(self):
        pass

demo = AnfisModel(X, Y, 4)
demo.train_hybrid_jang()

z = demo.predict()
for i in np.arange(z.shape[0]):
    print("Predict: " + str(z[i]) + " . Actual: " + str(Y[i]))

x_axis = np.arange(700)
plt.plot(x_axis, z)
plt.plot(x_axis, Y)
plt.xlabel('')
plt.ylabel('Requests number by interval')
plt.xlim([0, 700])
plt.title('Worldcup 98 data')
plt.legend(['Predicted time series ', 'Actual time series'], loc='upper left')
plt.savefig('compare.png')
plt.show()

