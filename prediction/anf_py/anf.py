import numpy as np
import skfuzzy as fuzz
from skfuzzy import gaussmf, gbellmf, sigmf
import random
import csv
import matplotlib.pyplot as plt
func_dict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}

# Khung chua tham so cho mo hinh ANFIS
# Hien tai moi chi ho tro 3 loai ham la gauss, gbell va sigmoi
def frame_parameter(mf: str, rule_number: int, window_size: int, mean1=20000, mean2=25000, sigma1=10000, sigma2=15000): # premise parameters
    x = [[['gaussmf', {'mean': random.uniform(20.0, 25.0), 'sigma': random.uniform(15.0, 20.0)}] for i in np.arange(window_size)] for j in np.arange(rule_number)]
    return np.asarray(x)

def consequence_parameter(rule_number: int, window_size: int):
    return np.ones((rule_number, window_size+1), dtype=float)

# Dau ra cua lop dau tien
# Lop thuc hien tinh toan do mo qua cac tap mo cho truoc
def first_layer(x:np.ndarray, fp: np.ndarray):
    ws, rn = fp.shape[1], fp.shape[0]
    temp =  [[func_dict[fp[i][j][0]](x[j], **fp[i][j][1]) for j in np.arange(ws)]
                for i in np.arange(rn)]
    return np.asarray(temp)

def loss_function(x, y):
    return ((x - y)**2).mean(axis=0)

# Dau ra cua lop thu 2
# Lop thuc hien tinh toan cac luat tu cac tap mo
def second_layer(ofl: np.ndarray):
    ws, rn = ofl.shape[1], ofl.shape[0]
    temp = np.ones(rn, dtype=float)
    for i in np.arange(rn):
        for j in np.arange(ws):
            temp[i] *= ofl[i][j]
    return temp

# Dau ra cua lop thu 3
def third_layer(osl: np.ndarray):
    #s = sum(osl)
    #length = osl.shape[0]
    #temp = [osl[i]/s for i in np.arange(length)]
    return osl / osl.sum()

# Dau ra cua lop thu 4
def fouth_layer(otl: np.ndarray, x:np.ndarray, cp:np.ndarray):
    mat = np.append(x, 1)
    temp = [otl[i]*np.dot(mat,cp[i]) for i in np.arange(cp.shape[0])]
    return np.asarray(temp)

# Dau ra cuoi
def fifth_layer(ofl: np.ndarray):
    return sum(ofl)


# Dao ham ham loi
def er_function(x, y):
    pass

# Lop chua mo hinh ANFIS
class ANFIS:

    def __init__(self, X: np.ndarray, Y: np.ndarray, mf: str, rule_number: int):
        self.X = X # Training_input
        self.Y = Y # Training_output
        self.mf = mf
        self.training_size = X.shape[0]
        self.rule_number = rule_number # So luat trong mang ANFIS
        if (X.shape[0]!=Y.shape[0]):
            print('Size error, check training i/oput')
            exit(0)
        try:
            self.window_size = X.shape[1]
        except IndexError as err:
            print('Training input must be 3-d array: ', err)
            exit(0)
        self.p_para = frame_parameter(self.mf, self.rule_number, self.window_size)
        self.c_para = consequence_parameter(self.rule_number, self.window_size)

    def summary(self): # Tong hop mo hinh mang ANFIS
        print('Training size: ', self.X.shape[0])
        print('Rule number  : ', self.rule_number)

    def half_first(self, x:np.ndarray):
        layer1 = first_layer(x, self.p_para)
        layer2 = second_layer(layer1)
        return third_layer(layer2)

    def half_last(self, hf, x):
        layer4 = fouth_layer(hf, x, self.c_para)
        return fifth_layer(layer4)


    def f_single(self, x):
        hf = self.half_first(x)
        wf = fouth_layer(hf, x, self.c_para)
        return wf

    def f_(self, x: np.ndarray):
        return np.asarray([self.f_single(x[i]) for i in np.arange(self.training_size)])

    def w_(self, x: np.ndarray):
        return np.asarray([self.half_first(x[i]) for i in np.arange(self.training_size)])

    # Phuong thuc du doan theo dau vao voi tham so trong mo hinh
    def output_single(self, x: np.ndarray):
        hf = self.half_first(x)
        hl = self.half_last(hf, x)
        return hl

    # Su dung de tinh ra mot chuoi cac gia tri du doan tu 1 mang cho truoc
    # Su dung de tinh loss function va in ra man hinh
    def output(self, inp_value):
        return np.asarray([self.output_single(inp_value[i]) for i in np.arange(self.training_size)])

    # Dung cho tap test
    def predict(self, x:np.ndarray):
        return np.asarray([self.output_single(x[i]) for i in np.arange(x.shape[0])])

    # loss_function su dung de lam ham muc tieu trong GD
    def lossFunction(self):
        predict_value = self.output(self.X)
        actual_value = self.Y
        return ((predict_value - actual_value)**2).mean(axis=0)

    def fix_p_para(self, mean1, mean2, sigma1, sigma2):
        self.p_para = frame_parameter(self.mf, self.rule_number, self.window_size, mean1, mean2, sigma1, sigma2)

    def lse(self):
        #print(self.lossFunction())
        # Khai bao
        y_ = np.array(self.Y)[np.newaxis].T
        a = np.ones((self.training_size, (self.window_size+1) * self.rule_number), dtype=float)
        w = np.asarray([self.half_first(self.X[i]) for i in np.arange(self.training_size)])
        # Bat dau tien hanh linear regression
        for i in np.arange(self.training_size):
            for j in np.arange(self.rule_number):
                for k in np.arange(self.window_size):
                    a[i][j*(self.window_size+1)+k] = w[i][j]*self.X[i][k]
                a[i][j*(self.window_size+1)+self.window_size] = w[i][j]
        c = np.dot(np.linalg.pinv(a), y_)
        self.c_para = np.reshape(c, self.c_para.shape)
        #print(self.lossFunction())
        return

    # Dao ham ham loi ( lay ham Gauss), tien hanh dao ham cho tat ca cac truong hop
    def derivError(self, mf='gauss', variable='mean'):
        temp = np.zeros(self.p_para.shape)
        d = self.predict(self.X)
        y = self.Y
        x = self.X
        f = self.f_(self.X)
        w = self.w_(self.X)
        for k in np.arange(self.training_size):
            for i in np.arange(self.window_size):
                for j in np.arange(self.rule_number):
                    #sigma
                    temp[j][i][0] += (y[k] - d[k]) * (d[k] - f[k][j]) * w[k][j] * ((x[k][i] - self.p_para[j][i][1]['sigma'])) / (self.p_para[j][i][1]['mean']**2)
                    # mean
                    temp[j][i][1] += (y[k] - d[k]) * (d[k] - f[k][j]) * w[k][j] * ((x[k][i] - self.p_para[j][i][1]['sigma'])**2) / (self.p_para[j][i][1]['mean']**3)
        #print('done')
        return temp

    def deE(self):
        pass

    # Su dung GD
    def gd(self, epochs=1, eta=0.01, k=0.95, max_loop=10):
        loop = 1
        ll = 1
        #print('loop: ', np.sqrt(self.lossFunction()))
        derivE = self.derivError('gauss','mean')
        #  Xu ly voi cac tham so mean
        for i in np.arange(self.rule_number):
            for j in np.arange(self.window_size):
                self.p_para[i][j][1]['mean'] -= eta*derivE[i][j][0]
                self.p_para[i][j][1]['sigma'] -= eta*derivE[i][j][1]

    # Su dung giai thuat hon hop
    def hybridTraining(self, max_loop=50):
        loop = 1
        while ( loop < max_loop):
            self.lse()
            a = np.sqrt(self.lossFunction())
            self.gd()
            print('Loop: ', loop, '\tLSE RMSE: ', a , '\tGD RMSE: ', np.sqrt(self.lossFunction()))
            loop += 1

# Su dung bo du lieu WC
#data = np.genfromtxt('w20.csv', delimiter=',')
#x = data[:-1-140,:-1]
#y = data[:-1-140,-1]
#a = ANFIS(x, y, 'gauss', 2)
#x_test = data[-1-140+1:-1,:-1]
#y_test = data[-1-140+1:-1,-1]
#print(x.shape)
#a.hybridTraining()
#print(np.sqrt(loss_function(a.predict(x_test), y_test)))
#x_axis = np.arange(1, 140, 1)
#pred = plt.plot(x_axis, a.predict(x_test), label='predict')
#act = plt.plot(x_axis, y_test, label='actual')
#plt.legend()
#plt.show()
