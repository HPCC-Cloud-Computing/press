import numpy as np
import skfuzzy as fuzz
from skfuzzy import gaussmf, gbellmf, sigmf
import random 
import csv
import matplotlib.pyplot as plt
func_dict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}

# Khung chua tham so cho mo hinh ANFIS
# Hien tai moi chi ho tro 3 loai ham la gauss, gbell va sigmoi
def frame_parameter(mf: str, rule_number: int, window_size: int): # premise parameters
    x = [[['gaussmf', {'mean': random.uniform(15000, 20000), 'sigma': random.uniform(10000, 12500)}]  for i in np.arange(window_size)] for j in np.arange(rule_number)]
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
    s = sum(osl)
    length = osl.shape[0]
    temp = [osl[i]/s for i in np.arange(length)]
    return np.asarray(temp)

# Dau ra cua lop thu 4
def fouth_layer(otl: np.ndarray, x:np.ndarray, cp:np.ndarray):
    mat = np.append(x, 1)
    temp = [otl[i]*np.dot(mat,cp[i]) for i in np.arange(cp.shape[0])]
    return np.asarray(temp)

# Dau ra cuoi
def fifth_layer(ofl: np.ndarray):
    return sum(ofl) 


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
        
    # Phuong thuc du doan theo dau vao voi tham so trong mo hinh
    def output_single(self, x: np.ndarray):
        #layer1 = first_layer(x, self.p_para)
        #print(layer1)
        #layer2 = second_layer(layer1)
        #print(layer2)
        #layer3 = third_layer(layer2)
        #layer4 = fouth_layer(layer3, x, self.c_para)
        #layer5 = fifth_layer(layer4)
        hf = self.half_first(x)
        hl = self.half_last(hf, x)
        return hl
    
    # Su dung de tinh ra mot chuoi cac gia tri du doan tu 1 mang cho truoc
    # Su dung de tinh loss function va in ra man hinh
    def output(self, inp_value):
        return np.asarray([self.output_single(inp_value[i]) for i in np.arange(self.training_size)])
    
    def predict(self, x:np.ndarray):
        return np.asarray([self.output_single(x[i]) for i in np.arange(x.shape[0])])

    # loss_function su dung de lam ham muc tieu trong GD
    def loss_function(self):
        predict_value = self.output(self.X)
        actual_value = self.Y
        return ((predict_value - actual_value)**2).mean(axis=0)
    
    def fix_p_para(self):
        self.p_para[0][0][1]['mean'] += 10000

    def lse(self):
        y_ = np.array(self.Y)[np.newaxis].T
        a = np.ones((self.training_size, (self.window_size+1) * self.rule_number), dtype=float)
        w = np.asarray([self.half_first(self.X[i]) for i in np.arange(self.training_size)])
        for i in np.arange(self.training_size):
            for j in np.arange(self.rule_number):
                for k in np.arange(self.window_size):
                    a[i][j*(self.window_size+1)+k] = w[i][j]*self.X[i][k]
                a[i][j*(self.window_size+1)+self.window_size] = w[i][j]
        #temp_c_para = np.array(self.c_para.ravel())[np.newaxis].T
        c = np.dot(np.linalg.pinv(a), y_)
        self.c_para = np.reshape(c, self.c_para.shape)
        return 
    
    # Su dung GD
    def gd(self, epochs=1, eta=0.9, k=0.01, max_loop=1000):
        pass

    # Su dung giai thuat hon hop
    def HybridTraining(self):
        pass

    # Phuong thuc xuat hinh
    def figure(self):
        pass

## Kich ban test thu nghiem
data = np.genfromtxt('w20.csv', delimiter=',')
x = data[:-1-140,:-1]
y = data[:-1-140,-1]
#y = np.asarray([31231., 13212., 41230, 45730, 57320, 12365], dtype=np.float64)
x_test = data[-1-140+1:-1,:-1]
y_test = data[-1-140+1:-1,-1]
a = ANFIS(x, y, 'gauss', 5)
#print(a.half_first([12331, 1231]))
a.lse()
print(np.sqrt(loss_function(a.predict(x_test), y_test)))
x_axis = np.arange(1, 140, 1)
pred = plt.plot(x_axis, a.predict(x_test), label='predict')
act = plt.plot(x_axis, y_test, label='actual')
plt.legend()
plt.show()
