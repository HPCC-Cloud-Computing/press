import numpy as np


class StatePredictor(object):
    def __init__(self, time_series, M, max_value, min_value=0.0):
        self.time_series = time_series
        self.M = M
        self.max_value = max_value
        self.min_value = min_value
        # Độ rộng 1 bin
        self.width = (max_value - min_value) / M

    # Xác định state của giá trị
    def state(self, value):
        return int(np.floor((value - self.min_value) / self.width))

    # Hàm xây dựng ma trận chuyển trạng thái
    # Input: 
    #       -chuỗi thời gian
    #       -M: số trạng thái
    #       -max_value: giá trị lớn nhất của tài nguyên
    # Output:
    #       -Ma trận chuyển trạng thái p, với mỗi phần tử p[i][j] là xác suất chuyển từ trạng thái i sang j
    #       (p có kích thước M * M)
    def state_transition_matrix(self):
        # Xây dựng ma trận chuyển trạng thái
        p = np.zeros((self.M, self.M))
        s = [self.state(value) for value in self.time_series]
        for i in range(len(s) - 1):
            p[s[i]][s[i + 1]] += 1
        for i in range(M):
            s = sum(p[i])
            if s != 0:
                p[i] = p[i] / sum(p[i])
            else:
                p[i] = p[i] / M
        return np.matrix(p)

    # Dự báo trạng thái
    # Input:
    #       - current_value: giá trị hiện tại (số nguyên 0 -> (M-1))
    #       - t: thời điểm cần dự báo
    #       - tp: ma trận chuyển trạng thái
    # Output:
    #       - Trạng thái t thời điểm tiếp theo
    def predict(self, t, current_value):
        p = self.state_transition_matrix()
        current_state = self.state(current_value)
        predict_value = []
        pt = np.identity(self.M)
        for i in range(t):
            pt = pt * p
            # Xác suất trạng thái
            prob = pt[current_state]
            # print(prob)
            predict_state = prob.argmax()
            # print(predict_state)
            # Giá trị dự báo là giá trị lớn nhất của trạng thái
            predict_value.append(
                (predict_state + 1) * self.width + self.min_value)
        print(predict_value)

        return predict_value
