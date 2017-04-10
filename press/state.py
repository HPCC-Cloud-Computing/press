import numpy as np

# Hàm xây dựng ma trận chuyển trạng thái
# Input: 
#       -chuỗi thời gian
#       -M: số trạng thái
#       -max_value: giá trị lớn nhất của tài nguyên
# Output:
#       -Ma trận chuyển trạng thái p, với mỗi phần tử p[i][j] là xác suất chuyển từ trạng thái i sang j
#       (p có kích thước M * M)

def state_transition_matrix(time_series, M, max_value):
    # Độ rộng 1 bin
    width = max_value / M
    # Xác định state của giá trị
    def state(value):
        return int(np.floor(value / width))
    
    # Xây dựng ma trận chuyển trạng thái
    p = np.zeros((M, M))
    s = [state(value) for value in time_series]
    for i in range(len(s) - 1):
        p[s[i]][s[i+1]] += 1
    for i in range(M):
        p[i] = p[i] / sum(p[i])
        
    return np.matrix(p)


# Dự báo trạng thái
# Input:
#       - state: trạng thái hiện tại (số nguyên 0 -> (M-1))
#       - t: thời điểm cần dự báo
#       - tp: ma trận chuyển trạng thái
# Output:
#       - Trạng thái sau t thời điểm
def state_predict(current_state, t, p):
    pt = p ** t
    print(pt)
    # Tìm chỉ số phần tử có giá trị lớn nhất tại dòng state
    predict_state = pt[state].argmax()
    
    return predict_state