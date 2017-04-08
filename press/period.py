import numpy as np
from scipy.stats.stats import pearsonr

# Xác định giá trị dominant frequency
# Input: Chuỗi thời gian 
# Output: dominant frequency
def dominant_freq(time_series):
    n = len(time_series)
    fourier = np.fft.rfft(time_series)
    max_freq_component = max(fourier[1:], key=abs)
    for i in range(n):
        if fourier[i] == max_freq_component:
            index_max = i
            break
    dominant_freq = index_max / n
    return dominant_freq

# Xác định chữ kí của pattern
# Input: time series, dominant frequency, sampling rate
# Output: Dãy giá trị dự báo nếu phát hiện chu kì, ngược lại trả về None
def signature_predict(time_series, domin_freq, sampling_rate):
    z = int(sampling_rate / domin_freq)
    q = len(time_series) // z
    # Chia time series thành q pattern window bằng nhau
    p = []
    for i in range(q):
        p.append(time_series[i*z : (i+1)*z])
        
    # Xác định sự tương quan giữa 2 pattern window:
    # 2 pattern window tương quan nếu hệ số tương quan >= 0.85 và tỉ số giá trị trung bình >= 0.95
    for i in range(q):
        for j in range(i+1, q):
            relation = pearsonr(p[i], p[j])
            if abs(relation[0]) < 0.85:
                return None
            mi, mj = np.mean(p[i]), np.mean(p[j])
            if min(mi, mj) / max(mi, mj) < 0.95:
                return None
    # Xác định chữ kí của pattern
    signature = [0.0] * z
    for i in range(z):
        signature[i] = sum(p[j][i] for j in range(q)) / q
    return signature