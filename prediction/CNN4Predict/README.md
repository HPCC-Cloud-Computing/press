# Sử dụng CNN cho bài toán forecast time series

## Tổng quan
### Dataset 

- Worldcup98 dataset day6-10

### Input/Output
- X = x[1], x[2], ..., x[n] là chuỗi các thông tin về số request trong khoảng thời gian 600s
- Input: Các dãy số: x[t],x[t+1],...,x[t+p-1] (k > 0)
- Output:x[t+p]
- Train data: Day 6-9 {x[i]} ( 0 < i < 577)
- Test data: Day 10 {x[j]} ( 576 < i < 721)
- Tham số:
    - CNN: p - window_size của input, tức số neuron tại input layer
    - K-shift: K - thể hiện Output là giá trị trễ k phần tử của input tương ứng ( x[t-k] = y[t])

### Mục tiêu
- Đưa ra giá trị y sao cho hàm RMSE giữa các giá trị y dự đoán và x tương ứng khớp với nhau
### So sánh hiệu quả của CNN
- Tham chiếu hiệu quả thông qua hàm RMS
- So sánh hiệu quả của CNN với các window size: 20, 30, 50
- So sánh hiệu quả của CNN với K-shift