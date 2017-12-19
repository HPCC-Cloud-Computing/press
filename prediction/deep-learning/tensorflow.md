# TensorFlow

## Giới thiệu
TensorFlow là một framework Deep Learning do Google xây dựng.
TensorFlow hỗ trợ tính toán song song trên cả CPU và GPU.

## Cài đặt
[Hướng dẫn cài đặt đầy đủ](https://www.tensorflow.org/install/)

**Cài đặt Anaconda và TensorFlow trên Ubuntu 16.04**
1. Vào trang chủ của [Anaconda](https://www.continuum.io/downloads), download và cài đặt.
(Ở đây sử dụng phiên bản Anaconda 4.3.0 (64-bit) với python 3.6.0)
2. Tạo môi trường conda, đặt tên là tensorflow
```bash
$ conda create -n tensorflow
```
3. Kích hoạt môi trường vừa tạo
```bash
$ source activate tensorflow
```
4. Cài đặt tensorflow
```bash
(tensorflow)$ pip install --ignore-installed --upgrade 
```
5. Kiểm tra cài đặt
Trong Terminal gõ lệnh
```bash
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```
Kết quả
```bash
Hello, TensorFLow
```

## TensorFlow cơ bản
### Import thư viện
```python
import tensorflow as tf
```
Cách này cho phép sử dụng tất cả các class, method, symbol của TensorFlow.

### Tensor
Tensor là đơn vị dữ liệu chính trong TensorFlow, nó được biểu diễn dưới dạng mảng n chiều. 
Trong đó n được gọi là *rank* của tensor. Ví dụ:
```python
5   # tensor có rank 0
[1., 2., 3.]    # tensor có rank 1, đây là vector có kích thước [3]
[[1., 2., 3.], [4.1, 5.1, 6.1]] # tensor có rank 2, đây là ma trận kích thước [2, 3]
[[[1., 2., 3.]], [[4.1, 5.1, 6.1]]] # tensor rank 3, kích thước [2, 1, 3]
```

### Computational graph
Computational graph là dãy các toán tử được sắp xếp thành đồ thị(graph).
Mỗi node của graph nhận 0 hoặc một và tensor làm input và cho output là một tensor.
Một node cũng có thể là hằng số.

Ví dụ, tạo 2 node là hằng số có kiểu số thực.
```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)    # mặc định kiểu là tf.float32
print(node1, node2)
```
Kết quả
```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```

### Session
Trong ví dụ trên, kết quả in ra không phải là 3.0 và 4.0 như mong muốn.
Để tính toán trên các node, ta phải chạy computational graph trong một **session**.
Một **session** bao gồm các điều khiển và trạng thái trong quá trình thực thi TensorFlow.

Tạo và thực thi một Session
```python
sess = tf.Session()
# code
sess.close()
```
hoặc
```python
with tf.Session() as sess:
    sess.run(f)
```

Trong ví dụ ở phần trên, để thực hiện tính toán trên `node1` và `node2`, ta phải tạo một session và gọi phương thức `run()`
```python
sess = tf.Session()
print(sess.run([node1, node2]))
```
Kết quả
```bash
[3.0, 4.0]
```

Ví dụ, tạo ra `node3` thực hiện cộng 2 giá trị của `node1` và `node2`
```python
node3 = tf.add(node1, node2)
print(node3)
print('sess.run(node3):', sess.run(node3))
```
Kết quả
```
Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
```

### Placeholder
Một graph có thể nhận input từ bên ngoài bằng cách sử dụng placeholder. 
Khi chạy một node được tạo ra bằng `placeholder`, ta phải cung cấp input cho node đó.
Ví dụ
```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a + b    # toán tử + thay cho tf.add()
print(sess.run(add_node, {a: 3.0, b: 4.5}))
print(sess.run(add_node, {a: [1, 3], b: [2, 4]}))
```
Kết quả
```
7.5
[ 3. 7.]
```

### Variable
Ví dụ, tạo model tuyến tính
```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

Mặc định, `variable` không được khởi tạo khi gọi phương thức `tf.Variable`. 
Để khời tạo tất cả các biến trong TensorFlow:
```python
init = tf.global_variables_initializer()
sess.run(init)
```

Do `x` là một placeholder nên ta có thể tính `linear_model` cho nhiều giá trị của `x`
```python
print(sess.run(linear_model, {x:[1,2,3,4]}))
```
Kết quả
```
[ 0.          0.30000001  0.60000002  0.90000004]
```

Tính loss function 
```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)    # tính tổng các bình phương
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

Để thay đổi giá trị một biến, sử dụng hàm `tf.asign`. 

Ví dụ, thay giá trị biến `W` và `b` thành giá trị tối ưu (-1.0 và 1.0).
```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

### API **tf.train**
**TensorFlow** cung cấp sẵn các **optimizer** để thay đôi giá trị các biến nhắm tối thiểu hóa hàm mất mát (loss function). 
Ví dụ với **gradient descent**, giá trị các biến sẽ được chỉnh sửa sau mỗi lần lặp bằng cách trừ đi đạo hàm của hàm mất mát nhân với hệ số học.

Ví dụ
```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
```

Kết quả, `W` và `b` sau khi tối ưu
```
[array([-0.9999969], dtype=float32), array([ 0.99999082],
 dtype=float32)]
```

Chương trình hoàn chỉnh
```python
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
sess.close()
```

### **tf.contrib.learn**
**tf.contrib.learn** là thư viện trong TensorFlow nhằm đơn giản hóa việc thực hiện các thuật toán trong Machine Learning, bao gồm:
- thực thi vòng lặp huấn luyện
- thực thi vòng lặp đánh giá
- quản lý tập dữ liệu
- quản lý việc chuyển dữ liệu

tf.contrib.learn cung cấp sẵn các mô hình thông dụng.

Ví dụ, thực hiện chương trình hồi quy tuyến tính
```python
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
estimator.evaluate(input_fn=input_fn)
```

## Tham khảo
[1] https://www.tensorflow.org/get_started/get_started

[2] http://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow-tutorial/