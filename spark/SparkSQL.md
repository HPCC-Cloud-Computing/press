# Spark SQL, DataFrames and Datasets

## Overview
**Spark SQL** là module trong Spark cho phép xử lý dữ liệu có cấu trúc.

### SQL
Spark SQL có thể dùng để thực hiện truy vấn SQL hoặc lấy dữ liệu từ *Hive*.
Khi thực hiện *SQL* trong ngôn ngữ lập trình khác (Java, Scala, Python), kết quả được trả về dưới dạng **Dataset** hoặc **DataFrame**.

### Dataset
**Dataset** là một collection của dữ liệu được lưu trữ phân tán. 
Dataset được thêm vào từ phiên bản Spark 1.6.
Dataset API được hỗ trợ trên Java và Scala, không được hỗ trợ trên Python. 
Tuy nhiên, nhiều tính năng của Dataset API vẫn thực hiện được trên Python (VD: truy cập đến các trường trong hàng bằng `row.columnName`).

### DataFrame
**DataFrame** là Dataset dưới dạng bảng với các cột có tên.
Khái niệm DataFrame tương tự như khái niệm bảng trong cơ sở dữ liệu quan hệ hoặc khái niệm data frame trong Python.

DataFrame có thể được tạo ra từ:
  * file dữ liệu có cấu trúc
  * bảng trong Hive hoặc trong cơ sở dữ liệu bên ngoài
  * RDD có sẵn

DataFrame API được hỗ trợ trên Java, Scala, Python và R.
