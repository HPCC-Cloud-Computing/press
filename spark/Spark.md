# Apache Spark
## Các thành phần của Spark
![Spark Components](https://www.tutorialspoint.com/apache_spark/images/components_of_spark.jpg)
- Spark Core: Bao gồm các chức năng cơ bản của Spark (quản lý task, quản lý bộ nhớ, khắc phục lỗi, tương tác với hệ thống lưu trữ...). Spark Core cũng cung cấp các API định nghĩa *Resilient Distributed Datasets - RDD*.
- Spark SQL: Hỗ trợ tương tác với Spark qua truy vấn SQL.
- Spark Streaming: Thành phần của Spark giúp xử lý data streaming. (VD: log files được sinh ra bởi web server)
- Spark MLlib: Thư viện cung cấp các chức năng, giải thuật thông dụng trong *Machine Learning*.
- Spark GraphX: Cung cấp các API xử lý đò thị, bao gồm cả tính toán song song.

## Cài đặt
* Yêu cầu: Đã cài đặt Java
* [Download Spark](https://spark.apache.org/downloads.html)
* Giải nén và đặt *enviroment variables*
```bash
SPARK_HOME=<path-to-spark>
PATH=$PATH:$SPARK_HOME/bin
```
* Kiểm tra cài đặt
```bash
$ pyspark 
```

## Initializing Spark (sử dụng Python)
Để thực hiện việc truy cập và tính toán trên các cluster, chương trình Spark trước hết phải tạo ra đối tượng `SparkContext`.
Để tạo ra đối tượng `SparkContext`, cần tạo ra đối tượng `SparkConf` để chứa thông tin về chương trình.
```python
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
```
`master` là URL của cluster, gán "local" để chạy ở local mode.

Chạy Spark ở chế độ command:
```bash
$ pyspark
```
Mặc định khi chạy `pyspark`, một đối tượng `SparkContext` được tạo ra với tên là `sc`.

Để xem các thông tin về các tiến trình Spark đang hoạt động, truy cập vào `<url-cluster>:4040`.

## RDD - Resilient Distributed Datasets
RDD là cấu trúc dữ liệu cơ bản của Spark, bao gồm tập các object được lưu trữ phân tán.
Mỗi dataset trong RDD được chia thành các phân vùng logic, có thể được tính toán trên các node khác nhau.

2 cách để tạo ra RDD:
- Tạo ra từ dữ liệu sẵn có được lưu trên đĩa
- Tham chiếu đến dataset ở hệ thống lưu trữ bên ngoài như `share file system`, `HBase`, `HDFS` hoặc các nguồn dữ liệu trong `Hadoop Input Format`.

Ví dụ:
```python
# Tạo ra RDD từ file text
logFile = sc.textFile("log.txt")

# Tạo ra RDD từ list có sẵn
listNum = sc.parallelize([2, 3, 5, 7, 11, 13])

# Tạo ra RDD từ một RDD khác:
# VD: Lọc ra các error messages trong log
errorLine = logFile.filter(lambda x: "error" in x) 
```

### RDD Operations
Gồm 2 loại: `Transformation` và `Action`

#### Transformations
Là operation trả về một RDD mới. `Transformation` có cơ chế `lazy evaluation`, tức là nó chỉ thực sự hoạt động khi gặp một `action`.

Một số API thông dụng:
* Tạo ra 1 RDD từ 1 RDD có sẵn.(VD: 1 đối tượng tên `rdd` được tạo ra từ list [1, 2, 3, 3])

|API|Chức năng|Ví dụ|Kết quả|
|---|---------|------|-----|
|`map(func)`|Thực thi một *function* trên mỗi phần tử của RDD và trả về RDD của kết quả|`rdd.map(lamba x: x+1)`|[2, 3, 4, 4]|
|`filter(func)`|Trả về RDD gồm các phần tử thoả mãn điều kiện|`rdd.filter(lambda x: x != 1)`|[2, 3, 3]|
|`distinct()`|Loại bỏ các thành phần lặp lại|`rdd.distinct()`|[1, 2, 3]|

* Tạo ra RDD từ 2 RDD có sẵn. (VD: đối tượng `rdd` chứa [1, 2, 3] và `other` chứa [3, 4, 5])

|API|Ví dụ|Kết quả|
|---|------|-----|
|`union`|`rdd.union(other`)|[1, 2, 3, 3, 4, 5]|
|`intersection`|`rdd.intersection(other)|[3]|
|`subtract`|`rdd.subtract(other)`|[1, 2]|
|`cartesian`|`rdd.cartesian(other)`|[(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]|

#### Actions
Là operation trả về một giá trị cụ thể cho chương trình.
Một số API thông dụng:
(Ví dụ với đối tượng `rdd` chứa [1, 2, 3, 3])

|API|Chức năng|Ví dụ|Trả về|
|---|---------|-----|------|
|`collect()`|Trả về tất cả phần tử của RDD|`rdd.collect()`|[1, 2, 3, 3]|
|`count()`|Lấy số lượng phần tử|`rdd.count()`|4|
|`take(num)`|Lấy `num` phần tử của rdd|`rdd.take(2)`|[1, 2]|
|`top(num)`|Lấy `num` phần tử cuối cùng|`rdd.top(num)`|[3, 3]|
|`reduce(func)`|Tổ hợp các phần tử|`rdd.reduce(lambda x, y: x-y)`|-7|

### Lazy Evaluation
`Spark` không thực sự thực hiện `transformation` cho đến khi gặp một `action`.
Khi gọi một `transformation` của một RDD, `Spark` không thực hiện ngay mà nó ghi lại một *meta data* nhằm đảm bảo `transformation` sẽ được thực hiện khi có yêu cầu.
Spark sử dụng `lazy evaluation` nhằm giảm lượng tính toán bằng cách nhóm các operation lại với nhau. 

### Persistence(Caching)
Khi muốn sử dụng một RDD nhiều lần, để tránh phải tính toán lại RDD, ta sử dụng API `persist(storageLevel)`.
Storage level được cung cấp trong module `pyspark.StorageLevel`.
![Storage level](https://image.slidesharecdn.com/sparkarchitecture-jdkievv04-151107124046-lva1-app6892/95/apache-spark-architecture-70-638.jpg?cb=1446900275)

Với option `MEMORY_ONLY`, có thể thay thế bằng phương thức `cache()`.

## Key-Value Pairs
Spark cung cấp các hàm tính toán với loại RDD gồm cặp giá trị *key-value*.
Những RDD như thế được gọi là *Pair RDD*.

### Tạo Pair RDD

Ví dụ:
```python
# Tạo RDD gồm các từ trong file text
inputFile = sc.textFile("input.txt")

# Tạo Pair RDD với từ đầu tiên làm key
lines = inputFile.map(lambda x: (x.split(" ")[0], x)) 
```

### Một số API
Ngoài các API với RDD thông thường, Spark còn cung cấp thêm các API cho Pair RDD.

(Ví dụ với một đối tượng `rdd` chứa [(1, 2), (3, 4), (3, 6)])

|API|Chức năng|Ví dụ|Kết quả|
|----|--------|------|-------|
|`groupByKey()`|Nhóm các *value* có cùng *key*|`rdd.groupByKey()`|[(1, [2]), (3, [4, 6])]|
|`reduceByKey(func)`|Thực hiện tính toán các *value* có cùng *key*|`rdd.reduceByKey(lambda x, y: x + y)`|[(1, 2), (3, 10)]|
|`mapValues(func)`|Thực hiện `func` trên các *value*, không làm thay đổi *key*|`rdd.mapValues(lambda x: x+1)`|[(1, 3), (3, 5), (3, 7)]|
|`keys()`|Trả về RDD chứa các *key*|`rdd.keys()`|[1, 3, 3]|
|`values()`|Trả về RDD chứa các *value*|`rdd.values()`|[2, 4, 6]|
|`sortByKey()`|Sắp xếp theo *key*|`rdd.sortByKey()`|[(1, 2), (3, 4), (3, 6)]|

## Một số chú ý
- Sử dụng Spark DataFrames
- Không gọi `collect()` đối với RDD lớn
- Reduce trước khi Joining
- Tránh `groupByKey()` trên RDD lớn

## Tham khảo
* [Tutorialspoint - Apache Spark](https://www.tutorialspoint.com/apache_spark)
* [Spark Programming Guide](https://spark.apache.org/docs/latest/programming-guide.html)
* Ebook: [Learning Spark](http://bigdata.ui.ac.ir/library/files/books03.pdf)