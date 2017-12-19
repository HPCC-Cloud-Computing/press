# Xây dựng hàm convert về số lượng box và các bộ vector-label
## Mô tả
* Input: Một chuỗi các giá trị request được lấy từ một hoặc nhiều file nhập vào
* Output: Các bộ vector-label được lưu trong file CSV 
## Thử nghiệm
* Chạy với câu lệnh sau:
```
python preprocess.py [-h] [-o OUTPUT] [-i INPUT] [--bmr BMR] [--vs VS]
                     [--sim SIM] [--som SOM] [--start START] [--end END]
```
* Với ý nghĩa từng tham số như sau:
    * INPUT: Đường dẫn tới file input
    * BMR: Box max request - Số request tối đa trong một box
    * VS: Kích thước vector input
    * SIM: Scale in max: Số scale in tối đa
    * SOM: Scale out max: Số scale out tối đa
    * START: Dữ liệu ngày bắt đầu (World cup dataset only)
    * END: Dữ liệu ngày kết thúc (World cup dataset only)