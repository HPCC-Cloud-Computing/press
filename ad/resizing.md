Mở rộng dung lượng nhớ
===
Mục đích
--- 
* Trước hết, kiến trúc lambda mà chúng ta xây dựng để tiến hành lấy dữ liệu và phân tích log sẽ một lượng tài nguyên đáng kể để có thể tiến hành lưu trữ và xử lý các dữ liệu này. 
* Vì các tài nguyên trong quá trình xử lý thực tế có thể không lường trước được lớn như thế nào (cái này có thể không đúng vì đây là lần đầu làm với môi trường thật), do vậy sẽ cần phải có giải pháp mở rộng dung lượng nhớ cho các node lưu trữ nhiều dữ liệu.
* Vấn đề đặt ra ở đây không phải chỉ dừng tại việc mở rộng dung lượng bộ nhớ cho các node mà còn phải đảm bảo các node này vẫn hoạt động bình thường, không xảy ra thất thoát dữ liệu trong node và liên kết giữa các node với nhau.

Tổng quan việc tiến hành
---
### Đối tượng tiến hành
Các dữ liệu về log sẽ được lưu trong 3 node là **log01**, **log02** và **log03**. Do vậy việc mở rộng dung lượng bộ nhớ sẽ được thực hiện trên 3 con node này.
### Các bước tiến hành
Việc tiến hành mở rộng dung lượng nhớ như sau:
* Lưu trữ các dữ liệu có thể bị mất trong quá trình tiến hành.
* Xóa các phân vùng cản trở đến việc mở rộng phân vùng mục tiêu
* Mở rộng phân vùng mục tiêu
* Khôi phục lại một số phân vùng cần thiết

Chi tiết việc tiến hành
---
### Kiểm tra dung lượng
Sử dụng lệnh ```slblk``` , ta thu được:
```
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sr0 	11:0	1  447M  0 rom  
sda  	8:0	0   70G  0 disk
├─sda1   8:1	0  500M  0 part /boot
├─sda2   8:2	0   25G  0 part /
├─sda3   8:3	0	2G  0 part [SWAP]
├─sda4   8:4	0	1K  0 part
└─sda5   8:5	0 12,5G  0 part /home
```
### Lưu trữ dữ liệu
* Dữ liệu ta cần lưu trữ ở đây là thư mục ```/home``` , do vậy ta sẽ chạy lệnh như sau:
```
tar cvzf home-backup.tar.gz /home
```
* Vị trí lưu có thể tùy chỉnh.
### Thực hiện các thao tác chính
* Phân vùng cần mở rộng : ```/dev/sda2```
* Sau khi đã lưu dữ liệu tại thư mục home, tức partition ```/dev/sda5``` , ta sẽ tiến hành xóa các phân vùng à ```/dev/sda3``` , ```/dev/sda4```, ```/dev/sda5```
* Chỉnh sửa thư mục ```/etc/fstab```
* Xóa các thông tin liên quan đến các phân vùng cần xóa.
* Thực hiện các câu lệnh sau:
```
umount /dev/sda5
```
* Nếu device busy thì thưc hiện tìm các process và xóa:
```
fuser -mv /dev/sda5
```
* Thực hiện thao tác trên ```/dev/sda```
```
fdisk /dev/sda
```
* Xóa các phân vùng (cả ```sda2``` )
* Không được xóa phân vùng 1, điều này không cần thiết và tự gây khó khăn cho mình.
* Tạo phân vùng ```sda2``` với dữ liệu được mở rộng
* Nên để dư tầm 10gb cho phần swap và những phần có thể phát sinh
* Tạo phân vùng swap với phần dữ liệu còn lại
* Lưu trữ các thao tác vừa thực hiện
```
Resize /dev/sda2
resize2fs /dev/sda2
```
Reboot
### Khôi phục dữ liệu
* Giải nén các dữ liệu cần khôi phục:
```
tar xvzf home-backup.tar.gz
```
### Kiểm tra các thông tin
* Sử dụng lệnh:
```
df -h
```
* Ta thu được:
```
Filesystem  	Size  Used Avail Use% Mounted on
/dev/sda2    	59G  4,5G   52G   9% /
tmpfs       	939M   12K  939M   1% /dev/shm
/dev/sda1   	477M   52M  400M  12% /boot
```
* Sử dụng lệnh:
```
lsblk
```
* Ta thu được:
```]
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sr0 	11:0	1  447M  0 rom  
sda  	8:0	0   70G  0 disk
├─sda1   8:1	0  500M  0 part /boot
├─sda2   8:2	0   60G  0 part /
└─sda3   8:3	0  9,5G  0 part
```
Tài liệu tham khảo
---
* Phần xóa các phân vùng : [https://www.howtoforge.com/linux_resizing_ext3_partitions_p3](https://www.howtoforge.com/linux_resizing_ext3_partitions_p3)
* Phần resize phân vùng mới: [https://www.howtoforge.com/linux_resizing_ext3_partitions](https://www.howtoforge.com/linux_resizing_ext3_partitions)
* Phần resize phần vùng root: [https://askubuntu.com/questions/24027/how-can-i-resize-an-ext-root-partition-at-runtime](https://askubuntu.com/questions/24027/how-can-i-resize-an-ext-root-partition-at-runtime)
* Phần tạo phân vùng swap: [http://www.computernetworkingnotes.com/file-system-administration/how-to-create-swap-partition.html](http://www.computernetworkingnotes.com/file-system-administration/how-to-create-swap-partition.html)
* Phần nén và giải nén với câu lệnh tar: [https://www.tecmint.com/18-tar-command-examples-in-linux/](https://www.tecmint.com/18-tar-command-examples-in-linux/)


