Tìm hiểu về High Availability (HA) trong Cloud
======
Giới thiệu
------
* Trong đời sống, sẽ có nhiều trường hợp xảy ra những rủi ro về hiệu năng của các Server xảy ra từ các chuỗi sự kiện, xuất hiện 1 spike đột ngột trong lưu thông có thể dẫn tới sự cố về điện. Các Server có thể bị mất ổn định - bất luận là ở nơi mà các ứng dụng được quản là trong cloud hay là một máy chủ vật lý. Trong thực tế thì điều này không thể tránh khỏi. Tuy nhiên, để phần nào đảm bảo rằng những sự cố trên không thể xảy ra, việc cần làm là nâng cấp hệ thống nhằm phòng tránh rủi ro. 
* Câu trả lời là sử dụng các thiết lập và kiến trúc High Availability. Kiến trúc HA là một tiếp cận theo phần tử, các module và thực thi các service của một hệ thống đảm bảo hiệu năng tổ chức được tối ưu, kể cả những thời điểm gặp tải cao. Dù chưa có các yêu cầu hay ràng buộc cụ thể nào đối với các hệ thống kiểu này, nên dưới dây sẽ là một số cách tiếp cận của riêng người viết và sưu tầm.


HA là gì?
------
* Trong tính toán, thuật ngữ HA thường được mô tả một khoảng thời gian một Service được coi là sẵn sàng, đồng nghĩa với thời gian yêu cầu hệ thống phải phản hồi 1 request từ user. HA là một kiểu khả năng của hệ thống phải đảm bảo về mức sẵn sàng ở trạng thái cao.

Downtime
------
### Downtime là gì? ###
 * Downtime là một khoảng thời gian àm hệ thống hay mạng không có sẵn để sử dụng, hay không sẵn sàng đáp lại
### Phân loại Downtime ###
 * Scheduled Downtime: Gây ra từ việc maintencance các Service. Điều này thì hoàn toàn không thể tránh khỏi. Bao gồm việc nâng cấp phần mêm bằng  bản vá, cập nhật phần mêm hoặc thậm chí là một vài thay đổi trong mạng database
 * Unscheduled Downtime: Gây ra bởi các sự kiện không thấy được như: Lỗi phần cứng, lỗi phần mềm. Điều này xảy ra do mất điện hoặc một linh kiên bị hỏng. Gây ảnh hưởng xấu đến hiệu năng tính toán.
### Hậu quả mà Downtime gây ra ###
 * Khiến một tổ chức bị thất thoát rất do tất cả các dịch vụ sử dụng trên server sẽ phải trong trạng thái Hold khi hệ thống sập
 * Tùy vào tính chất của các Service mà các request chỉ có ý nghĩa trong một vài thời gian hạn định, do vậy tính sẵn sàng mức thấp có thể làm giảm hiệu quả của các tính toán hoặc hoàn toàn không hiệu quả.
### Nhiệm vụ đặt ra ###
 * Áp dụng một kiến trúc HA để đảm bảo rằng hệ thống hoặc ứng dụng được thiết lập nhằm kiểm soát các workload khác nhau và các lỗi chứa downtime nhằm đưa downtime xuống thấp nhất hoặc zero downtime.

### Availability được xác định như thế nào? ###
 * Các tổ chức và cộng đồng, công ty tận dụng hoàn toàn hạ tâng Cloud phải luôn yêu cầu tài nguyên 24/7. Availability có thể được đo bằng tỉ lệ thời gian hệ thống này có sẵn
>        -   x = (n-y)/n
>
>        n là tổng số phút trong một tháng
>
>        y là tổng số phút mà service không có sẵn
 * VD: Một hế thống có chỉ số Availability là 0.99 ( tức 99%) thì trong một năm downtime sẽ là 3.65 ngày
 * HA đơn giản sẽ liên hệ tới một phần tử hoặc hệ thống được hoạt động liên tục trong một khoảng thời gian được kì vọng. Tuy nhiện hầu hết không thể đạt tới tiêu chuẩn về sẵn sàng cho một sản phẩm hệ thống là 1.00. Trong kinh doanh, HA gần như là bắt buộc nhằm chống lại rủi ro từ việc thiếu thốn tài nguyên hệ thống. Các rủi ro ở đây có thể lên đên hàng triệu đô tiền lỗ.
 
 
Cơ chế hoạt động của HA
------
 * Các phương thức HA được dựa trên kĩ thuật đáp ứng lại lỗi cho cơ sở hạ tầng.
 * Cách hoạt động đặc trưng phải yêu cầu các phần mềm và thiết lập chuyên biệt.
 

Sự quan trọng của HA
------
 * Khi thiết lập các hệ thống sản xuất tốt kèo dài, việc tối thiểu hóa downtime và gián đoán service là một ưu tiên cao. Bất chấp hệ thống và phần mềm có đáng tin cậy đến đâu, nhưng vấn đề sự cố đều có thể xảy ra bất cứ lúc nào làm sập các application server.
 * Việc sử dụng HA cho cơ sở hạ tâng là một trong những chiến lược có hiệu quả để giảm thiểu tác động từ nhiều biến cố khác nhau. Các hệ thống HA có thể phục hồi từ động các lỗi mà server hoặc phần tử, linh kiên gặp phải.
 
 
Xây dựng hệ thống được HA
------
> Loại trừ các nút lỗi trong cơ sở hạ tầng.
>
> Nút lỗi gây gián đoạn một service nếu nút này trở nên không sẵn sàng
>
> => Xử lý: Từng lớp của stack đều được phải có các bản sao dự bị. 
### Các trường hợp xảy ra: ###
  Server down
  > Load Balancer phân tán đều đến 2 server
  >
  > Nếu xảy ra server chính sập, sẽ có 1 cờ của server đó gửi tới loadbalancer để loadbalancer chuyển hướng tới bản sao .
  > 
  LoadBalancer down
  > Việc xây dựng theo hệ thống top-down sẽ không xử lý được trường hợp loadbalancer down do các tính toán và điều hướng lúc này đều nằm dưới loadblancer sẽ không hoạt động tác động lên tầng trên được.
  >
  > Giải pháp: Nhóm các loadbalancer đặc dụng cho một nhiệm vụ vào một cụm kèm theo khả năng phát hiện và phục hồi lỗi tương đương.
  >
  > Phục hồi Load Balancer bằng cách chịu lỗi tới loadblancer bản sao, hàm ý rằng một thay đổi DNS được thực thi để trỏ tên domain tới IP bản sao. Nhưng mất thời gian vào việc thực thi trên Internet => Gây downtime nghiêm trọng
  > 
  > Giải pháp: Sử dụng LoadBalancer DNS round-robin
  >
  > Nhược điểm: Không tin cậy
  >
  > => Sử dụng floating IP: Remap địa chỉ IP theo yêu cầu, loại bỏ đi phần lan truyền và phân bị thất thoát khi đổi DNS bằng cách cung cấp một IP tĩnh để dễ dàng remap nếu cần thiết.
  >
  Database down
  > CAP Theorem:
  * Tính nhất quán: Không được xảy ra nhiều giá trị trong cùng một đơn vị dữ liệu
  * Tính sẵn sàng: Hoạt động đẩy đủ
  * Chịu được phân chia: phản hồi chính xác lỗi của các nút hoặc mạng
  * Một hệ thống chỉ cần đáp ứng 2 trong 3 tiêu chí.
  >
  > Các database chạy trên các server độc lập ( như Amazon RDS, chưa tìm hiểu) và dễ xảy ra crash. Nếu xảy ra crash sẽ dẫn tới mất dữ liệu người sử dụng và có thể rất đáng giá.
  >
  > => Giải pháp: Sử dụng NoSQL
  >
  * Khả năng phục hồi tối đa: Sử dụng các hệ thống phân tán nhằm tạo nhiều bản sao của database nhằm khắc phục các lỗi trên từng nút hiệu quả.
  * Tính sẵn sàng với sự nhất quán: Đảm bảo các yêu cầu về đọc và ghi vẫn được chấp nhận ngay cả khi nhiều server offline, nghĩa là cho phép hệ thống không thật sự nhất quán trong một thời gian ngắn ( vài mili-giây). Tuy vậy, việc không sẵn sàng có thể gây ra thất thoát doanh thu, giảm sự tin tưởng của người sử dụng ( như các ứng dụng game, bán hàng, quảng cáo, ...) nên cần cân bằng giữa availability và consistency.
  * Khả năng đọc và ghi: Trong trường hợp một server hoặc một mạng bị lỗi - thậm chí là database "HA" NoSQL - ta vẫn sẽ phải chấp nhập yêu cầu ghi nhưng lại không cho phép dữ liệu được truy cập cho đến khi cụm được sửa. Giải pháp ở đây là tạo bản sao request và thực thi trên một database bản sao. Các database bản sao sẽ tạm thời được sử dụng thay cho các database chính cho đến khi lỗi được sửa. Vì vậy, việc đọc và ghi vẫn được giữ ở mức HA.
                
                
    
    
    
