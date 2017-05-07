# TỔNG QUAN VỀ CLOUD COMPUTING
---
Được tham khảo từ tài liệu: Cloud computing principles and Paradigms
* Tham khảo sơ đồ Mind-map theo link sau: [Mindmap] - Cloud
##  I. Cloud Computing là gì?
    Một công nghệ ảo hóa (Virtualization) các tài nguyên tính toán và dịch vụ, ứng dụng thông qua môi trường Internet.
##  II. Kiến trúc của Cloud Computing
    Kiến trúc của Cloud Computing nhất quán đối với từng sản phẩm mà theo sự đề xuất của các nhà sản xuất. Nhìn chung, Cloud Computing bao gồm các phần sau:
        - Infrastructure as a Service (IaaS)
        - Platform as a Service (PaaS)
        - Software as a Service (SaaS)

##  III. Gốc rễ của Cloud computing
###  1. Ảo hóa phần cứng 
    Đây là nền tảng cốt lõi của hệ thống Cloud Computing. Sử dụng các tài nguyên vật lý ( như RAM, CPU, HDD, SSD) và các cấu trúc hệ thống vật lý ( Network, Server, ... ) trên một môi trường mô phỏng máy thật thông qua hypervisor.
    -   Hypervisor
    -   VMWare ESXi
    -   Xen 
    -   KVM
###  2. Các thiết bị ảo và Định dạng ảo
####   a.  Thiết bị ảo
    -   Các ổ đĩa ảo ( VM Disk Image)
####   b.  Open Virtualization Format (OVF):
    -   Mô tả những đặc điểm của phần cứng VM như:
        -   Files
        -   Thông tin OS, start-up, các thao tác shutdown, các đĩa ảo, metadata chứa các thông tin sản xuất và giấy phép
        -   Các VM phức hợp
        -   Mở rộng: Virtual Machine Contracts (VMC)
####   c.  Điện toán tự trị (Autonomic Computing)
    -   Cơ sở của đặc điểm self-service được đề cập đến ở phần sau
    -   Làm giảm những tác động của con người
    -   Tự quản lý
    -   Quản lý các mức dịch vụ chạy các ứng dụng
    -   Quản lý dung lượng data center
    -   Phục hồi thảm họa được dự liệu
    -   Tự động cung ứng VM
##  IV. Các lớp của cloud
### 1.  Infrastructure as a Service (IaaS)
    Thực hiện các Service cloud ở cấp độ cơ sở hạ tầng, bao gồm các tác vụ sau:
        -   Computation
        -   Storage 
        -   Communication
        -   Virtual Infrastructure Manager (VIM)
        -   Amazon Web Services (AWS)
    Đặc điểm của IaaS:
        -   Hiện diện theo địa lý
        -   Tương tác người sử dụng và truy cập đến các Server
        -   Advance Reservation của sức chứa
        -   Tự động Scaling và cân bằng tải ( Load Balancer)
        -   Service-Level Agreements (SLAs)
        -   Hypervisor và lựa chọn OS
### 2.  Platform as a Service (PaaS)
    Thực hiện các Service cloud ở cấp độ các ứng dụng hệ thống, ứng dụng nền tảng, bao gồm các tác vụ sau:
        -   Raw computing and storage services
        -   Làm các cloud có khả năng lập trình
        -   Tạo và thiết lập các application mà không cần thiết phải biết bao nhiêu tài nguyên máy sẽ được sử dụng
        -   Các mô hình lập trình đa ngôn ngữ và ứng dụng đặc thù
        -   Google AppEngine, Mircrosoft Azure, Force.com
    Đặc điểm của PaaS:
        -   Các mấu lập trình, ngôn ngữ lập trình và framework
        -   Các tùy trình lưu trữ: Files và database
### 3.  Software as a Service (SaaS)
    Thực hiện các Service cloud cho người sử dụng cuối thông qua cổng Web:
        -   Giảm thiểu gánh nặng cho việc duy trì phần mề
        -   Đơn giản hóa việc phát triển và kiểm thử
        -   SálesForce.com đi kèo với hệ thống CRM
##  V. Các loại Cloud
### 1.  Public Clouds
    -   Dịch vụ bên thứ 3, cung cấp các cơ sở hạ tầng và service cloud hướng đến nhiều thuê bao.
    -   Tài nguyên gần như vô hạn ( luôn ở mức đáp ứng được)
    -   Tính co giãn lớn, thích hợp trong việc tối ưu hóa
    -   Vấn đề bảo mật, an toán thông tin chưa thật sự đảm bảo
### 2.  Private Clouds
    -   Hoạt động trên một Data Center/ cơ sở hạ tầng cho việc sử dụng nội bộ
    -   Tính bảo mật cao
    -   Khó khăn trong việc scaling và xử lý các vấn đề overload
### 3.  Hybrid Clouds
    -   Private Cloud được cung ứng tài nguyên tính toán từ public Cloud
    -   Khắc phục nhược điểm của 2 hệ thống trên
    -   Xây dựng phức tạp
### 4.  Community Clouds
    -   Được chia sẻ bởi một vài tổ chức và hỗ trợ một cộng đồng chia sẻ liên quan
##  VI. Các đặc điểm kỳ vọng về Cloud so với các hệ thống tính toán khác
### 1.  Self-service
    -   Hoạt động không qua sự can thiệp của con người
    -   Các cổng tiếp nhận và thanh toán sử dụng
### 2.  Tính co giãn:
    -   Cung cấp tài nguyên một cách nhanh chóng bất kì kiểu request nào tại mọi thời điểm
    -   Tự động Scale up và Scale down
    -   Khả năng tùy chỉnh
## VII. Quản lý cơ sở hạ tầng Cloud
    Dựa trên nền tảng Virtual Infrastructure Manager (VIM)
    Các đặc điểm:
        -   Hỗ trợ ảo hóa.
            -   Các tài nguyên ảo hóa được định kích cỡ hoặc thay đổi kích cỡ với từng nhu cầu cụ thể
            -   Thực thi ảo hóa phần cứng
            -   Tạo một cơ sở hạ tầng ảo mà các phần tử của một data center xử lý chung cho đa người dùng
        -   Self-SerVice, On-Demand Resources Provisioning
            -   Sinh ra các khởi tạo của một server tương tác với người quản trị hệ thống
        -   Multiple Backend Hypervisors
            -   Cung cấp các trình điều khiển nối được để tương tác với multiple hypervisor
            -   Ảo hóa lưu trữ
                -   Trừu tượng lưu trữ logic từ kho lưu trữ vật lý
                -   Cho phép tạo các đĩa ảo độc lập với thiết bị và vị trí
                -   Hệ thống lưu trữ trao đổi nội bộ thông qua SAN ( Storage Area Network )
                -   Fibre Channel, iSCSI, NFS
                -   Một bộ điều khiển kho lưu trữ cung cấp tầng trừu tượng giữa lưu trữ ảo và lưu trữ vật lý
                -   VMWare, Citrix, ...
            -   Tương tác với các Public Clouds
                -   Mở rộng sức chứa hạ tầng tính toán cục bộ in-house bằng việc mượn các tài nguyên từ các dịch vụ Public Clouds
                -   Các private clouds cũng có thể tương tác với các public clouds nhằm yêu cầu thêm tài nguyên và quản lý chúng.
            -   Mạng ảo
                -   VLAN: Cho phép VMs được nhóm thành các domain cùng broadcast
                -   VMs có thể được cấu hình để chặn các nguồn lưu lượng từ các con VM thuộc về các mạng khác
                -   VPN: Mô tả một mảng private an toàn và ảo ở lớp trên cùng của một mạng
            -   Cấp phát tài nguyên động
                -   Hệ thống quản lý các tài nguyên chưa dùng đến và cấp phát tự động các tài nguyên này khi nhận được request
            -   Cluster ảo
                -   Quản lý toàn bộ các nhóm VMs
                -   Có lợi cho việc cung cấp các cluster ảo tính toán theo yêu cầu, các VM kết nối trong cho các ứng dụng Internet nhiều mức
            -   Reservation và Negotiation Mechanism
                -   Cho phép người sử dụng đặt trước các tài nguyên trong một thời gian nào đó
                -   Đặc biết có lợi trong các hệ thống cloud mà tài nguyên tại nhiều thời điểm bị thưa ( các cloud nhỏ hoặc không nhận nhiều request)
                -   Nâng cao hiệu năng business
            -   High Availability và Data Recovery
                -   High Availibility (HA): tối thiểu downtime ứng dung và phòng tránh các rủi ro business. Các tính năng được bảo vệ từ host chứ không phải từ VM ( nói một cách đơn giản là tỉ lệ đáp ứng ngay lập tức cao)
                    -   Cơ chế scale trong HA:
                        -   Tăng theo chiều dọc: Tăng tài nguyên cho từng con VM
                        -   Tăng theo chiều ngang: Tăng tài nguyên cho hệ thống bằng cách thêm VM
                -  Data Recovery, Backup
##  VIII.   Thách thức và rủi ro
    -   Bảo mật, an toàn và tin cậy
    -   Data Lock-in và chuẩn hóa
    -   Tính sẵn sàng, Fault-Torrance và phục hồi thảm họa
    -   Quản lý tài nguyên và hiệu suất năng lượng
    => Vấn đề được đặt ra: tối ưu hiệu năng ứng dụng, tối thiểu năng lượng tiêu hao.
   [Mindmap]: <https://mm.tt/879222167?t=DdJr89UqHr>
