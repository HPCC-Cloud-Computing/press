# Cài đặt MariaDB trên server bằng Ansible

Server: CentOS 6
Các bước cài đặt:
- Cài đặt Ansible
- Cấu hình tên server, IP trong file `/etc/ansible/hosts`
``` bash
[vm]
192.168.20.202 ansible_user=root
```
- Chạy lệnh: `ansible-playbook -s mariadb.yml`