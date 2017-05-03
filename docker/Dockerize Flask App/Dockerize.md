# Dockerize Flask Application

## Tạo cây thư mục
```
web 
    ___ app.py
    ___ Dockerfile
    ___ requirements.txt
```
Trong đó:
- app.py: file chứa source code của ứng dụng
- requirements.txt: chứa tên các thư viện cần cài đặt thêm
- Dockerfile: chứa nội dung cần thiết để tạo Docker image

## Nội dung các file

### File app.py
``` python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
```

### File requirements.txt
```bash 
Flask==0.12
```

### Dockerfile
```Dockerfile
# Format: FROM    repository[:version]
# File image gốc, ở đây là ubuntu 
FROM       ubuntu:latest

# Thực hiện cài đặt python 
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

# Copy các file vào thư mục app, cái đặt môi trường làm việc
COPY . /app
WORKDIR /app

# Cài đặt thư viện cần thiết
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]
```

## Tạo docker image
```bash 
docker build -t helloworldapp:latest . 
```

## Chạy Docker container 
```bash 
docker run -d -p 5000:5000 helloworldapp
```
**Kiểm tra container**
```bash
docker ps
```