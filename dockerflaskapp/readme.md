Dockerized Flask App
======
Required Sofware
------
* Docker (1.6.0 or above)
* Python ( 2.7 or above )
* Linux (Ubuntu 16.04 Xenial)

Create the Requirements File
------
* State the software required to be installed in the container.
* Create a file **requirements.txt** in folder:
> Flask==0.10.1
>

Create Flask App
------
* Create file **app.py** in folder:
```python
from flask import Flask 

app = Flask(__name__)

@app.route('/')

def hello_world():
    return 'Flask Dockerized'

if __name__ = "__main__":
    app.run(debug=True, host='0.0.0.0')
```

Create Dockerfile
------
* Need to create a docker image and deploy it
```Dockerfile
# Format: FROM    repository[:version]
FROM ubuntu:16.04

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]

```

Build the Docker Image 
------
* Run the following command to build docker image **demo** from directory:
``` sh
 $ docker build -t demo:latest .
```
* Check the Docker Image:
``` sh
 $ docker images 
```
* Docker Images list:
``` sh
REPOSITORY  TAG     IMAGE ID        CREATED         SIZE
demo        latest  55fe95ab08cc    4 seconds ago   445MB

```
Run the Docker Container
------
* Run the following command to run Docker container:
``` sh
docker run -d -p 5000:5000 demo
```
* Check the Docker Container:
``` sh
docker ps -a
```
* Docker Container List:
``` sh
CONTAINER ID    IMAGE       COMMAND             CREATED         STATUS          PORTS                   NAMES
3d8c1a57d962    demo:latest "python3 app.py"    8 seconds ago   Up 7 seconds    0.0.0.0:5000->5000/tcp  serene_allen
```