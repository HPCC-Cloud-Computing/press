# Format: FROM    repository[:version]
FROM       ubuntu:16.04
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app

ENTRYPOINT ["python3"]
CMD ["app.py"]

