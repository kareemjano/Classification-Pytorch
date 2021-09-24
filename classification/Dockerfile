FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3.7 python3.7-dev python3-pip

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

#COPY ./src /app/src

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV PYTHONIOENCODING=utf-8

#ENTRYPOINT [ "python3" ]
#CMD [ "python3", "run.py"]