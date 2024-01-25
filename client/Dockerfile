FROM python:3.10.11-slim-buster 

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/opt

RUN apt-get update 

COPY ./requirements.txt ${ROOT}/

RUN pip install -r ${ROOT}/requirements.txt
