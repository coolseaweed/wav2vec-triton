FROM python:3.10.11-slim-buster 

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/opt \
    WORKDIR=/workspace

RUN apt-get update 

COPY ./requirements.txt ${ROOT}/


WORKDIR ${WORKDIR}

COPY . ${WORKDIR}


RUN pip install -r ${ROOT}/requirements.txt
