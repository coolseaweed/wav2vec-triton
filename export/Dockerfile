FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/opt \
    WORKDIR=/workspace

RUN apt-get update && apt-get -y install \
    wget

COPY ./requirements.txt ${ROOT}/

WORKDIR ${WORKDIR}

COPY run.py ${WORKDIR}

RUN pip install -r ${ROOT}/requirements.txt
