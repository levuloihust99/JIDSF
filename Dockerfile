FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /app

COPY . .

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y python3 python3-venv python3-dev software-properties-common

RUN python3 -m venv .venv
RUN source .venv/bin/activate && \
    pip install -U pip && pip install -r requirements.txt
