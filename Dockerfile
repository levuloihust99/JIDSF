FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt .

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y --no-install-recommends python3 python3-venv python3-dev software-properties-common

RUN python3 -m venv .venv
RUN . .venv/bin/activate; pip install -U pip; pip install --no-deps -r requirements.txt
COPY . .

ENTRYPOINT [ "bash" ]