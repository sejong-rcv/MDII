FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
LABEL maintainer "Daechan Han <dchan@rcv.sejong.ac.kr>"

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    curl \
    git \
    libjpeg-dev \
    libpng-dev \
    unzip \
    wget

RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

RUN conda update -y conda && \
    conda create -n mdii python=3.7

ENV PATH /usr/local/envs/mdii/bin:$PATH
RUN echo "source activate mdii" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

COPY requirements.txt /tmp/requirements.txt
RUN source activate mdii && pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
