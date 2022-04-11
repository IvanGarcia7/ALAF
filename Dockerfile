FROM nvidia/cuda:10.0-base
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget
   
    
RUN apt-get install -y git python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu
RUN apt-get install protobuf-compiler python-pil python-lxml -y
RUN pip3 install jupyter
RUN pip3 install matplotlib
RUN pip3 install wget
RUN pip3 install dict2xml


RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
COPY --chown=tensorflow . /home/tensorflow/models

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN pip3 install -U pip
RUN pip3 install .

RUN pip3 install jupyter
RUN pip3 install opencv-python
RUN pip3 install opencv-contrib-python

RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

ENV TF_CPP_MIN_LOG_LEVEL 3




