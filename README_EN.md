# AIじゃんけんマシン for JETSON NANO
## Introduction
---
NVIDIA® Jetson Nano™上で、MediaPipeとPyTorchを使って制作したAIじゃんけんマシンです。

## 使用機器
---
- NVIDIA® Jetson Nano™
- Logicool Webcamera C505

## AIフレームワーク
---
- [MediaPipe](https://google.github.io/mediapipe/)
- [PyTorch](https://pytorch.org/)

## インストール
---
~~~
sudo apt update
sudo apt install python3-pip
sudo apt install curl

sudo apt update && \
sudo apt-get install -y \
build-essential cmake git unzip pkg-config \
libjpeg-dev libpng-dev libgtk2.0-dev \
python3-dev python3-numpy python3-pip \
libxvidcore-dev libx264-dev libssl-dev \
libtbb2 libtbb-dev libdc1394-22-dev \
gstreamer1.0-tools libv4l-dev v4l-utils \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
libvorbis-dev libxine2-dev \
libfaac-dev libmp3lame-dev libtheora-dev \
libopencore-amrnb-dev libopencore-amrwb-dev \
libopenblas-dev libatlas-base-dev libblas-dev \
liblapack-dev libeigen3-dev \
libhdf5-dev protobuf-compiler \
libprotobuf-dev libgoogle-glog-dev libgflags-dev \
libavutil55=7:3.4.2-2 libavutil-dev libavcodec-dev \
libavformat-dev libswscale-dev ffmpeg

sudo pip3 install pip --upgrade

sudo pip3 install opencv_contrib_python

git clone https://github.com/PINTO0309/mediapipe-bin && cd mediapipe-bin

./v0.8.5/numpy119x/mediapipe-0.8.5_cuda102-cp36-cp36m-
linux_aarch64_numpy119x_jetsonnano_L4T32.5.1_download.sh

sudo pip3 install \
numpy-1.19.4-cp36-none-manylinux2014_aarch64.whl \
mediapipe-0.8.5_cuda102-cp36-none-linux_aarch64.whl

sudo pip3 install opencv-python dataclasses
~~~

## ソースコード
- 
- 
- 

## コラボレータ
---
- 
- 
- 
## 動画リンク
---
- 
- 
- 


## 参考リンク
---
https://github.com/PINTO0309/mediapipe-bin