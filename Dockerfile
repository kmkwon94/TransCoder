FROM kmkwon94/transcodermodels AS build
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    gfortran \
    curl \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    vim \
    wget \
    locales \
    libgl1-mesa-glx \
    libssl-dev \ 
    libpcre3 \
    libpcre3-dev \ 
    python3 \
    python3-pip \ 
    python3-dev \ 
    build-essential \
    sudo \
    clang \ 
 && rm -rf /var/lib/apt/lists/*

#Set environment of Cuda 10.1
#ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#ENV CUDA_HOME=/usr/local/cuda

ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing" 
RUN export FORCE_CUDA="1"
RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN ["python3", "-m", "pip", "install", "-r", "requirements.txt"]
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /apex
RUN pip3 install -v --no-cache-dir ./
WORKDIR /TransCoder
COPY . /TransCoder
WORKDIR /TransCoder/XLM/tools
RUN git clone https://github.com/glample/fastBPE 
WORKDIR /TransCoder/XLM/tools/fastBPE
RUN python3 setup.py install
RUN mv fastBPE /TransCoder
WORKDIR /usr/lib/llvm-6.0/lib
RUN mv libclang.so.1 libclang.so
WORKDIR /TransCoder
COPY --from=build /root/checkpoints /TransCoder/checkpoints
CMD python3 server.py