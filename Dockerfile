# ────────────────────────────────────────────────
# Base: NVIDIA CUDA 13.0 + cuDNN + Ubuntu 22.04
# PyTorch: 2.11.0 (CUDA 13.0)
# ────────────────────────────────────────────────
FROM nvidia/cuda:13.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ────────────────────────────────────────────────
# System dependencies
# ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ────────────────────────────────────────────────
# PyTorch (CUDA 13.0)
# ────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu130

# ────────────────────────────────────────────────
# Python dependencies
# ────────────────────────────────────────────────
WORKDIR /workspace/riemannian_flow_det

COPY requirements.txt .
RUN pip install \
    pycocotools \
    ConfigArgParse \
    tensorboard \
    matplotlib \
    numpy \
    tqdm \
    scipy \
    Pillow

# ────────────────────────────────────────────────
# Detectron2 (source build — PyTorch 버전에 맞춰 빌드 필요)
# ────────────────────────────────────────────────
RUN pip install \
    fvcore \
    iopath \
    omegaconf \
    hydra-core \
    antlr4-python3-runtime==4.9.3 && \
    pip install git+https://github.com/facebookresearch/detectron2.git

# ────────────────────────────────────────────────
# Project code
# ────────────────────────────────────────────────
COPY . .

# data 및 outputs 마운트 포인트
RUN mkdir -p data/voc data/coco outputs/checkpoints outputs/logs outputs/figures

# ────────────────────────────────────────────────
# TensorBoard port
# ────────────────────────────────────────────────
EXPOSE 6006

CMD ["bash"]
