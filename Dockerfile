# ────────────────────────────────────────────────
# Base: NVIDIA CUDA 12.8.1 + cuDNN + Ubuntu 22.04
# PyTorch: 최신 (CUDA 12.8)
# ────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

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
# PyTorch (CUDA 12.8)
# ────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

# ────────────────────────────────────────────────
# Python dependencies
# ────────────────────────────────────────────────
WORKDIR /workspace/riemannian_flow_det

COPY requirements.txt .
RUN pip install -r requirements.txt

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
