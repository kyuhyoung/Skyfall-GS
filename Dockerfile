FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    gdal-bin libgdal-dev \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.8)
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install CLIP first (needs --no-build-isolation for pkg_resources)
RUN pip install --no-cache-dir --no-build-isolation \
    "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

# Install remaining requirements (skip CLIP line, already installed above)
COPY requirements.txt /tmp/requirements.txt
RUN grep -v "^clip @" /tmp/requirements.txt > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir rasterio sentencepiece && \
    rm /tmp/requirements.txt /tmp/requirements_filtered.txt

# CUDA architectures for building extensions (no GPU available during docker build)
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"
ENV FORCE_CUDA=1

# Install submodules (CUDA extensions)
COPY submodules /tmp/submodules
RUN pip install --no-cache-dir --no-build-isolation /tmp/submodules/diff-gaussian-rasterization-depth && \
    pip install --no-cache-dir --no-build-isolation /tmp/submodules/simple-knn && \
    pip install --no-cache-dir --no-build-isolation /tmp/submodules/fused-ssim && \
    rm -rf /tmp/submodules

WORKDIR /workspace

CMD ["/bin/bash"]
