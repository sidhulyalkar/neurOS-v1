# NeuroFM-X Training Container
# Optimized for AWS A100 instances

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace/neurofm

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install additional ML libraries
RUN pip3 install --no-cache-dir \
    tensorboard \
    wandb \
    scikit-learn \
    scipy \
    matplotlib \
    seaborn \
    pyyaml \
    tqdm

# Install neuroscience-specific libraries
RUN pip3 install --no-cache-dir \
    allensdk \
    h5py \
    pynwb

# Install Mamba dependencies (if using mamba-ssm)
RUN pip3 install --no-cache-dir \
    causal-conv1d>=1.1.0 \
    mamba-ssm

# Copy project files
COPY . .

# Install package in development mode
RUN pip3 install -e .

# Create directories for data and checkpoints
RUN mkdir -p /data /checkpoints /logs

# Set up entry point
ENTRYPOINT ["python3", "training/train.py"]
