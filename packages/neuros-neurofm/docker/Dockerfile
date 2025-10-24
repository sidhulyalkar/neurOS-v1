# NeuroFMx Training Container
# Optimized for H100 HGX with PyTorch 2.4+, CUDA 12.4, Mamba SSM, FlashAttention

FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    nvtop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Mamba SSM (optimized CUDA kernels)
RUN pip install --no-cache-dir mamba-ssm causal-conv1d

# Install FlashAttention 2 (for Perceiver-IO)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install xFormers (additional optimizations)
RUN pip install --no-cache-dir xformers

# Install Ray for distributed training
RUN pip install --no-cache-dir "ray[default,data,train]==2.34.0"

# Install cloud storage clients
RUN pip install --no-cache-dir boto3 s3fs

# Install experiment tracking
RUN pip install --no-cache-dir wandb tensorboard

# Install neuroscience data tools
RUN pip install --no-cache-dir \
    allensdk \
    mne \
    nilearn \
    nibabel \
    pynwb \
    dandi

# Copy NeuroFMx codebase
COPY packages/neuros-neurofm /workspace

# Install NeuroFMx package
RUN cd /workspace && pip install -e .

# Set environment variables for optimal H100 performance
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_NTHREADS=4
ENV NCCL_NSOCKS_PERTHREAD=8
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose Ray ports
EXPOSE 6379 8265 10001

# Default command
CMD ["/bin/bash"]
