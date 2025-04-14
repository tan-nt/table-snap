# Base image: PyTorch + CUDA 11.8 + cuDNN 8 (for GPU support)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy source code into the container
COPY . /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python requirements
RUN pip install --upgrade pip

# Install your base Python packages (adjust as needed)
RUN pip install \
    timm \
    opencv-python \
    numpy \
    pandas \
    scipy \
    matplotlib

# ✅ Install mmcv-full (compiled for torch 2.1.0 + CUDA 11.8)
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# Optional: set env for multithreaded CPU ops if used
ENV OMP_NUM_THREADS=8

# Set default training command (can override)
CMD ["python", "runner/train.py", "--cfg", "default"]
