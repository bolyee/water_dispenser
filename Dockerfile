# Base image with Python 3.10 and PyTorch 2.1.0 pre-installed with CUDA 12.1 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory inside container
WORKDIR /app

# Install system dependencies for audio (libsndfile) and computer vision (OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (Base image already has torch, torchaudio, and torchvision.
# Using --no-deps or letting pip check will keep the pre-installed CUDA PyTorch version)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (interactive bash)
CMD ["bash"]
