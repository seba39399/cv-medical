# Official PyTorch base image with CUDA 12.1 and cuDNN 9 support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data/COBRE_Dataset

# Set the working directory inside the container
WORKDIR /app

# Install minimal system dependencies for scientific and neuroimaging packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the cloud training script and shell automation scripts
COPY train_cloud.py .
COPY shell_scripts/ ./shell_scripts/

# Grant execution permissions to shell scripts
RUN chmod +x ./shell_scripts/*.sh

# Default command to execute when launching the container
CMD ["python3", "train_cloud.py"]