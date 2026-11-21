# Official PyTorch base image with CUDA 12.1 and cuDNN 9 support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data/COBRE_Dataset \
    PATH="/home/appuser/.local/bin:${PATH}"

# Set the working directory inside the container
WORKDIR /app

# Install minimal system dependencies and clean up apt lists in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and prepare mount point for neuroimaging data
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data/COBRE_Dataset && \
    chown -R appuser:appuser /app /data/COBRE_Dataset

# Copy requirements first to leverage Docker layer caching
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the cloud training script and shell automation scripts
COPY --chown=appuser:appuser train_cloud.py .
COPY --chown=appuser:appuser shell_scripts/ ./shell_scripts/

# Grant execution permissions to shell scripts
RUN chmod +x ./shell_scripts/*.sh

# Switch to non-root user for runtime security
USER appuser

# Default command to execute when launching the container
CMD ["python3", "train_cloud.py"]