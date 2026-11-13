#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- [1/3] Verifying GPU availability ---"
nvidia-smi

echo "--- [2/3] Setting environment variables ---"
export DATA_DIR="../COBRE_Dataset"
# O si prefieres montarlo desde una ruta absoluta en la instancia:
# export DATA_DIR="/home/ec2-user/COBRE_Dataset"

echo "--- [3/3] Launching Cloud Training Pipeline ---"
python3 ../aws/train_cloud.py

echo "--- Training execution finished successfully! ---"