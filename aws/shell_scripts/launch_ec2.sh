#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- Launching EC2 GPU Instance via AWS CLI ---"

aws ec2 run-instances \
    --image-id ami-00d34075413e2a6c5 \
    --instance-type g4dn.xlarge \
    --key-name cobre-gpu-key \
    --count 1 \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cobre-cnn-gpu}]'

echo "--- Instance launch command executed successfully, can start ---"