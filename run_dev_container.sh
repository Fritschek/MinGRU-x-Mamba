#!/bin/bash

# Start Docker service if not running
if ! systemctl is-active --quiet docker; then
    echo "🔧 Starting Docker..."
    sudo systemctl start docker
fi

# Run the NVIDIA PyTorch container with workspace mounted
echo "🚀 Launching NVIDIA PyTorch Docker container..."
docker run --gpus all -it --rm \
    -v $HOME/GitHub/MinGRU-x-Mamba:/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:23.06-py3
