# Use the specified RunPod base image
FROM runpod/base:1.0.3-cuda1281-ubuntu2404

# Set shell for bash
SHELL ["/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
# We use --break-system-packages because this is a dedicated container environment
COPY requirements.txt .

# 1. Install heavy AI frameworks first (from specific torch index)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --break-system-packages

# 2. Install remaining utilities from standard PyPI
RUN pip install --no-cache-dir runpod boto3 requests pillow diffusers transformers accelerate safetensors peft scipy realesrgan --break-system-packages

# Copy scripts into the container
COPY runpod_bootstrap.sh .
COPY handler.py .
COPY s3_utils.py .

# Make the bootstrap script executable
RUN chmod +x runpod_bootstrap.sh

# Environment variables (Can be overridden in RunPod)
ENV MODEL_ID="Tongyi-MAI/Z-Image"
ENV HF_HOME="/runpod-volume/huggingface"
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1

# Start the bootstrap script
CMD ["./runpod_bootstrap.sh"]
