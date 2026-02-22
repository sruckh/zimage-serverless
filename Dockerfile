# Use the specified RunPod base image
FROM runpod/base:1.0.3-cuda1281-ubuntu2404

# Set shell for bash
SHELL ["/bin/bash", "-c"]

# Set working directory
WORKDIR /app

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
