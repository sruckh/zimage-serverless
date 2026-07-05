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

# 1. Install heavy AI frameworks first (from specific torch index).
# Pinned to match the flash-attn wheel runpod_bootstrap.sh installs, which is
# built specifically against the torch2.8 C++ ABI
# (flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-*.whl). An unpinned install here
# previously drifted to whatever the cu128 index's latest release was at
# image-build time, breaking that wheel's ABI at import with "undefined
# symbol" errors from libtorch's C10 CUDA layer. These versions already
# matched what requirements.txt documented (see note there) but were never
# actually applied to the real install command until now.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --break-system-packages

# 2a. Pre-install cryptography in isolation with --ignore-installed. The base
# image ships cryptography 41.0.7 via apt/dpkg (no pip RECORD file), and
# runpod's own dependency chain (-> paramiko -> cryptography>=48.0.1) makes
# the main install below try to upgrade it, which fails with "error:
# uninstall-no-record-file" since pip can't verify what to remove from a
# non-pip-managed install. This is pip's own documented remedy
# (https://github.com/pypa/pip/issues/12645). Scoped to its own command since
# --ignore-installed is a whole-command flag, not a per-package one -- doing
# it here keeps the main install below from ignoring already-satisfied
# packages it shouldn't (numpy, sympy, jinja2, etc. already in the base image).
RUN pip install --no-cache-dir --ignore-installed "cryptography>=48.0.1" --break-system-packages

# 2b. Install remaining utilities from standard PyPI
RUN pip install --no-cache-dir runpod boto3 requests pillow "diffusers==0.37.1" transformers accelerate safetensors peft scipy spandrel --break-system-packages

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
