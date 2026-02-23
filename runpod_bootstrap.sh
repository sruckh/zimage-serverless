#!/bin/bash
set -e

# Configuration
VOLUME_PATH="/runpod-volume/zimage-diffusion"
INSTALL_FLAG="$VOLUME_PATH/.installed_v2" # Bump version to re-trigger
LOG_FILE="$VOLUME_PATH/bootstrap.log"

export HF_HOME="${HF_HOME:-/runpod-volume/huggingface}"
mkdir -p "$VOLUME_PATH"
mkdir -p "$HF_HOME"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- Bootstrap started at $(date) ---"

if [ ! -f "$INSTALL_FLAG" ]; then
    echo "First start with new optimized image. Caching models..."
    
    # We use the system python now, no venv needed for core libs
    # Flash Attention might need a local build if the wheel doesn't match
    echo "Checking Flash Attention..."
    pip install flash-attn --no-build-isolation || echo "Flash Attention build failed, falling back to standard attention."

    export MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image}"
    echo "Pre-caching model: $MODEL_ID..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"

    touch "$INSTALL_FLAG"
else
    echo "Environment already optimized."
fi

echo "Starting RunPod Handler..."
exec python3 handler.py
