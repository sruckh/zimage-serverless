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
    
    # Flash Attention - use specific wheel to avoid 30min compilation
    echo "Installing Flash Attention from pre-built wheel..."
    FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    pip install "$FLASH_ATTN_URL" --break-system-packages || echo "Flash Attention wheel failed, falling back to source (slow)..." && pip install flash-attn --break-system-packages

    export MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image}"
    echo "Pre-caching model: $MODEL_ID..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"

    touch "$INSTALL_FLAG"
else
    echo "Environment already optimized."
fi

echo "Starting RunPod Handler..."
exec python3 handler.py
