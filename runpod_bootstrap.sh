#!/bin/bash
set -e

# Configuration
VOLUME_PATH="/runpod-volume/zimage-diffusion"
INSTALL_FLAG="$VOLUME_PATH/.installed_v3" # Bump version to re-trigger
LOG_FILE="$VOLUME_PATH/bootstrap.log"

export HF_HOME="${HF_HOME:-/runpod-volume/huggingface}"
export UPSCALE_MODEL_URL="${UPSCALE_MODEL_URL:-https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-RealPLSKR.pth}"
export UPSCALE_MODEL_PATH="${UPSCALE_MODEL_PATH:-/runpod-volume/zimage-diffusion/models/upscale/4xPurePhoto-RealPLSKR.pth}"
mkdir -p "$VOLUME_PATH"
mkdir -p "$HF_HOME"
mkdir -p "$(dirname "$UPSCALE_MODEL_PATH")"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- Bootstrap started at $(date) ---"

if [ ! -f "$INSTALL_FLAG" ]; then
    echo "First start with new optimized image. Caching models..."

    # Ensure diffusers >= 0.37.1 which includes Z-Image LoRA context_refiner fix (PR #13209).
    # This runs at container startup to guarantee the right version regardless of Docker layer cache.
    echo "Upgrading diffusers to ensure Z-Image LoRA fix is present..."
    pip install --upgrade "diffusers==0.37.1" --break-system-packages

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

# Ensure the upscaler checkpoint exists on volume even when base install is already complete.
if [ ! -f "$UPSCALE_MODEL_PATH" ]; then
    echo "Caching upscaler model to volume: $UPSCALE_MODEL_PATH"
    curl -L --fail --retry 3 --retry-delay 2 "$UPSCALE_MODEL_URL" -o "$UPSCALE_MODEL_PATH"
else
    echo "Upscaler model already cached: $UPSCALE_MODEL_PATH"
fi

# Ensure the famegridZIB checkpoint exists on volume even when base install is already complete.
FAMEGRID_CHECKPOINT_PATH="/runpod-volume/zimage-diffusion/models/checkpoints/famegridZIB_v10.safetensors"
mkdir -p "$(dirname "$FAMEGRID_CHECKPOINT_PATH")"
if [ ! -f "$FAMEGRID_CHECKPOINT_PATH" ]; then
    if [ -n "$CIVITAI_TOKEN" ]; then
        echo "Downloading famegridZIB_v10 checkpoint to volume: $FAMEGRID_CHECKPOINT_PATH"
        wget "https://civitai.com/api/download/models/2847800?token=${CIVITAI_TOKEN}" \
            -O "${FAMEGRID_CHECKPOINT_PATH}.tmp" \
            --tries=3 \
            --show-progress && \
            mv "${FAMEGRID_CHECKPOINT_PATH}.tmp" "$FAMEGRID_CHECKPOINT_PATH" || \
            echo "WARNING: famegridZIB_v10 download failed — check CIVITAI_TOKEN and connectivity."
    else
        echo "WARNING: CIVITAI_TOKEN not set, skipping famegridZIB_v10 checkpoint download."
    fi
else
    echo "famegridZIB_v10 checkpoint already cached: $FAMEGRID_CHECKPOINT_PATH"
fi

echo "Starting RunPod Handler..."
exec python3 handler.py
