#!/bin/bash

# Configuration
VOLUME_PATH="/runpod-volume/zimage-diffusion"
VENV_PATH="$VOLUME_PATH/venv"
INSTALL_FLAG="$VOLUME_PATH/.installed"

# Ensure the volume directory exists
mkdir -p "$VOLUME_PATH"

if [ ! -f "$INSTALL_FLAG" ]; then
    echo "First cold start. Installing software to $VOLUME_PATH..."

    # Create Python 3.12 Virtual Environment
    python3.12 -m venv "$VENV_PATH"
    
    # Activate venv
    source "$VENV_PATH/bin/activate"

    # Install Core Dependencies
    pip install --upgrade pip
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    
    # Install Diffusion & Utilities
    pip install diffusers transformers accelerate safetensors boto3 runpod requests pillow
    
    # Install Flash Attention
    FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    pip install "$FLASH_ATTN_URL"

    # Set the installed flag
    touch "$INSTALL_FLAG"
    echo "Installation complete."
else
    echo "Software already installed. Skipping installation."
fi

# Final check: Activate the venv and start the handler
source "$VENV_PATH/bin/activate"

# Export model ID if not set
export MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image}"

echo "Starting RunPod Handler..."
python handler.py
