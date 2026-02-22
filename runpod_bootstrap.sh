#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Configuration
VOLUME_PATH="/runpod-volume/zimage-diffusion"
VENV_PATH="$VOLUME_PATH/venv"
INSTALL_FLAG="$VOLUME_PATH/.installed"
LOG_FILE="$VOLUME_PATH/bootstrap.log"

# Default HF_HOME if not set in Dockerfile/RunPod
export HF_HOME="${HF_HOME:-/runpod-volume/huggingface}"

# Ensure directories exist
mkdir -p "$VOLUME_PATH"
mkdir -p "$HF_HOME"

# Redirect stdout and stderr to the log file and also to the console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- Bootstrap started at $(date) ---"

if [ ! -f "$INSTALL_FLAG" ]; then
    echo "First cold start. Installing software to $VOLUME_PATH..."

    # Create Python 3.12 Virtual Environment
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating venv at $VENV_PATH..."
        python3.12 -m venv "$VENV_PATH"
    fi
    
    # Activate venv
    echo "Activating venv..."
    source "$VENV_PATH/bin/activate"

    # Install Core Dependencies
    echo "Installing PyTorch 2.8.0..."
    pip install --upgrade pip
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    
    # Install Diffusion & Utilities
    echo "Installing diffusers and utilities..."
    pip install diffusers transformers accelerate safetensors boto3 runpod requests pillow
    
    # Install Flash Attention
    echo "Installing Flash Attention 2.8.3..."
    FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    pip install "$FLASH_ATTN_URL"

    # Export model ID if not set (needed for hf download)
    export MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image}"

    # Pre-cache the model using the new 'hf' CLI for faster cold starts
    echo "Pre-caching model: $MODEL_ID..."
    if [ -n "$HF_TOKEN" ]; then
        hf download "$MODEL_ID" --token "$HF_TOKEN"
    else
        hf download "$MODEL_ID"
    fi

    # Sanity Checks
    echo "Running sanity checks..."
    python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"

    # Set the installed flag
    touch "$INSTALL_FLAG"
    echo "Installation complete."
else
    echo "Software already installed (flag found at $INSTALL_FLAG). Skipping installation."
    source "$VENV_PATH/bin/activate"
fi

# Export model ID if not set
export MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image}"

# Ensure HF_TOKEN is logged as present (but not the value)
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN is set."
else
    echo "HF_TOKEN is not set (optional)."
fi

echo "Starting RunPod Handler..."
# Use exec to replace the shell process with the python process
exec python handler.py
