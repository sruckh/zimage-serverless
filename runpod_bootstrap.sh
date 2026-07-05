#!/bin/bash
set -e

# Configuration
VOLUME_PATH="/runpod-volume/zimage-diffusion"
INSTALL_FLAG="$VOLUME_PATH/.installed_v5" # Bump version to re-trigger
LOG_FILE="$VOLUME_PATH/bootstrap.log"

export HF_HOME="${HF_HOME:-/runpod-volume/huggingface}"
UPSCALE_DIR="${UPSCALE_DIR:-/runpod-volume/zimage-diffusion/models/upscale}"
mkdir -p "$VOLUME_PATH"
mkdir -p "$HF_HOME"
mkdir -p "$UPSCALE_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- Bootstrap started at $(date) ---"

if [ ! -f "$INSTALL_FLAG" ]; then
    echo "First start with new optimized image. Caching models..."

    # diffusers and Flash Attention are now baked into the image at build time
    # (see Dockerfile) rather than installed here. A runtime install gated by
    # this flag on the persistent network volume was unreliable for packages:
    # pip installs land in the container's own ephemeral site-packages, but
    # this flag lives on the shared, persistent volume -- so a new worker
    # container cold-starting after an earlier worker already touched this
    # flag would silently skip reinstalling them and run without Flash
    # Attention. Model caching below is genuinely volume-cacheable (the
    # downloaded weights themselves live under $HF_HOME on the volume), so it
    # stays gated here.
    export MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image}"
    echo "Pre-caching model: $MODEL_ID..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"

    touch "$INSTALL_FLAG"
else
    echo "Environment already optimized."
fi

# Pre-stage the default upscaler to the volume for a fast first request. Any other
# model in the handler's registry (UPSCALE_MODELS) downloads lazily on first use.
DEFAULT_UPSCALE_URL="https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth"
DEFAULT_UPSCALE_PATH="$UPSCALE_DIR/4xNomosWebPhoto_RealPLKSR.pth"
if [ ! -f "$DEFAULT_UPSCALE_PATH" ]; then
    echo "Pre-staging default upscaler to volume: $DEFAULT_UPSCALE_PATH"
    curl -L --fail --retry 3 --retry-delay 2 "$DEFAULT_UPSCALE_URL" -o "$DEFAULT_UPSCALE_PATH"
else
    echo "Default upscaler already cached: $DEFAULT_UPSCALE_PATH"
fi

# Ensure the famegridZIB checkpoint exists on volume even when base install is already complete.
# Set USE_CIVITAI_CHECKPOINT=false (also read by handler.py) to skip this entirely and run
# stock Tongyi-MAI/Z-Image base weights instead -- e.g. to test LoRA compatibility, since a
# LoRA trained against stock Base may bind more weakly against an already-finetuned checkpoint.
# Matches the same falsy aliases as handler.py's _to_bool (case-insensitive
# false/0/no/off), so the two can't disagree about whether the checkpoint is in use.
case "${USE_CIVITAI_CHECKPOINT:-true}" in
    [Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]|[Oo][Ff][Ff])
        echo "USE_CIVITAI_CHECKPOINT=false, skipping famegridZIB_v10 checkpoint download."
        ;;
    *)
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
        ;;
esac

# Ensure spandrel is installed (used by handler.py to load the upscaler models).
# Runs on every start, outside the first-run install gate, so workers provisioned
# before spandrel was added get it without requiring an image rebuild. Idempotent:
# the import check short-circuits when it is already present.
if ! python3 -c "import spandrel" 2>/dev/null; then
    echo "Installing spandrel (upscaler loader)..."
    pip install spandrel --break-system-packages
fi

echo "Starting RunPod Handler..."
exec python3 handler.py
