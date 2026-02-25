# Z-Image RunPod Serverless Worker

This project implements a RunPod serverless worker for the **Z-Image** base model. it supports dynamic LoRA loading from external URLs (e.g., Backblaze B2 presigned URLs) and uploads high-quality JPG outputs back to S3-compatible storage.

## Features

- **High-Performance Image:** Core dependencies are pre-baked into the Docker image for near-instant startup (<20s imports).
- **Persistent Volume Support:** Model weights are cached on `/runpod-volume/huggingface` to avoid re-downloading.
- **Z-Image Aligned Defaults:** Uses 50 steps with base-model aligned CFG defaults (`cfg_normalization=False`, `cfg_truncation=1.0`).
- **Scheduler Control:** Supports `use_beta_sigmas` to toggle FlowMatch beta-sigma scheduling.
- **Adaptive VAE Tiling:** Keeps VAE tiling off at 1024-ish outputs by default to reduce potential tile artifacts, while enabling it for larger images.
- **Optional Two-Pass Refinement:** Upscales pass-1 output with RealESRGAN (`4xPurePhoto-RealPLSKR`) and runs a Z-Image img2img refinement pass for extra detail.
- **Dynamic LoRA Support:** Load LoRAs from any URL at runtime with automatic cleanup.
- **S3 Integration:** Automatically uploads generated images to an S3-compatible bucket (configured for Backblaze B2).

## Environment Variables (RunPod Configuration)

Configure these variables in your RunPod Endpoint/Template:

| Variable | Description | Default |
|----------|-------------|---------|
| `S3_ENDPOINT_URL` | **Required.** Your S3/B2 endpoint (e.g., `https://s3.us-west-004.backblazeb2.com`) | - |
| `S3_ACCESS_KEY_ID` | **Required.** S3 Access Key ID | - |
| `S3_SECRET_ACCESS_KEY` | **Required.** S3 Secret Access Key | - |
| `S3_BUCKET_NAME` | **Required.** Name of the bucket to upload images to | - |
| `S3_BASE_URL` | Optional. Base URL for the returned link. Defaults to `endpoint/bucket` | - |
| `MODEL_ID` | HuggingFace Repo ID or local path | `Tongyi-MAI/Z-Image` |
| `HF_TOKEN` | Optional. Hugging Face token for private models. | - |
| `SECOND_PASS_DEFAULT_ENABLED` | Optional. Enables two-pass refinement by default. | `true` |
| `UPSCALE_MODEL_URL` | Optional. URL for the RealESRGAN `.pth` model. | Starinspace 4xPurePhoto-RealPLSKR |
| `UPSCALE_MODEL_PATH` | Optional. Cached path for the upscaler model. | `/runpod-volume/zimage-diffusion/models/upscale/4xPurePhoto-RealPLSKR.pth` |

The bootstrap script now checks `UPSCALE_MODEL_PATH` on every start and downloads it once if missing, even when `.installed_v2` already exists.

## Monitoring & Debugging

On the first cold start, you can monitor the installation progress in the **RunPod logs** or by checking the persistent log file:
`/runpod-volume/zimage-diffusion/bootstrap.log`

## Endpoint Input Parameters

When making a call to the `/run` or `/runsync` endpoint, use the following JSON structure in the `input` field:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | String | **Yes** | - | The text prompt for generation. |
| `lora_url` | String | No | - | URL to the `.safetensors` LoRA file. |
| `negative_prompt` | String | No | `""` | Text to avoid in the generation. |
| `width` | Integer | No | `1024` | Image width. |
| `height` | Integer | No | `1024` | Image height. |
| `steps` | Integer | No | `50` | Number of inference steps. |
| `guidance_scale` | Float | No | `3.0` | CFG scale (3.0-4.5 recommended for likeness). |
| `cfg_normalization`| Boolean | No | `False` | Z-Image base default behavior. |
| `cfg_truncation` | Float | No | `1.0` | 1.0 recommended; lower to fix over-saturation. |
| `max_sequence_length`| Integer | No | `512` | Token limit for long prompts. |
| `seed` | Integer | No | `42` | Random seed for reproducibility. |
| `lora_scale` | Float | No | `0.85` | Strength of the LoRA adapter (0.8-0.9 recommended). |
| `use_beta_sigmas` | Boolean | No | `True` | Rebuilds FlowMatch scheduler with beta sigmas for cleaner denoising. |
| `vae_tiling` | Boolean | No | auto | Override adaptive VAE tiling behavior (`auto`: on only for >1024×1024 area). |
| `second_pass_enabled` | Boolean | No | env/default | Enables pass-2 upscale + img2img refinement. |
| `second_pass_upscale` | Float | No | `2.0` | RealESRGAN outscale factor for pass 2 (2.0 = 2x). |
| `second_pass_strength` | Float | No | `0.3` | Img2img strength for pass 2. |
| `second_pass_steps` | Integer | No | `10` | Img2img denoising steps for pass 2. |
| `second_pass_guidance_scale` | Float | No | `2.8` | CFG scale for pass 2. |
| `second_pass_seed` | Integer | No | `seed` | Seed for pass 2 reproducibility. |
| `second_pass_cfg_normalization` | Boolean | No | `False` | CFG normalization toggle for pass 2. |
| `second_pass_cfg_truncation` | Float | No | `1.0` | CFG truncation for pass 2. |
| `second_pass_use_beta_sigmas` | Boolean | No | `use_beta_sigmas` | Scheduler beta-sigma toggle for pass 2. |
| `second_pass_vae_tiling` | Boolean | No | auto | Override adaptive VAE tiling for pass 2. |
| `adapter_name` | String | No | - | Unique ID (auto-generated if not provided). |

### Example Request Body

```json
{
  "input": {
    "prompt": "A professional portrait of K1mScum in a futuristic setting",
    "lora_url": "https://f004.backblazeb2.com/file/my-bucket/K1mScum.safetensors?params...",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 4.5,
    "seed": 12345
  }
}
```

## Setup & Deployment

1. **Build the Image:**
   ```bash
   docker build -t your-registry/zimage-serverless:latest .
   ```
2. **Push to Registry:**
   ```bash
   docker push your-registry/zimage-serverless:latest
   ```
3. **Configure RunPod:**
   - Create a new Serverless Template using the image.
   - Attach a **Network Volume** to `/runpod-volume`.
   - Set the necessary Environment Variables.
   - Deploy the endpoint.

## Project Structure

- `Dockerfile`: Minimal base image configuration.
- `runpod_bootstrap.sh`: Handles persistent software installation and venv activation.
- `handler.py`: The RunPod serverless worker logic.
- `s3_utils.py`: Utility for S3/B2 uploads.
- `.gitignore`: Prevents local artifacts and secrets from being committed.
