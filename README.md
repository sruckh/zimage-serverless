# Z-Image RunPod Serverless Worker

This project implements a RunPod serverless worker for the **Z-Image** base model. it supports dynamic LoRA loading from external URLs (e.g., Backblaze B2 presigned URLs) and uploads high-quality JPG outputs back to S3-compatible storage.

## Features

- **Minimal Container:** Heavy dependencies (PyTorch, Flash Attention) are installed at runtime on the first cold start to keep the image small.
- **Persistent Volume Support:** Software is installed into a virtual environment on `/runpod-volume/zimage-diffusion`. A `.installed` flag ensures installation happens only once.
- **Optimized Defaults:** Uses optimized parameters (50 steps, 3.0 CFG) and VAE tiling for professional-quality realism.
- **Dynamic LoRA Support:** Load LoRAs from any URL at runtime. LoRAs are downloaded to ephemeral storage and cleaned up after each job.
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
| `cfg_normalization`| Boolean | No | `True` | Set to True for realistic skin textures. |
| `cfg_truncation` | Float | No | `1.0` | 1.0 recommended; lower to fix over-saturation. |
| `max_sequence_length`| Integer | No | `512` | Token limit for long prompts. |
| `seed` | Integer | No | `42` | Random seed for reproducibility. |
| `lora_scale` | Float | No | `0.85` | Strength of the LoRA adapter (0.8-0.9 recommended). |
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
