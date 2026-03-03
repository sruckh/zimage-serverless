# Z-Image RunPod Serverless Worker

This project implements a RunPod serverless worker for the **Z-Image** base model. it supports dynamic LoRA loading from external URLs (e.g., Backblaze B2 presigned URLs) and uploads high-quality JPG outputs back to S3-compatible storage.

## Features

- **High-Performance Image:** Core dependencies are pre-baked into the Docker image for near-instant startup (<20s imports).
- **Persistent Volume Support:** Model weights are cached on `/runpod-volume/huggingface` to avoid re-downloading.
- **Photorealism-Oriented Defaults:** Uses 50 steps with realism-leaning CFG defaults (`cfg_normalization=True`, `cfg_truncation=1.0`).
- **Scheduler Control:** Supports `use_beta_sigmas` to toggle FlowMatch beta-sigma scheduling.
- **Adaptive VAE Tiling:** Keeps VAE tiling off at 1024-ish outputs by default to reduce potential tile artifacts, while enabling it for larger images.
- **Optional Two-Pass Refinement:** Upscales pass-1 output with the `4xPurePhoto-RealPLSKR` checkpoint and runs a Z-Image img2img refinement pass for extra detail.
- **Dynamic Multi-LoRA Support:** Load one or more LoRAs from any URL at runtime. Multiple LoRAs are downloaded in parallel and blended by weight. Legacy single-LoRA inputs remain fully supported.
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
| `UPSCALE_MODEL_URL` | Optional. URL for the upscaler `.pth` model. | Starinspace 4xPurePhoto-RealPLSKR |
| `UPSCALE_MODEL_PATH` | Optional. Cached path for the upscaler model. | `/runpod-volume/zimage-diffusion/models/upscale/4xPurePhoto-RealPLSKR.pth` |
| `UPSCALE_USE_CUDA` | Optional. Runs the RealPLKSR upscaler on CUDA when `true`; CPU by default to reduce VRAM pressure. | `false` |

The bootstrap script now checks `UPSCALE_MODEL_PATH` on every start and downloads it once if missing, even when `.installed_v2` already exists.

## Monitoring & Debugging

On the first cold start, you can monitor the installation progress in the **RunPod logs** or by checking the persistent log file:
`/runpod-volume/zimage-diffusion/bootstrap.log`

## Endpoint Input Parameters

When making a call to the `/run` or `/runsync` endpoint, use the following JSON structure in the `input` field:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | String | **Yes** | - | The text prompt for generation. |
| `loras` | Array | No | - | **Preferred multi-LoRA input.** Array of `{"url": "...", "scale": 1.0}` objects. `scale` is optional per entry (default `0.85`). See [LoRA blending](#lora-blending) below. |
| `lora_url` | String | No | - | **Legacy.** URL to a single `.safetensors` LoRA file. Ignored when `loras` is provided. |
| `lora_scale` | Float | No | `0.85` | **Legacy.** Scale for the single `lora_url` adapter (0.8–0.9 recommended). Ignored when `loras` is provided. |
| `negative_prompt` | String | No | `""` | Text to avoid in the generation. |
| `width` | Integer | No | `1024` | Image width. |
| `height` | Integer | No | `1024` | Image height. |
| `steps` | Integer | No | `50` | Number of inference steps. |
| `guidance_scale` | Float | No | `3.0` | CFG scale (3.0–4.5 recommended for likeness). |
| `cfg_normalization`| Boolean | No | `True` | CFG normalization toggle (enabled by default for photorealism). |
| `cfg_truncation` | Float | No | `1.0` | 1.0 recommended; lower to fix over-saturation. |
| `max_sequence_length`| Integer | No | `512` | Token limit for long prompts. |
| `seed` | Integer | No | `42` | Random seed for reproducibility. |
| `use_beta_sigmas` | Boolean | No | `True` | Explicitly rebuilds FlowMatch scheduler with/without beta sigmas. |
| `vae_tiling` | Boolean | No | auto | Override adaptive VAE tiling behavior (`auto`: on only for >1024×1024 area). |
| `second_pass_enabled` | Boolean | No | env/default | Enables pass-2 upscale + img2img refinement. |
| `second_pass_upscale` | Float | No | `1.5` | Output scale factor for pass 2 upscaling (1.5x default to reduce VRAM). |
| `second_pass_strength` | Float | No | `0.18` | Img2img strength for pass 2 (lower default to preserve LoRA likeness). |
| `second_pass_steps` | Integer | No | `8` | Img2img denoising steps for pass 2. |
| `second_pass_guidance_scale` | Float | No | `1.2` | CFG scale for pass 2 (lower default to avoid over-rewrite). |
| `second_pass_seed` | Integer | No | `seed` | Seed for pass 2 reproducibility. |
| `second_pass_cfg_normalization` | Boolean | No | `False` | CFG normalization toggle for pass 2. |
| `second_pass_cfg_truncation` | Float | No | `1.0` | CFG truncation for pass 2. |
| `second_pass_max_sequence_length`| Integer | No | `min(max_sequence_length, 384)` | Token limit for pass-2 refinement to reduce VRAM pressure. |
| `second_pass_use_beta_sigmas` | Boolean | No | `use_beta_sigmas` | Scheduler beta-sigma toggle for pass 2. |
| `second_pass_vae_tiling` | Boolean | No | `True` | VAE tiling for pass 2 (enabled by default for memory headroom). |
| `second_pass_vae_slicing` | Boolean | No | `True` | VAE slicing for pass 2 (enabled by default for memory headroom). |

### LoRA Blending

When using multiple LoRAs via the `loras` array, their contributions are blended additively using the PEFT **cat-method**:

```
output = base_model + Σ(scale_i × internal_scaling_i × LoRA_i)
```

The `scale` values are **independent multipliers**, not percentages of a shared budget. A request with `[{"scale": 1.0}, {"scale": 0.25}]` gives a true **4:1 influence ratio** — LoRA 1 contributes four times as much as LoRA 2, both added on top of the base model. All LoRA files are downloaded in parallel to minimise latency.

### Example Request Bodies

**Single LoRA (legacy — still fully supported):**

```json
{
  "input": {
    "prompt": "A professional portrait of K1mScum in a futuristic setting",
    "lora_url": "https://f004.backblazeb2.com/file/my-bucket/K1mScum.safetensors?params...",
    "lora_scale": 0.85,
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 4.5,
    "seed": 12345
  }
}
```

**Multiple LoRAs — character LoRA at full strength blended with a style LoRA at quarter strength (4:1 ratio):**

```json
{
  "input": {
    "prompt": "A professional portrait of K1mScum in a futuristic setting, cinematic lighting",
    "loras": [
      {"url": "https://f004.backblazeb2.com/file/my-bucket/K1mScum.safetensors?params...", "scale": 1.0},
      {"url": "https://f004.backblazeb2.com/file/my-bucket/cinematic-style.safetensors?params...", "scale": 0.25}
    ],
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 4.5,
    "seed": 12345
  }
}
```

**No LoRA:**

```json
{
  "input": {
    "prompt": "A photorealistic landscape at golden hour",
    "width": 1024,
    "height": 1024,
    "steps": 50,
    "seed": 42
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
