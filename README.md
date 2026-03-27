# Z-Image RunPod Serverless Worker

This project implements a RunPod serverless worker for the **Z-Image** base model. It supports dynamic LoRA loading from external URLs (e.g., Backblaze B2 presigned URLs) and uploads lossless PNG outputs back to S3-compatible storage.

## Features

- **High-Performance Image:** Core dependencies are pre-baked into the Docker image for near-instant startup (<20s imports).
- **Persistent Volume Support:** Model weights are cached on `/runpod-volume/huggingface` to avoid re-downloading.
- **Photorealism-Oriented Defaults:** Uses 50 steps, `guidance_scale=4.0`, realism-leaning CFG defaults (`cfg_normalization=True`, `cfg_truncation=1.0`), and a built-in negative prompt to suppress common artifacts.
- **Flash Attention 2:** Enabled at model load time via `attn_implementation="flash_attention_2"` when the `flash-attn` package is available (RTX 4090/5090 and newer). Falls back to PyTorch SDPA automatically. Enabled at init rather than post-load to remain fully compatible with LoRA injection.
- **Scheduler Control:** Supports `use_beta_sigmas` to toggle FlowMatch beta-sigma scheduling and `shift` to adjust the composition/detail balance.
- **Adaptive VAE Tiling:** Keeps VAE tiling off at 1024-ish outputs by default to reduce potential tile artifacts, while enabling it for larger images.
- **Optional Two-Pass Refinement:** Upscales pass-1 output with the `4xPurePhoto-RealPLSKR` checkpoint and runs a Z-Image img2img refinement pass for extra detail.
- **Dynamic Multi-LoRA Support:** Load one or more LoRAs from any URL at runtime. Multiple LoRAs are downloaded in parallel and blended by weight. Legacy single-LoRA inputs remain fully supported. Handles multiple LoRA key formats (kohya, diffusers-native, Flux2/Klein) with automatic conversion and alpha-key patching.
- **VRAM-Optimized:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically to reduce allocator fragmentation. Second-pass upscale defaults to 1.25× to stay within 24 GB when LoRAs are loaded.
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
| `TORCH_COMPILE` | Optional. Compiles the transformer for faster inference after warm-up. First request will be slower. | `false` |

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
| `negative_prompt` | String | No | *(see below)* | Text to avoid in the generation. A photorealism-oriented default is applied when omitted; pass `""` to disable. |
| `width` | Integer | No | `1024` | Image width. |
| `height` | Integer | No | `1024` | Image height. |
| `steps` | Integer | No | `50` | Number of inference steps. |
| `guidance_scale` | Float | No | `4.0` | CFG scale (3.0–5.0 recommended; 4.0 matches official Z-Image guidance). |
| `cfg_normalization`| Boolean | No | `True` | CFG normalization toggle (enabled by default for photorealism). |
| `cfg_truncation` | Float | No | `1.0` | 1.0 recommended; lower to fix over-saturation. |
| `max_sequence_length`| Integer | No | `512` | Token limit for long prompts. |
| `seed` | Integer | No | `42` | Random seed for reproducibility. |
| `use_beta_sigmas` | Boolean | No | `True` | Explicitly rebuilds FlowMatch scheduler with/without beta sigmas. |
| `shift` | Float | No | model default (~3.0) | Scheduler shift controlling composition vs detail balance. Higher values (5–7) favour creative composition; lower (1–2) favour detail. 3.0–3.5 is a good photorealism sweet spot. |
| `vae_tiling` | Boolean | No | auto | Override adaptive VAE tiling behavior (`auto`: on only for >1024×1024 area). |
| `second_pass_enabled` | Boolean | No | env/default | Enables pass-2 upscale + img2img refinement. |
| `second_pass_upscale` | Float | No | `1.25` | Output scale factor for pass 2 upscaling. 1.25x keeps peak VRAM within 24 GB with LoRAs loaded; use 1.5 only on cards with more headroom. |
| `second_pass_strength` | Float | No | `0.30` | Img2img strength for pass 2. 0.25–0.40 adds detail without straying from the first-pass composition. |
| `second_pass_steps` | Integer | No | `20` | Img2img denoising steps for pass 2. |
| `second_pass_guidance_scale` | Float | No | `4.0` | CFG scale for pass 2. Matches pass-1 default so the refinement reinforces photorealistic detail rather than softening it. |
| `second_pass_seed` | Integer | No | `seed` | Seed for pass 2 reproducibility. |
| `second_pass_cfg_normalization` | Boolean | No | `True` | CFG normalization toggle for pass 2 (now matches pass 1 default for photorealism). |
| `second_pass_cfg_truncation` | Float | No | `1.0` | CFG truncation for pass 2. |
| `second_pass_max_sequence_length`| Integer | No | `min(max_sequence_length, 384)` | Token limit for pass-2 refinement to reduce VRAM pressure. |
| `second_pass_use_beta_sigmas` | Boolean | No | `use_beta_sigmas` | Scheduler beta-sigma toggle for pass 2. |
| `second_pass_vae_tiling` | Boolean | No | `False` | VAE tiling for pass 2. Disabled by default — tiling causes visible seams/fraying at second-pass image sizes. Use slicing (`second_pass_vae_slicing`) instead. |
| `second_pass_vae_slicing` | Boolean | No | `True` | VAE slicing for pass 2 (enabled by default for memory headroom). |

### LoRA URL Format

LoRA URLs can point to any direct-download `.safetensors` file (Backblaze B2, presigned S3, etc.).

**HuggingFace URLs:** Use the `/resolve/` path, **not** the `/blob/` path. The `/blob/` URL returns an HTML page viewer, not the binary file, and will cause a load error.

```
# Correct — direct binary download
https://huggingface.co/<owner>/<repo>/resolve/main/<file>.safetensors

# Wrong — returns HTML page
https://huggingface.co/<owner>/<repo>/blob/main/<file>.safetensors
```

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
