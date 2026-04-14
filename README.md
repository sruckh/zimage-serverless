# Z-Image RunPod Serverless Worker

This project implements a RunPod serverless worker for the **Z-Image** base model. It supports dynamic LoRA loading from external URLs (e.g., Backblaze B2 presigned URLs) and uploads lossless PNG outputs back to S3-compatible storage.

## Features

- **High-Performance Image:** Core dependencies are pre-baked into the Docker image for near-instant startup (<20s imports).
- **Persistent Volume Support:** Model weights are cached on `/runpod-volume/huggingface` to avoid re-downloading.
- **Photorealism-Oriented Defaults:** Uses high-quality defaults (`cfg_normalization=True`, `use_beta_sigmas=True`) and automatic step/guidance optimization based on the model variant (Base vs Turbo) to suppress artifacts.
- **High-Fidelity VAE:** Forces VAE decoding to `float32` to eliminate jagged artifacts and pixelation often seen in high-step `bfloat16` generations.
- **Flash Attention 2:** Enabled at model load time via `attn_implementation="flash_attention_2"` when the `flash-attn` package is available (RTX 4090/5090 and newer). Falls back to PyTorch SDPA automatically.
- **FlowMatch Scheduler:** Z-Image uses `FlowMatchEulerDiscreteScheduler` — a flow matching architecture. DPM++, DDIM, Euler Ancestral and other DDPM-era samplers are not compatible.
- **Consistent Shift Across Passes:** The `shift` parameter is applied to both the base pass and the second-pass refinement scheduler for consistent behavior.
- **Adaptive VAE Tiling:** Keeps VAE tiling off at 1024-ish outputs by default to reduce potential tile artifacts, while enabling it for larger images.
- **Optional Two-Pass Refinement:** Upscales pass-1 output with the `4xPurePhoto-RealPLSKR` checkpoint and runs a Z-Image img2img refinement pass for extra detail. Upscaler runs on CUDA when `UPSCALE_USE_CUDA=true` (recommended for 24 GB cards).
- **Dynamic Multi-LoRA Support:** Load one or more LoRAs from any URL at runtime. Multiple LoRAs are downloaded in parallel and blended by weight. Handles all common LoRA key formats: kohya (`lora_down/lora_up`), diffusers-native (`lora_A/lora_B`), ComfyUI-exported (`diffusion_model.*` prefix), OneTrainer/Kohya exports (`lora_unet_` prefix, `context_refiner`, `noise_refiner`), and Flux2/Klein — with automatic alpha-key patching and format conversion (including `transformer_blocks.` to native `layers.` mapping).
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
| `UPSCALE_USE_CUDA` | Optional. Runs the RealPLKSR upscaler on CUDA when `true`. Recommended for 24 GB cards; CPU default avoids OOM on smaller cards. | `false` |
| `TORCH_COMPILE` | Optional. Compiles the transformer for faster inference after warm-up. First request will be slower. | `false` |

The bootstrap script checks `UPSCALE_MODEL_PATH` on every start and downloads it once if missing.

## Monitoring & Debugging

On the first cold start, you can monitor the installation progress in the **RunPod logs** or by checking the persistent log file:
`/runpod-volume/zimage-diffusion/bootstrap.log`

## Endpoint Input Parameters

When making a call to the `/run` or `/runsync` endpoint, use the following JSON structure in the `input` field:

### Base Pass Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | String | **Yes** | - | The text prompt for generation. |
| `loras` | Array | No | - | **Preferred multi-LoRA input.** Array of `{"url": "...", "scale": 1.0}` objects. `scale` is optional per entry (default `0.85`). See [LoRA blending](#lora-blending) below. |
| `lora_url` | String | No | - | **Legacy.** URL to a single `.safetensors` LoRA file. Ignored when `loras` is provided. |
| `lora_scale` | Float | No | `0.85` | **Legacy.** Scale for the single `lora_url` adapter. Ignored when `loras` is provided. |
| `negative_prompt` | String | No | *(see below)* | Text to avoid in the generation. A photorealism-oriented default is applied when omitted; pass `""` to disable. |
| `width` | Integer | No | `1024` | Image width in pixels. |
| `height` | Integer | No | `1024` | Image height in pixels. |
| `steps` | Integer | No | `auto` | Number of inference steps. Auto-optimizes to `50` (Base) or `8` (Turbo) when omitted. |
| `guidance_scale` | Float | No | `auto` | CFG scale. Auto-optimizes to `4.0` (Base) or `1.0` (Turbo) when omitted. 4.0–5.5 recommended for Base photorealism. |
| `cfg_normalization` | Boolean | No | `true` | CFG normalization. Recommended `true` for photorealistic results to prevent over-saturation. |
| `cfg_truncation` | Float | No | `1.0` | CFG truncation. 1.0 recommended; lower to fix over-saturation. |
| `max_sequence_length` | Integer | No | `512` | Token limit for long prompts. |
| `seed` | Integer | No | `42` | Random seed for reproducibility. |
| `use_beta_sigmas` | Boolean | No | `true` | Enables FlowMatch beta-sigma scheduling. Recommended `true` for optimal Z-Image noise distribution and artifact reduction. |
| `shift` | Float | No | `3.0` | Scheduler shift applied to **both** base and second-pass schedulers. Lower (2–3) favours fine detail and photorealism; higher (5–7) favours creative composition. 3.0 is the photorealism sweet spot. |
| `vae_tiling` | Boolean | No | auto | Override adaptive VAE tiling (`auto`: enabled only for outputs larger than 1024×1024). |

### Second Pass (Upscale + Refinement) Parameters

The second pass upscales the base output with RealPLKSR then runs img2img refinement. Enabled by default via `SECOND_PASS_DEFAULT_ENABLED`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `second_pass_enabled` | Boolean | No | env default | Enables pass-2 upscale + img2img refinement. |
| `second_pass_upscale` | Float | No | `1.25` | Upscale factor for pass 2. 1.25× fits within 24 GB with LoRAs loaded. Use 1.5× only on cards with more headroom. |
| `second_pass_strength` | Float | No | `0.30` | Img2img denoising strength. Lower values (0.10–0.20) preserve more first-pass detail; higher values (0.30–0.45) add more refinement but may soften character features. |
| `second_pass_steps` | Integer | No | `20` | Denoising steps for pass 2. |
| `second_pass_guidance_scale` | Float | No | `4.0` | CFG scale for pass 2. |
| `second_pass_seed` | Integer | No | `seed` | Seed for pass 2 (defaults to same as base pass). |
| `second_pass_cfg_normalization` | Boolean | No | `false` | CFG normalization for pass 2. |
| `second_pass_cfg_truncation` | Float | No | `1.0` | CFG truncation for pass 2. |
| `second_pass_max_sequence_length` | Integer | No | `512` | Token limit for pass-2 prompt. Synced with base pass for deep prompt understanding. |
| `second_pass_use_beta_sigmas` | Boolean | No | `use_beta_sigmas` | Beta-sigma toggle for pass-2 scheduler. Defaults to the base pass value. |
| `second_pass_vae_tiling` | Boolean | No | `false` | VAE tiling for pass 2. Disabled by default — tiling causes visible seams at second-pass image sizes. Use slicing instead. |
| `second_pass_vae_slicing` | Boolean | No | `true` | VAE slicing for pass 2. Enabled by default for VRAM headroom. |

### Scheduler Notes

Z-Image uses `FlowMatchEulerDiscreteScheduler` (flow matching). **DPM++, DDIM, Euler Ancestral, and other DDPM-era samplers are not supported and not compatible.** The equivalent quality controls are:

| FlowMatch Parameter | Effect |
|---|---|
| `float32 VAE` | **Enabled.** Automatically eliminates jagged/pixelated artifacts in generations. |
| `shift=3.0` | Detail/photorealism-focused |
| `shift=6.0+` | Creative/composition-focused |
| `steps=50` | Standard quality (30 is acceptable for drafts) |

## LoRA URL Format

LoRA URLs can point to any direct-download `.safetensors` file (Backblaze B2, presigned S3, etc.).

**HuggingFace URLs:** Use the `/resolve/` path, **not** the `/blob/` path. The `/blob/` URL returns an HTML page viewer, not the binary file, and will cause a load error.

```
# Correct — direct binary download
https://huggingface.co/<owner>/<repo>/resolve/main/<file>.safetensors

# Wrong — returns HTML page
https://huggingface.co/<owner>/<repo>/blob/main/<file>.safetensors
```

**Supported LoRA formats:**
- **Kohya** (`lora_down`/`lora_up` keys) — standard training format
- **Diffusers-native** (`lora_A`/`lora_B` keys) — diffusers training format
- **ComfyUI-exported** (`diffusion_model.*` prefix) — LoRAs saved from ComfyUI workflows
- **OneTrainer/Kohya Generic** (`lora_unet_` prefix, `context_refiner`, `noise_refiner` keys) — supported via manual mapping
- **Civitai Custom** (`adaLN_modulation` keys, `layers.` prefix) — supported via manual mapping (e.g. AmberNoir)
- **Flux2/Klein** (`double_blocks`/`single_blocks` keys) — converted automatically

Missing alpha keys are synthesized automatically (alpha=rank, scale=1.0).

## LoRA Blending

When using multiple LoRAs via the `loras` array, their contributions are blended additively using the PEFT **cat-method**:

```
output = base_model + Σ(scale_i × internal_scaling_i × LoRA_i)
```

The `scale` values are **independent multipliers**, not percentages of a shared budget. A request with `[{"scale": 1.0}, {"scale": 0.25}]` gives a true **4:1 influence ratio** — LoRA 1 contributes four times as much as LoRA 2, both added on top of the base model. All LoRA files are downloaded in parallel to minimise latency.

**LoRA scale recommendations:**
- Character/subject LoRA: `0.9–1.0`
- Style LoRA: `0.3–0.6` (lower when combined with a character LoRA to avoid overpowering it)

## Example Request Bodies

**Single LoRA (legacy — still fully supported):**

```json
{
  "input": {
    "prompt": "A professional portrait of K1mScum in a futuristic setting",
    "lora_url": "https://huggingface.co/Gemneye/K1mScum-ZImage-Base/resolve/main/K1mScum-000086-Z-Image-Base.safetensors",
    "lora_scale": 0.9,
    "width": 1024,
    "height": 1024,
    "steps": 50,
    "guidance_scale": 4.5,
    "shift": 3.0,
    "seed": 12345
  }
}
```

**Multiple LoRAs — character LoRA blended with a style LoRA:**

```json
{
  "input": {
    "prompt": "A professional portrait of K1mScum in a futuristic setting, cinematic lighting",
    "loras": [
      {"url": "https://huggingface.co/Gemneye/K1mScum-ZImage-Base/resolve/main/K1mScum-000086-Z-Image-Base.safetensors", "scale": 1.0},
      {"url": "https://f004.backblazeb2.com/file/my-bucket/cinematic-style.safetensors", "scale": 0.4}
    ],
    "width": 1024,
    "height": 1024,
    "steps": 50,
    "guidance_scale": 4.5,
    "shift": 3.0,
    "second_pass_strength": 0.15,
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
    "guidance_scale": 4.0,
    "shift": 3.0,
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
   - Set the necessary Environment Variables (see above).
   - Set `UPSCALE_USE_CUDA=true` if your worker has a 24 GB card.
   - Deploy the endpoint.

## Project Structure

- `Dockerfile`: Base image — installs Python dependencies including `diffusers==0.37.1`.
- `runpod_bootstrap.sh`: Runs on every cold start — installs Flash Attention, caches model weights, ensures correct diffusers version.
- `handler.py`: The RunPod serverless worker logic.
- `s3_utils.py`: Utility for S3/B2 uploads.
- `.gitignore`: Prevents local artifacts and secrets from being committed.
