# Z-Image RunPod Serverless Worker

This project implements a RunPod serverless worker for the **Z-Image** base model. It supports dynamic LoRA loading from external URLs (e.g., Backblaze B2 presigned URLs) and uploads lossless PNG outputs back to S3-compatible storage.

## Features

- **High-Performance Image:** Core dependencies are pre-baked into the Docker image for near-instant startup (<20s imports).
- **Persistent Volume Support:** Model weights are cached on `/runpod-volume/huggingface` to avoid re-downloading.
- **Photorealism-Oriented Defaults:** Uses official Tongyi-MAI recommendations (50 steps, `cfg_normalization=true`, `shift=1.0`, `guidance_scale=4.5`) and automatic step/guidance optimization based on the model variant (Base vs Turbo) to ensure stable, high-quality results.
- **Stable Mixed-Precision Inference:** Automatically handles `bfloat16` transformer weights and `float32` VAE decoding via `torch.autocast`. LoRA weights are explicitly cast to the model's native precision at load time to prevent "bias type mismatch" errors.
- **High-Fidelity VAE:** Forces VAE decoding to `float32` to eliminate jagged artifacts and pixelation often seen in high-step `bfloat16` generations.
- **Flash Attention 2:** Enabled at model load time via `attn_implementation="flash_attention_2"` when the `flash-attn` package is available (RTX 4090/5090 and newer). Falls back to PyTorch SDPA automatically.
- **FlowMatch Scheduler:** Z-Image uses `FlowMatchEulerDiscreteScheduler` — a flow matching architecture. DPM++, DDIM, Euler Ancestral and other DDPM-era samplers are not compatible.
- **Consistent Shift Across Passes:** The `shift` parameter is applied to both the base pass and the optional img2img refinement scheduler for consistent behavior.
- **Adaptive VAE Tiling:** Keeps VAE tiling off at 1024-ish outputs by default to reduce potential tile artifacts, while enabling it for larger images.
- **Selectable Detail Upscalers (spandrel):** Choose an upscaler from a curated registry via `upscale_model` (see [Available Upscalers](#available-upscalers)). Models are loaded with [spandrel](https://github.com/chaiNNer-org/spandrel), so the architecture (RealPLKSR, ESRGAN, etc.) is auto-detected from the checkpoint. Each model is downloaded **once, lazily, on first use** to the persistent volume and cached thereafter. The default is `nomos_webphoto` (`4xNomosWebPhoto_RealPLKSR`), a natural realistic-photo upscaler that restores detail rather than hallucinating fake facial micro-detail. The upscaler runs on **CPU by default** to keep VRAM free for the diffusion passes; set `UPSCALE_USE_CUDA=true` only if you have spare VRAM headroom.
- **img2img Hires-Fix (default ON):** The final image is upscaled and then lightly **re-diffused** through Z-Image img2img (`second_pass_strength=0.42`). This repaints away GAN/SR artifacts (fake lashes/brows/hair) and the Z-Image Base under-denoising residual, producing more natural faces. Set `second_pass_enabled=false` to get the raw single-pass model upscale instead.
- **Dynamic Multi-LoRA Support:** Load one or more LoRAs from any URL at runtime. Multiple LoRAs are downloaded in parallel and blended by weight. Handles all common LoRA key formats: kohya (`lora_down/lora_up`), diffusers-native (`lora_A/lora_B`), ComfyUI-exported (`diffusion_model.*` prefix), OneTrainer/Kohya exports (`lora_unet_` prefix, `context_refiner`, `noise_refiner`), and Flux2/Klein — with automatic alpha-key patching and format conversion (including `transformer_blocks.` to native `layers.` mapping).
- **VRAM-Optimized:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically to reduce allocator fragmentation. The default img2img hires-fix upscale is `1.25×` to stay within 24 GB when LoRAs are loaded; the single-pass detail upscale (when the hires-fix is disabled) defaults to `1.5×`.
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
| `UPSCALE_DEFAULT_ENABLED` | Optional. Enables the single-pass detail upscale (used only when the hires-fix is disabled). | `true` |
| `SECOND_PASS_DEFAULT_ENABLED` | Optional. Enables the img2img hires-fix refinement by default. | `true` |
| `UPSCALE_DEFAULT_MODEL` | Optional. Registry key of the default upscaler (see [Available Upscalers](#available-upscalers)). | `nomos_webphoto` |
| `UPSCALE_DIR` | Optional. Volume directory where upscaler checkpoints are cached. | `/runpod-volume/zimage-diffusion/models/upscale` |
| `UPSCALE_USE_CUDA` | Optional. Runs the upscaler on CUDA when `true`. CPU default keeps VRAM free for the diffusion passes (with the hires-fix on by default, VRAM is the bottleneck on 24 GB). Enable only with spare headroom. | `false` |
| `TORCH_COMPILE` | Optional. Compiles the transformer for faster inference after warm-up. First request will be slower. | `false` |

The bootstrap pre-stages the default upscaler (`UPSCALE_DEFAULT_MODEL`) to the volume on start. Any other registry model downloads lazily in the handler on first use and is then cached on the volume.

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
| `steps` | Integer | No | `auto` | Number of inference steps. Auto-optimizes to `50` (Base) or `9` (Turbo) when omitted. 28–50 recommended for Base. |
| `guidance_scale` | Float | No | `auto` | CFG scale. Auto-optimizes to `4.5` (Base) or `0.0` (Turbo) when omitted. 3.0–5.0 recommended for Base photorealism; 4.5 is the empirically tested sweet spot. |
| `cfg_normalization` | Boolean | No | `true` | CFG normalization. `true` is the official Tongyi-MAI recommendation for photorealism; corrects guidance vector magnitude to preserve detail in complex scenes. Pass `false` for stylistic/artistic outputs. |
| `cfg_truncation` | Float | No | `1.0` | CFG truncation. 1.0 recommended; lower to fix over-saturation. |
| `max_sequence_length` | Integer | No | `512` | Token limit for long prompts. |
| `seed` | Integer | No | `42` | Random seed for reproducibility. |
| `use_beta_sigmas` | Boolean | No | `false` | Enables FlowMatch beta-sigma scheduling. Recommended `false` for official Z-Image noise distribution. |
| `shift` | Float | No | `1.0` | Scheduler shift. `1.0` is the Z-Image architecture/scheduler default and matches official Tongyi-MAI inference settings. Higher values (e.g. `3.0`) over-weight early composition steps and starve detail refinement, producing softer/underbaked output — raise only if a specialized LoRA requires it. |
| `upscale_model` | String | No | `nomos_webphoto` | Which upscaler from the registry to use. One of the [Available Upscalers](#available-upscalers) keys. Used by both the hires-fix upscale and the single-pass detail upscale. Unknown keys return an error. |
| `upscale_enabled` | Boolean | No | `true` | Runs the single-pass detail upscale on the final image (ComfyUI-style pure super-resolution, no repaint). Set `false` to return the raw generation. Skipped when `second_pass_enabled=true` (the hires-fix path does its own upscale). |
| `upscale_factor` | Float | No | `1.5` | Net output scale for the single-pass detail upscale. The model's native scale always runs (adding detail); the result is then resized to this factor. |
| `vae_tiling` | Boolean | No | auto | Override adaptive VAE tiling (`auto`: enabled only for outputs larger than 1024×1024). |

### img2img Hires-Fix Parameters (default ON)

This is the **default** finishing path. The base output is upscaled with the selected `upscale_model` and then **re-diffused** through Z-Image img2img. The light second pass (`second_pass_strength=0.42`) repaints away upscaler GAN/SR artifacts (fake lashes/brows/hair) and the Z-Image Base under-denoising residual, producing more natural faces. It is **ON by default** (`SECOND_PASS_DEFAULT_ENABLED=true`); when enabled it replaces the single-pass detail upscale. Set `second_pass_enabled=false` to get the raw single-pass model upscale instead.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `second_pass_enabled` | Boolean | No | `true` | Enables the img2img hires-fix (upscale + re-diffusion). Replaces the single-pass detail upscale when `true`. Set `false` for the raw single-pass upscale. |
| `second_pass_upscale` | Float | No | `1.25` | Upscale factor for pass 2. 1.25× fits within 24 GB with LoRAs loaded. Use 1.5× only on cards with more headroom. |
| `second_pass_strength` | Float | No | `0.42` | Img2img denoising strength. `0.42` is tuned to clean the Z-Image Base incomplete-denoising artifact (see upstream issue #144) without losing composition. Lower (0.20–0.30) to preserve more first-pass character detail; higher (0.50+) for heavy refinement. |
| `second_pass_steps` | Integer | No | `28` | Denoising steps for pass 2. |
| `second_pass_guidance_scale` | Float | No | `4.5` | CFG scale for pass 2. |
| `second_pass_seed` | Integer | No | `seed` | Seed for pass 2 (defaults to same as base pass). |
| `second_pass_cfg_normalization` | Boolean | No | `true` | CFG normalization for pass 2. Matches base pass default for consistent realism. |
| `second_pass_cfg_truncation` | Float | No | `1.0` | CFG truncation for pass 2. |
| `second_pass_max_sequence_length` | Integer | No | `512` | Token limit for pass-2 prompt. Synced with base pass for deep prompt understanding. |
| `second_pass_use_beta_sigmas` | Boolean | No | `use_beta_sigmas` | Beta-sigma toggle for pass-2 scheduler. Defaults to the base pass value. |
| `second_pass_vae_tiling` | Boolean | No | `false` | VAE tiling for pass 2. Disabled by default — tiling causes visible seams at second-pass image sizes. Use slicing instead. |
| `second_pass_vae_slicing` | Boolean | No | `true` | VAE slicing for pass 2. Enabled by default for VRAM headroom. |

### Available Upscalers

Pass one of these keys as `upscale_model`. All are loaded via spandrel (architecture auto-detected) and downloaded once to the volume on first use. Adding a model is config-only — extend the `UPSCALE_MODELS` registry in `handler.py` with a name, direct download URL, and filename.

| Key | Model | Arch | Scale | Notes |
|-----|-------|------|-------|-------|
| `nomos_webphoto` | 4xNomosWebPhoto_RealPLKSR | RealPLKSR | 4× | **Default.** Natural realistic-photo upscaler (Philip Hofmann). Restores detail rather than hallucinating fake facial micro-detail — best general choice for people. |
| `nomos_webphoto_esrgan` | 4xNomosWebPhoto_esrgan | ESRGAN | 4× | Same realistic dataset/target as the default, ESRGAN architecture. Slightly different detail character. |
| `purephoto` | 4xPurePhoto-RealPLSKR | RealPLKSR | 4× | Legacy default. A sharpening/detail-injection model — crisper, but can over-process faces into a synthetic look. |

> Source: [OpenModelDB](https://openmodeldb.info/). The native scale is read from the checkpoint; `upscale_factor` / `second_pass_upscale` set the final output scale (the model runs at its native scale, then the result is resized).

### Scheduler Notes

Z-Image uses `FlowMatchEulerDiscreteScheduler` (flow matching). **DPM++, DDIM, Euler Ancestral, and other DDPM-era samplers are not supported and not compatible.** The equivalent quality controls are:

| FlowMatch Parameter | Effect |
|---|---|
| `float32 VAE` | **Enabled.** Automatically eliminates jagged/pixelated artifacts in generations. |
| `shift=1.0` | Z-Image architecture/scheduler default; preserves detail refinement (raising it softens output). |
| `cfg_normalization=true` | Official Tongyi-MAI recommendation for realism; corrects guidance magnitude across the scene. |
| `steps=50` | Official Base-model sweet spot for detail (28–50 range). |

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
    "second_pass_enabled": true,
    "second_pass_strength": 0.25,
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

> **Note on output size:** With the default img2img hires-fix (`second_pass_enabled=true`, `second_pass_upscale=1.25`), a `1024×1024` request returns a **`1280×1280`** PNG. If you instead disable the hires-fix and use the single-pass detail upscale (`second_pass_enabled=false`, `upscale_factor=1.5`), a `1024×1024` request returns **`1536×1536`**. The generation runs at the requested `width`/`height`; the upscale is applied afterward.

**Choosing a different upscaler:**

```json
{
  "input": {
    "prompt": "A professional studio portrait, soft window light",
    "width": 832,
    "height": 1216,
    "steps": 50,
    "upscale_model": "nomos_webphoto_esrgan",
    "seed": 42
  }
}
```

**Raw generation (no upscale, no hires-fix):**

```json
{
  "input": {
    "prompt": "A photorealistic landscape at golden hour",
    "width": 1024,
    "height": 1024,
    "steps": 50,
    "second_pass_enabled": false,
    "upscale_enabled": false,
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
   - Leave `UPSCALE_USE_CUDA=false` (default) on 24 GB cards — the upscaler stays on CPU so VRAM is reserved for the diffusion passes.
   - Deploy the endpoint.

> **VRAM / OOM note:** With the img2img hires-fix on by default, the second diffusion pass (at `second_pass_upscale=1.25×`) is the main VRAM consumer on 24 GB cards, especially with LoRAs loaded or larger `width`/`height`. If the second pass hits CUDA OOM, the handler **degrades gracefully**: it logs the OOM and returns the upscaled (non-re-diffused) image instead of failing the job. To avoid the second pass entirely, set `second_pass_enabled=false` per request or `SECOND_PASS_DEFAULT_ENABLED=false` on the endpoint.

## Project Structure

- `Dockerfile`: Base image — installs Python dependencies including `diffusers==0.37.1`.
- `runpod_bootstrap.sh`: Runs on every cold start — installs Flash Attention, caches model weights, ensures correct diffusers version.
- `handler.py`: The RunPod serverless worker logic.
- `s3_utils.py`: Utility for S3/B2 uploads.
- `.gitignore`: Prevents local artifacts and secrets from being committed.
