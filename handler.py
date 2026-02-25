import time
t_start = time.time()
import runpod
import os
import torch
import requests
import uuid
import traceback
from PIL import Image
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, FlowMatchEulerDiscreteScheduler
from s3_utils import upload_image_to_s3

print(f"--- Initializing Handler (Imports took {time.time() - t_start:.2f}s) ---")

# Environment Variables
MODEL_ID = os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image")
MODEL_DIR = "/runpod-volume/zimage-diffusion/models" # Runpod volume location
OUTPUT_DIR = "/tmp/outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model cache to avoid re-loading across jobs in the same session
pipe = None
img2img_pipe = None
upscaler = None

UPSCALE_MODEL_URL = os.environ.get(
    "UPSCALE_MODEL_URL",
    "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-RealPLSKR.pth",
)
UPSCALE_MODEL_PATH = os.environ.get(
    "UPSCALE_MODEL_PATH",
    "/runpod-volume/zimage-diffusion/models/upscale/4xPurePhoto-RealPLSKR.pth",
)

def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default

def _configure_scheduler(pipeline, use_beta_sigmas):
    """
    Rebuild scheduler from the loaded model config while optionally enabling beta sigmas.
    """
    current_use_beta_sigmas = bool(pipeline.scheduler.config.get("use_beta_sigmas", False))
    if current_use_beta_sigmas == use_beta_sigmas:
        return
    try:
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            use_beta_sigmas=use_beta_sigmas,
        )
        print(f"Scheduler configured with use_beta_sigmas={use_beta_sigmas}")
    except Exception as e:
        print(
            f"Warning: could not configure scheduler use_beta_sigmas={use_beta_sigmas} "
            f"(keeping model default): {repr(e)}"
        )

def _download_file(url, destination_path):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    tmp_path = f"{destination_path}.tmp_{uuid.uuid4().hex}"
    print(f"Downloading file from {url} -> {destination_path}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, destination_path)
    print(f"Download complete: {destination_path}")

def get_pipeline():
    global pipe
    if pipe is None:
        hf_token = os.environ.get("HF_TOKEN")
        print(f"Loading model: {MODEL_ID}")
        pipe = ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None
        )

        pipe.to("cuda")
        print("Model loaded successfully.")
    return pipe

def get_img2img_pipeline():
    global img2img_pipe
    if img2img_pipe is None:
        base_pipe = get_pipeline()
        img2img_pipe = ZImageImg2ImgPipeline(**base_pipe.components)
        img2img_pipe.to("cuda")
        print("ZImageImg2ImgPipeline initialized from base pipeline components.")
    return img2img_pipe

def get_upscaler():
    global upscaler
    if upscaler is None:
        if not os.path.exists(UPSCALE_MODEL_PATH):
            _download_file(UPSCALE_MODEL_URL, UPSCALE_MODEL_PATH)

        # Lazy import to avoid startup overhead when second pass is disabled.
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upscaler = RealESRGANer(
            scale=4,
            model_path=UPSCALE_MODEL_PATH,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        print("RealESRGAN upscaler initialized.")
    return upscaler

def upscale_image(image, outscale):
    import cv2
    import numpy as np

    upsampler = get_upscaler()
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out_bgr, _ = upsampler.enhance(bgr, outscale=outscale)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)

def download_lora(url):
    """
    Downloads LoRA to ephemeral storage (/tmp).
    """
    local_path = f"/tmp/{uuid.uuid4()}.safetensors"
    print(f"Downloading LoRA from {url} to {local_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path

def handler(job):
    """
    The main RunPod serverless handler.
    """
    try:
        job_input = job.get("input", {})

        # 1. Parse Input with Defaults
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing 'prompt' in input."}

        lora_url = job_input.get("lora_url")
        negative_prompt = job_input.get("negative_prompt", "")
        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 1024))
        steps = int(job_input.get("steps", 50))
        guidance_scale = float(job_input.get("guidance_scale", 3.5))
        seed = int(job_input.get("seed", 42))
        lora_scale = float(job_input.get("lora_scale", 0.85))

        # Z-Image base defaults from official docs: cfg_normalization=False, cfg_truncation=1.0
        cfg_normalization = _to_bool(job_input.get("cfg_normalization"), default=False)
        cfg_truncation = float(job_input.get("cfg_truncation", 1.0))
        max_sequence_length = int(job_input.get("max_sequence_length", 512))

        # Enable beta-sigmas by default for cleaner denoising schedule; can be overridden per request.
        env_use_beta_sigmas = _to_bool(os.environ.get("USE_BETA_SIGMAS"), default=True)
        use_beta_sigmas = _to_bool(job_input.get("use_beta_sigmas"), default=env_use_beta_sigmas)

        # Optional second-pass refinement: upscale + Z-Image img2img
        second_pass_enabled_default = _to_bool(os.environ.get("SECOND_PASS_DEFAULT_ENABLED"), default=True)
        second_pass_enabled = _to_bool(
            job_input.get("second_pass_enabled"),
            default=second_pass_enabled_default,
        )
        second_pass_upscale = float(job_input.get("second_pass_upscale", 2.0))
        second_pass_strength = float(job_input.get("second_pass_strength", 0.3))
        second_pass_steps = int(job_input.get("second_pass_steps", 10))
        second_pass_guidance_scale = float(job_input.get("second_pass_guidance_scale", 2.8))
        second_pass_seed = int(job_input.get("second_pass_seed", seed))
        second_pass_cfg_normalization = _to_bool(
            job_input.get("second_pass_cfg_normalization"),
            default=False,
        )
        second_pass_cfg_truncation = float(job_input.get("second_pass_cfg_truncation", 1.0))
        second_pass_use_beta_sigmas = _to_bool(
            job_input.get("second_pass_use_beta_sigmas"),
            default=use_beta_sigmas,
        )

        # Keep VAE tiling off at 1024-ish outputs unless explicitly requested.
        vae_tiling_input = job_input.get("vae_tiling")
        if vae_tiling_input is None:
            vae_tiling = (width * height) > (1024 * 1024)
        else:
            vae_tiling = _to_bool(vae_tiling_input, default=False)

        second_pass_vae_tiling_input = job_input.get("second_pass_vae_tiling")
        if second_pass_vae_tiling_input is None:
            second_pass_vae_tiling = (width * height * (second_pass_upscale ** 2)) > (1024 * 1024)
        else:
            second_pass_vae_tiling = _to_bool(second_pass_vae_tiling_input, default=True)
        
        # Generate a unique adapter name for this request to avoid PEFT collisions
        request_id = str(uuid.uuid4())[:8]
        adapter_name = job_input.get("adapter_name", f"adapter_{request_id}")

        # 2. Setup Pipeline
        pipeline = get_pipeline()
        img2img_pipeline = get_img2img_pipeline() if second_pass_enabled else None
        _configure_scheduler(pipeline, use_beta_sigmas)
        if img2img_pipeline is not None:
            _configure_scheduler(img2img_pipeline, second_pass_use_beta_sigmas)

        if vae_tiling:
            pipeline.vae.enable_tiling()
        else:
            pipeline.vae.disable_tiling()
        if img2img_pipeline is not None:
            if second_pass_vae_tiling:
                img2img_pipeline.vae.enable_tiling()
            else:
                img2img_pipeline.vae.disable_tiling()
        print(
            f"Inference controls: use_beta_sigmas={use_beta_sigmas}, "
            f"cfg_normalization={cfg_normalization}, cfg_truncation={cfg_truncation}, "
            f"vae_tiling={vae_tiling}, second_pass_enabled={second_pass_enabled}"
        )
        
        # 3. Handle LoRA - Clean start every time to prevent artifacts and "smearing"
        # CRITICAL: In a persistent worker, we must unload old weights before loading new ones
        print("Unloading any existing LoRA weights for a clean slate...")
        pipeline.unload_lora_weights()
        if img2img_pipeline is not None and img2img_pipeline.transformer is not pipeline.transformer:
            img2img_pipeline.unload_lora_weights()
        
        lora_path = None
        if lora_url:
            lora_path = download_lora(lora_url)
            print(f"Loading LoRA from {lora_path} with scale {lora_scale}")
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])
            if img2img_pipeline is not None and img2img_pipeline.transformer is not pipeline.transformer:
                img2img_pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
                img2img_pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])
            print(f"LoRA '{adapter_name}' loaded successfully.")

        # 4. Generate Image
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Generating image: prompt='{prompt}', size={width}x{height}, seed={seed}, scale={guidance_scale}")
        
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            cfg_normalization=cfg_normalization,
            cfg_truncation=cfg_truncation,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

        # 4b. Optional second pass: upscale then refine with img2img
        if second_pass_enabled and img2img_pipeline is not None:
            print(
                "Running second pass refinement "
                f"(upscale={second_pass_upscale}, strength={second_pass_strength}, "
                f"steps={second_pass_steps}, guidance={second_pass_guidance_scale})"
            )
            upscaled_image = upscale_image(result, second_pass_upscale)
            second_generator = torch.Generator("cuda").manual_seed(second_pass_seed)
            result = img2img_pipeline(
                prompt=prompt,
                image=upscaled_image,
                negative_prompt=negative_prompt if negative_prompt else None,
                strength=second_pass_strength,
                num_inference_steps=second_pass_steps,
                guidance_scale=second_pass_guidance_scale,
                cfg_normalization=second_pass_cfg_normalization,
                cfg_truncation=second_pass_cfg_truncation,
                max_sequence_length=max_sequence_length,
                generator=second_generator,
            ).images[0]

        # 5. Save as High-Quality JPG
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        result.save(output_path, format="JPEG", quality=95)
        print(f"Image saved to {output_path}")

        # 6. Upload to S3
        s3_url = upload_image_to_s3(output_path, output_filename)
        print(f"Image uploaded to S3: {s3_url}")

        # 7. Cleanup (Ephemeral Storage)
        if lora_path and os.path.exists(lora_path):
            os.remove(lora_path)
        if os.path.exists(output_path):
            os.remove(output_path)

        return {"image_url": s3_url}

    except Exception as e:
        print(f"Error in handler: {repr(e)}")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
