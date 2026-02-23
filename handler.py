import time
t_start = time.time()
import runpod
import os
import torch
import requests
import uuid
import traceback
from PIL import Image
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler
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
        
        # Enable stable high-quality settings
        pipe.vae.enable_tiling()
        pipe.to("cuda")
        print("Model loaded successfully.")
    return pipe

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
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        steps = job_input.get("steps", 50)
        guidance_scale = job_input.get("guidance_scale", 3.5)
        seed = job_input.get("seed", 42)
        lora_scale = job_input.get("lora_scale", 0.85)
        
        # New high-quality parameters from Z-Image guide
        cfg_normalization = job_input.get("cfg_normalization", True)
        cfg_truncation = job_input.get("cfg_truncation", 0.8) # 0.8 fixes saturation/blurriness
        max_sequence_length = job_input.get("max_sequence_length", 512)
        
        # Generate a unique adapter name for this request to avoid PEFT collisions
        request_id = str(uuid.uuid4())[:8]
        adapter_name = job_input.get("adapter_name", f"adapter_{request_id}")

        # 2. Setup Pipeline
        pipeline = get_pipeline()
        
        # Enable VAE tiling for better quality and memory efficiency
        pipeline.vae.enable_tiling()

        # 3. Handle LoRA - Clean start every time
        # CRITICAL: Always unload previous LoRAs to prevent stacking/corruption in a persistent worker
        print("Unloading any existing LoRA weights...")
        pipeline.unload_lora_weights()
        
        lora_path = None
        if lora_url:
            lora_path = download_lora(lora_url)
            print(f"Loading LoRA from {lora_path} with scale {lora_scale}")
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])
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
