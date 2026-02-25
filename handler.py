import time
t_start = time.time()
import runpod
import os
import gc
import torch
from torch import nn
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
UPSCALE_USE_CUDA_ENV = os.environ.get("UPSCALE_USE_CUDA")

class DCCM(nn.Sequential):
    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )

class PLKConv2d(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        self.idx = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x

class EA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)

class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ):
        super().__init__()
        self.channel_mixer = DCCM(dim)
        pdim = int(dim * split_ratio)
        self.lk = PLKConv2d(pdim, kernel_size)
        self.attn = EA(dim) if use_ea else nn.Identity()
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm = nn.GroupNorm(norm_groups, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)
        return x + x_skip

class RealPLKSR(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = 4,
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        use_ea: bool = True,
        norm_groups: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.upscale = upscaling_factor
        self.feats = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea) for _ in range(n_blocks)]
            + [nn.Dropout2d(dropout)]
            + [nn.Conv2d(dim, out_ch * upscaling_factor**2, 3, 1, 1)]
        )
        self.to_img = nn.PixelShuffle(upscaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats(x) + torch.repeat_interleave(x, repeats=self.upscale**2, dim=1)
        return self.to_img(x)

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

def _free_cuda_cache(stage_label=None):
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    if stage_label:
        print(f"Cleared CUDA cache at stage: {stage_label}")

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
        loadnet = torch.load(UPSCALE_MODEL_PATH, map_location="cpu")
        if isinstance(loadnet, dict):
            if "params_ema" in loadnet and isinstance(loadnet["params_ema"], dict):
                state_dict = loadnet["params_ema"]
            elif "params" in loadnet and isinstance(loadnet["params"], dict):
                state_dict = loadnet["params"]
            elif "state_dict" in loadnet and isinstance(loadnet["state_dict"], dict):
                state_dict = loadnet["state_dict"]
            else:
                nested_dicts = [v for v in loadnet.values() if isinstance(v, dict)]
                matched = None
                for candidate in nested_dicts:
                    if "feats.0.weight" in candidate:
                        matched = candidate
                        break
                state_dict = matched if matched is not None else loadnet
        else:
            state_dict = loadnet

        if not isinstance(state_dict, dict) or "feats.0.weight" not in state_dict:
            raise RuntimeError(
                "Upscaler checkpoint is not a supported RealPLKSR-style state dict."
            )

        # Remove common wrappers like `module.`
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        dim = int(state_dict["feats.0.weight"].shape[0])
        n_blocks = len([k for k in state_dict.keys() if k.startswith("feats.") and k.endswith(".channel_mixer.0.weight")])
        kernel_size = int(state_dict["feats.1.lk.conv.weight"].shape[2])
        split_ratio = float(state_dict["feats.1.lk.conv.weight"].shape[0]) / float(dim)
        use_ea = "feats.1.attn.f.0.weight" in state_dict

        feat_weight_indices = []
        for key in state_dict.keys():
            if key.startswith("feats.") and key.endswith(".weight"):
                try:
                    feat_weight_indices.append(int(key.split(".")[1]))
                except Exception:
                    pass
        if not feat_weight_indices:
            raise RuntimeError("Could not infer output conv index from upscaler checkpoint.")
        last_feat_idx = max(feat_weight_indices)
        out_channels = int(state_dict[f"feats.{last_feat_idx}.weight"].shape[0])
        scale = int((out_channels // 3) ** 0.5)

        model = RealPLKSR(
            in_ch=3,
            out_ch=3,
            dim=dim,
            n_blocks=n_blocks,
            upscaling_factor=scale,
            kernel_size=kernel_size,
            split_ratio=split_ratio,
            use_ea=use_ea,
            norm_groups=4,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load RealPLKSR upscaler checkpoint with strict key matching: {e}"
            ) from e

        use_cuda_for_upscaler = _to_bool(UPSCALE_USE_CUDA_ENV, default=False) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda_for_upscaler else "cpu")
        model = model.to(device).eval()
        if device.type == "cuda":
            model = model.half()
        upscaler = {
            "model": model,
            "scale": scale,
            "device": device,
            "half": device.type == "cuda",
        }
        print(f"RealPLKSR upscaler initialized (scale={scale}, device={device.type}).")
    return upscaler

def upscale_image(image, outscale):
    import numpy as np

    upsampler = get_upscaler()
    model = upsampler["model"]
    model_scale = float(upsampler["scale"])
    device = upsampler["device"]
    use_half = bool(upsampler["half"])

    rgb = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    if use_half:
        tensor = tensor.half()

    with torch.inference_mode():
        output = model(tensor).clamp(0, 1)
    output = output.float().cpu().squeeze(0).permute(1, 2, 0).numpy()

    target_w = int(round(image.width * outscale))
    target_h = int(round(image.height * outscale))
    native_w = int(round(image.width * model_scale))
    native_h = int(round(image.height * model_scale))

    out_img = Image.fromarray((output * 255.0).round().astype(np.uint8))
    if (target_w, target_h) != (native_w, native_h):
        out_img = out_img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    return out_img

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

def _normalize_lora_key_prefix(key):
    normalized = key
    for prefix in ("base_model.model.", "diffusion_model.", "transformer."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized

def _looks_like_flux2_klein_lora(state_dict):
    markers = (
        "double_blocks.",
        "single_blocks.",
        "img_attn.qkv.",
        "txt_attn.qkv.",
        "img_mlp.0.",
        "txt_mlp.0.",
    )
    for key in state_dict.keys():
        normalized = _normalize_lora_key_prefix(key)
        if any(marker in normalized for marker in markers):
            return True
    return False

def _convert_flux2_klein_lora_to_diffusers(state_dict):
    """
    Convert Flux2/Klein-style LoRA keys (double_blocks/single_blocks) to the
    transformer key layout expected by Diffusers Z-Image PEFT loading.
    """
    converted_state_dict = {}
    original_state_dict = {
        _normalize_lora_key_prefix(k): v for k, v in state_dict.items()
    }

    has_lora_down_up = any("lora_down" in k or "lora_up" in k for k in original_state_dict.keys())
    if has_lora_down_up:
        temp_state_dict = {}
        for k, v in original_state_dict.items():
            temp_state_dict[k.replace("lora_down", "lora_A").replace("lora_up", "lora_B")] = v
        original_state_dict = temp_state_dict

    num_double_layers = 0
    num_single_layers = 0
    for key in original_state_dict.keys():
        if key.startswith("single_blocks."):
            num_single_layers = max(num_single_layers, int(key.split(".")[1]) + 1)
        elif key.startswith("double_blocks."):
            num_double_layers = max(num_double_layers, int(key.split(".")[1]) + 1)

    lora_keys = ("lora_A", "lora_B")
    attn_types = ("img_attn", "txt_attn")

    for sl in range(num_single_layers):
        single_block_prefix = f"single_blocks.{sl}"
        attn_prefix = f"single_transformer_blocks.{sl}.attn"
        for lora_key in lora_keys:
            linear1_key = f"{single_block_prefix}.linear1.{lora_key}.weight"
            if linear1_key in original_state_dict:
                converted_state_dict[f"{attn_prefix}.to_qkv_mlp_proj.{lora_key}.weight"] = original_state_dict.pop(
                    linear1_key
                )
            linear2_key = f"{single_block_prefix}.linear2.{lora_key}.weight"
            if linear2_key in original_state_dict:
                converted_state_dict[f"{attn_prefix}.to_out.{lora_key}.weight"] = original_state_dict.pop(linear2_key)

    for dl in range(num_double_layers):
        transformer_block_prefix = f"transformer_blocks.{dl}"

        for lora_key in lora_keys:
            for attn_type in attn_types:
                attn_prefix = f"{transformer_block_prefix}.attn"
                qkv_key = f"double_blocks.{dl}.{attn_type}.qkv.{lora_key}.weight"
                if qkv_key not in original_state_dict:
                    continue

                fused_qkv_weight = original_state_dict.pop(qkv_key)
                if lora_key == "lora_A":
                    proj_keys = (
                        ["to_q", "to_k", "to_v"]
                        if attn_type == "img_attn"
                        else ["add_q_proj", "add_k_proj", "add_v_proj"]
                    )
                    for proj_key in proj_keys:
                        converted_state_dict[f"{attn_prefix}.{proj_key}.{lora_key}.weight"] = fused_qkv_weight
                else:
                    sample_q, sample_k, sample_v = torch.chunk(fused_qkv_weight, 3, dim=0)
                    if attn_type == "img_attn":
                        converted_state_dict[f"{attn_prefix}.to_q.{lora_key}.weight"] = sample_q
                        converted_state_dict[f"{attn_prefix}.to_k.{lora_key}.weight"] = sample_k
                        converted_state_dict[f"{attn_prefix}.to_v.{lora_key}.weight"] = sample_v
                    else:
                        converted_state_dict[f"{attn_prefix}.add_q_proj.{lora_key}.weight"] = sample_q
                        converted_state_dict[f"{attn_prefix}.add_k_proj.{lora_key}.weight"] = sample_k
                        converted_state_dict[f"{attn_prefix}.add_v_proj.{lora_key}.weight"] = sample_v

        proj_mappings = [
            ("img_attn.proj", "attn.to_out.0"),
            ("txt_attn.proj", "attn.to_add_out"),
        ]
        for org_proj, diff_proj in proj_mappings:
            for lora_key in lora_keys:
                original_key = f"double_blocks.{dl}.{org_proj}.{lora_key}.weight"
                if original_key in original_state_dict:
                    converted_state_dict[f"{transformer_block_prefix}.{diff_proj}.{lora_key}.weight"] = (
                        original_state_dict.pop(original_key)
                    )

        mlp_mappings = [
            ("img_mlp.0", "ff.linear_in"),
            ("img_mlp.2", "ff.linear_out"),
            ("txt_mlp.0", "ff_context.linear_in"),
            ("txt_mlp.2", "ff_context.linear_out"),
        ]
        for org_mlp, diff_mlp in mlp_mappings:
            for lora_key in lora_keys:
                original_key = f"double_blocks.{dl}.{org_mlp}.{lora_key}.weight"
                if original_key in original_state_dict:
                    converted_state_dict[f"{transformer_block_prefix}.{diff_mlp}.{lora_key}.weight"] = (
                        original_state_dict.pop(original_key)
                    )

    extra_mappings = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "final_layer.linear": "proj_out",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    }
    for org_key, diff_key in extra_mappings.items():
        for lora_key in lora_keys:
            original_key = f"{org_key}.{lora_key}.weight"
            if original_key in original_state_dict:
                converted_state_dict[f"{diff_key}.{lora_key}.weight"] = original_state_dict.pop(original_key)

    # Ignore alpha and known metadata-ish keys that are not needed for PEFT injection.
    leftovers = [k for k in original_state_dict.keys() if ".alpha" not in k]
    if leftovers:
        raise ValueError(f"Unconverted Flux2/Klein LoRA keys remain: {leftovers[:8]}")

    converted_state_dict = {f"transformer.{k}": v for k, v in converted_state_dict.items()}
    return converted_state_dict

def _load_lora_with_target_module_fallback(pipeline, lora_path, adapter_name, lora_scale):
    """
    Load a LoRA adapter and retry without adapter metadata when PEFT target module
    names in metadata don't match the current Z-Image transformer module names.
    """
    try:
        pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
    except Exception as e:
        message = str(e)
        target_module_mismatch = "Target modules" in message and "not found in the base model" in message
        if not target_module_mismatch:
            raise

        if not hasattr(pipeline, "lora_state_dict") or not hasattr(pipeline, "load_lora_into_transformer"):
            raise

        print(
            "LoRA metadata target module mismatch detected; retrying without metadata "
            f"(adapter='{adapter_name}')."
        )
        pipeline.unload_lora_weights()

        state_payload = pipeline.lora_state_dict(lora_path, return_lora_metadata=True)
        if isinstance(state_payload, tuple):
            state_dict = state_payload[0]
        else:
            state_dict = state_payload

        if _looks_like_flux2_klein_lora(state_dict):
            print("Detected Flux2/Klein LoRA format; converting keys for Z-Image transformer.")
            state_dict = _convert_flux2_klein_lora_to_diffusers(state_dict)

        pipeline.load_lora_into_transformer(
            state_dict=state_dict,
            transformer=pipeline.transformer,
            adapter_name=adapter_name,
            metadata=None,
            _pipeline=pipeline,
        )

    pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])

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
        second_pass_upscale = float(job_input.get("second_pass_upscale", 1.5))
        second_pass_strength = float(job_input.get("second_pass_strength", 0.18))
        second_pass_steps = int(job_input.get("second_pass_steps", 8))
        second_pass_guidance_scale = float(job_input.get("second_pass_guidance_scale", 1.2))
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
            _load_lora_with_target_module_fallback(pipeline, lora_path, adapter_name, lora_scale)
            if img2img_pipeline is not None and img2img_pipeline.transformer is not pipeline.transformer:
                _load_lora_with_target_module_fallback(
                    img2img_pipeline, lora_path, adapter_name, lora_scale
                )
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
            _free_cuda_cache("before_second_pass_img2img")
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
