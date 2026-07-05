#!/usr/bin/env python3
"""
Live before/after quality-check helper for the image-quality-improvement goal.

Loads RUNPOD_ENDPOINT_URL / RUNPOD_API_KEY / TEST_LORA_URL from .env (repo
root), submits the fixed test prompt from .goals/image-quality-improvement.md
to the deployed RunPod endpoint, polls to completion if the endpoint is async
(/run), and prints only the resulting image URL(s) -- never the API key.

Usage:
    python3 .goals/live_test.py before
    python3 .goals/live_test.py after
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

FIXED_PROMPT = (
    "A three-quarter street-style fashion portrait of woman in mid-stride along an urban "
    "sidewalk, her body angled slightly toward camera with a natural, confident walking "
    "cadence — weight shifting forward onto her leading foot, arms in easy motion at her "
    "sides. Her expression is composed and forward-focused, chin level, gaze directed just "
    "past the lens with the cool indifference of someone entirely at ease in her own "
    "presence.She wears a voluminous beige chunky-knit sweater dress that falls to "
    "mid-thigh — the open shoulder detail exposing one collarbone cleanly, the decorative "
    "lacing at the chest rendered with visible cord tension and eyelet hardware catching "
    "ambient light. The knit structure shows individual stitch definition, the fabric "
    "draping with realistic weight and slight swing from the motion of her stride, "
    "producing natural asymmetric folds along the hem. Her legs are fitted in tall, "
    "form-hugging brown suede over-the-knee boots with a structured block heel — the "
    "suede surface showing its characteristic fine nap texture, subtle compression "
    "wrinkles behind the knee from movement, and a matte finish that absorbs rather than "
    "reflects the ambient light. Her right hand grips the top handle of a large, rigid "
    "structured handbag in matching warm beige — clean corners, a minimal clasp hardware "
    "detail, smooth leather surface with a faint specular highlight along the top edge. "
    "Her left arm swings naturally forward in walking rhythm. The setting is a quiet urban "
    "street with an unbroken flat gray concrete wall running parallel behind her — its "
    "surface showing fine aggregate texture, faint weathering marks, and a long "
    "directional shadow cast obliquely across it from soft overhead daylight. The "
    "sidewalk beneath is smooth pale concrete, a subtle extension of the muted pastel "
    "palette. The entire environment stays restrained — no signage, no distracting props "
    "— allowing the subject and garment textures to own the frame completely. Lighting is "
    "soft diffused daylight from a high overcast sky, approximately 5500K, producing "
    "gentle directional shadows that fall with enough depth to sculpt the knit fabric "
    "folds and give the suede boots dimensional presence without blowing any highlights. A "
    "faint secondary bounce from the pale concrete wall provides a cool fill from the "
    "right, keeping shadow areas luminous rather than dead. Shot on a Sony A7R V with a "
    "85mm f/1.8 lens, three-quarter framing from just below the boot heel to just above "
    "the crown — subject filling the vertical frame with breathing room on both sides. "
    "Subtle cinematic depth of field with the subject in full sharp focus, the concrete "
    "wall behind her transitioning to a very slight soft focus toward the frame edges. "
    "Individual knit fiber strands catching sidelight, suede nap microscopically "
    "rendered, natural hair movement from stride, fine skin texture on the exposed "
    "shoulder and collarbone. High-fashion street photography editorial style. Muted, "
    "sophisticated color palette anchored by warm beige and brown against cool concrete "
    "gray. Vibrant but restrained — no oversaturation. 8K resolution, high dynamic range, "
    "no watermarks, no text in frame, no motion blur on subject, clean wall background "
    "with no graffiti."
)


def load_env():
    values = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def build_job_input(env):
    return {
        "prompt": FIXED_PROMPT,
        "seed": 42,
        "loras": [{"url": env["TEST_LORA_URL"], "scale": 0.85}],
    }


def submit(endpoint_url, api_key, job_input):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(endpoint_url, headers=headers, json={"input": job_input}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def poll_status(endpoint_url, api_key, job_id, timeout_s=600, interval_s=5):
    status_url = re.sub(r"/run$", f"/status/{job_id}", endpoint_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.time()
    while time.time() - start < timeout_s:
        resp = requests.get(status_url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status in ("COMPLETED", "FAILED"):
            return data
        time.sleep(interval_s)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout_s}s")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "before"
    env = load_env()
    endpoint_url = env["RUNPOD_ENDPOINT_URL"]
    api_key = env["RUNPOD_API_KEY"]
    job_input = build_job_input(env)

    result = submit(endpoint_url, api_key, job_input)

    if endpoint_url.rstrip("/").endswith("/run"):
        job_id = result["id"]
        print(f"[{label}] submitted async job {job_id}, polling...", file=sys.stderr)
        result = poll_status(endpoint_url, api_key, job_id)

    out_path = Path(__file__).resolve().parent / f"live_test_{label}.json"
    out_path.write_text(json.dumps({"job_input": job_input, "result": result}, indent=2))

    print(f"[{label}] status: {result.get('status')}")
    output = result.get("output")
    print(f"[{label}] output: {json.dumps(output, indent=2)}")
    print(f"[{label}] saved full record to {out_path}")


if __name__ == "__main__":
    main()
