#!/usr/bin/env python3
"""Control test: identical prompt/seed, NO LoRA at all, to check whether the
LoRA is actually contributing anything beyond what the text prompt alone
would produce."""
import json
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

PROMPT = (
    "K1mScum, A poised middle-aged woman with medium-length chestnut-brown hair "
    "styled in soft loose waves just past her shoulders, warm hazel eyes, subtle "
    "laugh lines framing her mouth, natural matte skin texture, wearing a "
    "tailored charcoal blazer over a simple blouse, seated or standing facing "
    "the camera in a classic headshot pose with shoulders slightly angled, soft "
    "even studio lighting with a gentle catchlight in the eyes, plain neutral "
    "gray or muted blue backdrop, sharp focus on the face with a subtle falloff "
    "toward the background, close-up framing from the chest up, warm and "
    "approachable professional expression, clean corporate portrait style with "
    "natural color grading."
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


def build_job_input():
    return {
        "prompt": PROMPT,
        "seed": 42,
        "width": 864,
        "height": 1152,
        "second_pass_enabled": True,
        # deliberately no "loras" key at all
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
    label = "no_lora"
    env = load_env()
    endpoint_url = env["RUNPOD_ENDPOINT_URL"]
    api_key = env["RUNPOD_API_KEY"]
    job_input = build_job_input()

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
