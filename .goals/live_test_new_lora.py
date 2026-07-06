#!/usr/bin/env python3
"""
Ad-hoc live test for the malcolmrey/zbase_aaliyah_v1 LoRA reported to produce
garbled output. Reuses live_test.py's fixed texture-test prompt (no trigger
word needed) but swaps in the new LoRA URL. Prints the full raw job result
(including any {"error": ...} from handler.py's top-level except) so we can
see whether the handler actually raised, versus silently returning a
corrupted-but-successful image.

Usage:
    python3 .goals/live_test_new_lora.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_test import FIXED_PROMPT, load_env, submit, poll_status  # noqa: E402

NEW_LORA_URL = "https://huggingface.co/malcolmrey/zbase/resolve/main/zbase_aaliyah_v1.safetensors"


def main():
    env = load_env()
    endpoint_url = env["RUNPOD_ENDPOINT_URL"]
    api_key = env["RUNPOD_API_KEY"]
    job_input = {
        "prompt": FIXED_PROMPT,
        "seed": 42,
        "loras": [{"url": NEW_LORA_URL, "scale": 0.85}],
    }

    result = submit(endpoint_url, api_key, job_input)
    if endpoint_url.rstrip("/").endswith("/run"):
        job_id = result["id"]
        print(f"submitted async job {job_id}, polling...", file=sys.stderr)
        result = poll_status(endpoint_url, api_key, job_id)

    out_path = Path(__file__).resolve().parent / "live_test_new_lora_result.json"
    out_path.write_text(json.dumps({"job_input": job_input, "result": result}, indent=2))

    print(f"status: {result.get('status')}")
    print(f"output: {json.dumps(result.get('output'), indent=2)}")
    print(f"saved full record to {out_path}")


if __name__ == "__main__":
    main()
