#!/usr/bin/env python3
"""Poll a RunPod async job status once and print it. Usage: check_status.py <job_id>"""
import json
import re
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def load_env():
    values = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def main():
    job_id = sys.argv[1]
    env = load_env()
    endpoint_url = env["RUNPOD_ENDPOINT_URL"]
    api_key = env["RUNPOD_API_KEY"]
    status_url = re.sub(r"/run$", f"/status/{job_id}", endpoint_url)
    resp = requests.get(status_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2)[:3000])


if __name__ == "__main__":
    main()
