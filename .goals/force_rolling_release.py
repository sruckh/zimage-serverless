#!/usr/bin/env python3
"""Force a RunPod endpoint rolling release via a no-op PATCH, per
https://docs.runpod.io/api-reference/endpoints/PATCH/endpoints/endpointId
"""
import json
import re
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
    env = load_env()
    endpoint_url = env["RUNPOD_ENDPOINT_URL"]
    api_key = env["RUNPOD_API_KEY"]
    match = re.search(r"/v2/([^/]+)/", endpoint_url + "/")
    endpoint_id = match.group(1)
    url = f"https://rest.runpod.io/v1/endpoints/{endpoint_id}"
    resp = requests.patch(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={},
        timeout=60,
    )
    print(resp.status_code)
    try:
        data = resp.json()
        print(json.dumps({k: data.get(k) for k in ("id", "name", "lastModifiedAt")}, indent=2))
    except ValueError:
        print(resp.text[:500])


if __name__ == "__main__":
    main()
