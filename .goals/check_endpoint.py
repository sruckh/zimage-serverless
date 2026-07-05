#!/usr/bin/env python3
"""Inspect endpoint + worker state via the RunPod REST management API."""
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
    url = f"https://rest.runpod.io/v1/endpoints/{endpoint_id}?includeWorkers=true"
    resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    data = resp.json()
    # The API returns each worker's full env block in plaintext (secrets included) --
    # never print that. Only surface the fields we actually need to diagnose scaling.
    workers = data.get("workers", [])
    summary = {
        "status_code": resp.status_code,
        "id": data.get("id"),
        "name": data.get("name"),
        "workersMin": data.get("workersMin"),
        "workersMax": data.get("workersMax"),
        "scalerType": data.get("scalerType"),
        "scalerValue": data.get("scalerValue"),
        "workers": [
            {
                "id": w.get("id"),
                "desiredStatus": w.get("desiredStatus"),
                "lastStatusChange": w.get("lastStatusChange"),
                "imageName": w.get("imageName"),
                "gpuTypeId": w.get("gpuTypeId"),
            }
            for w in workers
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
