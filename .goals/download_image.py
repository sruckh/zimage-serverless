#!/usr/bin/env python3
"""Download an image from a URL to a local path. Usage: download_image.py <url> <out_path>"""
import sys

import requests


def main():
    url, out_path = sys.argv[1], sys.argv[2]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"downloaded {len(resp.content)} bytes to {out_path}")


if __name__ == "__main__":
    main()
