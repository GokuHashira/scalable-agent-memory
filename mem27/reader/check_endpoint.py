"""Ping a vLLM OpenAI-compatible endpoint and do one sanity completion.

  python -m mem27.reader.check_endpoint --base http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import json
import urllib.request


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    args = ap.parse_args()
    models = _get(args.base + "/models")
    mid = models["data"][0]["id"]
    print("endpoint up; model:", mid)
    out = _post(args.base + "/chat/completions", {
        "model": mid, "temperature": 0.0, "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}]})
    print("sanity completion:", out["choices"][0]["message"]["content"].strip())


if __name__ == "__main__":
    main()
