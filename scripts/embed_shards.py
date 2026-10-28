"""Sharded bge-base embedding of every unique turn text (CPU or GPU), then merge into an EmbCache.

  python scripts/embed_shards.py embed --data data/lme_s --shard 0 --n-shards 4 --out runs/_emb/bge_base_shards
  python scripts/embed_shards.py merge --shards runs/_emb/bge_base_shards --out runs/_emb/bge_base
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.embed import EmbCache, key  # noqa: E402
from mem27.vschema import iter_jsonl  # noqa: E402


def embed(a):
    texts = {}
    for d in a.data:
        for n in iter_jsonl(Path(d) / "notes.jsonl"):
            if n["turn_idx"] >= 0:
                texts.setdefault(key(n["text"]), n["text"])
    mine = sorted(k for k in texts if int(k[:8], 16) % a.n_shards == a.shard)
    print(f"shard {a.shard}/{a.n_shards}: {len(mine)} of {len(texts)} unique texts", flush=True)
    ec = EmbCache(Path(a.out) / f"shard{a.shard}", device=a.device)
    t0 = time.time()
    done = 0
    for i in range(0, len(mine), a.chunk):  # chunks so a timeout keeps progress
        part = [texts[k] for k in mine[i:i + a.chunk]]
        info = ec.add(part, batch_size=a.batch)
        done += info["new"]
        print(f"  {i + len(part)}/{len(mine)} new={info['new']} {time.time() - t0:.0f}s", flush=True)
    print(f"done {done} in {time.time() - t0:.0f}s", flush=True)


def merge(a):
    out = EmbCache(a.out)
    for sd in sorted(Path(a.shards).glob("shard*")):
        keys = json.loads((sd / "keys.json").read_text())
        mat = np.load(sd / "emb.npy")
        new = [i for i, k in enumerate(keys) if k not in out.keys]
        base = len(out.keys)
        for j, i in enumerate(new):
            out.keys[keys[i]] = base + j
        out.mat = np.concatenate([out.mat, mat[new]]) if len(out.mat) else mat[new]
        print(sd.name, len(keys), "merged", len(new))
    out.save()
    print("total", len(out.keys), out.mat.shape)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--data", nargs="+", required=True)
    e.add_argument("--shard", type=int, required=True)
    e.add_argument("--n-shards", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default=None)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--chunk", type=int, default=8192)
    m = sub.add_parser("merge")
    m.add_argument("--shards", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    embed(a) if a.cmd == "embed" else merge(a)


if __name__ == "__main__":
    main()
