"""Precompute + cache bge-base embeddings for passages and queries.

Passages (one per turn): encoded plainly (role: text), normalized -> emb/<hid>.npy
Queries: encoded with the bge retrieval instruction -> emb/queries.npy + queries_ids.json

Sharded by history across GPUs: run one process per GPU with --gpu i --num-gpus N.
The process with --gpu 0 also encodes the queries (small).

  CUDA_VISIBLE_DEVICES=i python -m mem27.planner.embed \
      --data data/lme_s_k --gpu i --num-gpus 4
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, OrderedDict

import numpy as np

from mem27.schema import load_notes, load_questions, EMBED_MODEL
from mem27.planner.refcommon import passage_text, BGE_QUERY_INSTRUCTION


def group_notes_by_history(notes):
    g = OrderedDict()
    for n in notes:
        g.setdefault(n.history_id, []).append(n)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s_k")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--do-queries", action="store_true",
                    help="force query encoding regardless of gpu id")
    args = ap.parse_args()

    emb_dir = os.path.join(args.data, "emb")
    os.makedirs(emb_dir, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.max_seq_length = 512

    notes = load_notes(os.path.join(args.data, "notes.jsonl"))
    by_hist = group_notes_by_history(notes)
    hids = list(by_hist.keys())
    shard = hids[args.gpu::args.num_gpus]
    print(f"[gpu{args.gpu}] device={device} histories={len(shard)}/{len(hids)}", flush=True)

    done = 0
    for hid in shard:
        out = os.path.join(emb_dir, f"{hid}.npy")
        if os.path.exists(out):
            done += 1
            continue
        hnotes = by_hist[hid]
        texts = [passage_text(n.role, n.text) for n in hnotes]
        vecs = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        np.save(out, vecs.astype(np.float32))
        done += 1
        if done % 25 == 0:
            print(f"[gpu{args.gpu}] {done}/{len(shard)} histories", flush=True)
    print(f"[gpu{args.gpu}] passages done ({done} histories)", flush=True)

    if args.gpu == 0 or args.do_queries:
        qs = load_questions(os.path.join(args.data, "questions.jsonl"))
        qtexts = [BGE_QUERY_INSTRUCTION + q.question for q in qs]
        qvecs = model.encode(qtexts, batch_size=args.batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        np.save(os.path.join(emb_dir, "queries.npy"), qvecs.astype(np.float32))
        json.dump([q.question_id for q in qs],
                  open(os.path.join(emb_dir, "queries_ids.json"), "w"))
        print(f"[gpu{args.gpu}] queries done ({len(qs)})", flush=True)


if __name__ == "__main__":
    main()
