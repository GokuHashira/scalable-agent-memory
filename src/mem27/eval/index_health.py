"""Index-health scoring for H5 (V supports G): ANN recall@k against brute force at stream checkpoints.

    probes = sample_probes(emb, n=1000, seed=0)            # probe vectors (perturbed corpus rows)
    gt     = brute_force_topk(emb[:n_live], probes, k=10)  # exact ground truth on the live set
    r      = recall_at_k(ann_ids, gt)                      # ann_ids from G's index.search(probes, k)

`checkpoint_report()` runs the whole loop for a faiss index plus the matching live embedding matrix,
returning recall@k and search latency; `demo()` shows incremental HNSW insertion over a stream.
"""
from __future__ import annotations

import time

import numpy as np


def sample_probes(emb: np.ndarray, n: int = 1000, seed: int = 0, noise: float = 0.05) -> np.ndarray:
    """Probe queries = random corpus rows + small Gaussian noise, re-normalised (query-like, never exact)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb), size=min(n, len(emb)), replace=False)
    q = emb[idx].astype(np.float32) + noise * rng.standard_normal((len(idx), emb.shape[1])).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def brute_force_topk(emb: np.ndarray, queries: np.ndarray, k: int = 10, chunk: int = 65536) -> np.ndarray:
    """Exact inner-product top-k (cosine on normalised vectors), chunked over the corpus."""
    q = queries.astype(np.float32)
    best_s = np.full((len(q), k), -np.inf, np.float32)
    best_i = np.zeros((len(q), k), np.int64)
    for s in range(0, len(emb), chunk):
        block = emb[s:s + chunk].astype(np.float32)
        sc = q @ block.T
        part = np.argpartition(-sc, min(k, sc.shape[1] - 1), axis=1)[:, :k]
        ps = np.take_along_axis(sc, part, 1)
        cs = np.concatenate([best_s, ps], 1)
        ci = np.concatenate([best_i, part + s], 1)
        o = np.argsort(-cs, 1)[:, :k]
        best_s, best_i = np.take_along_axis(cs, o, 1), np.take_along_axis(ci, o, 1)
    return best_i


def recall_at_k(ann_ids: np.ndarray, gt: np.ndarray) -> float:
    k = gt.shape[1]
    return float(np.mean([len(set(a[:k]) & set(g)) / k for a, g in zip(ann_ids, gt)]))


def checkpoint_report(index, live_emb: np.ndarray, probes: np.ndarray, k: int = 10) -> dict:
    gt = brute_force_topk(live_emb, probes, k)
    t0 = time.perf_counter()
    _, ann = index.search(probes.astype(np.float32), k)
    ms = (time.perf_counter() - t0) * 1000 / len(probes)
    return {"n_live": len(live_emb), f"recall@{k}": recall_at_k(ann, gt), "search_ms_per_query": ms}


def demo(n: int = 50_000, d: int = 64, checkpoints: int = 5, seed: int = 0) -> list[dict]:
    """Synthetic sanity run: HNSW built on the first half, the rest inserted incrementally."""
    import faiss
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    half = n // 2
    idx.add(emb[:half])
    probes = sample_probes(emb, 500, seed)
    out = [checkpoint_report(idx, emb[:half], probes)]
    step = (n - half) // checkpoints
    for c in range(checkpoints):
        lo, hi = half + c * step, half + (c + 1) * step
        idx.add(emb[lo:hi])
        out.append(checkpoint_report(idx, emb[:hi], probes))
    return out


if __name__ == "__main__":
    for row in demo():
        print(row)
