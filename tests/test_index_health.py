"""H5 support: brute-force ground truth is exact; recall is 1 for exact search and drops for corrupted ids."""
import numpy as np

from mem27.eval.index_health import brute_force_topk, recall_at_k, sample_probes


def test_bruteforce_and_recall():
    rng = np.random.default_rng(0)
    e = rng.standard_normal((2000, 16)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    gt = brute_force_topk(e, e[:40], k=10, chunk=333)  # chunking must not change the result
    assert (gt[:, 0] == np.arange(40)).all()
    full = np.argsort(-(e[:40] @ e.T), axis=1)[:, :10]
    assert recall_at_k(full, gt) == 1.0
    bad = full.copy()
    bad[:, 5:] = 1999
    assert recall_at_k(bad, gt) < 0.6
    p = sample_probes(e, 50, seed=1)
    assert p.shape == (50, 16) and np.allclose(np.linalg.norm(p, axis=1), 1, atol=1e-5)
