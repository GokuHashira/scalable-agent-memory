"""EVAL-04: identical runs -> CI contains 0 (equivalent); shifted runs -> CI excludes 0."""
import numpy as np

from mem27.eval.bootstrap import paired_bootstrap


def test_identical_runs_equivalent():
    a = np.random.default_rng(1).integers(0, 2, 400).astype(float)
    r = paired_bootstrap(a, a.copy())
    assert r["ci_lo"] <= 0 <= r["ci_hi"] and r["verdict"] == "equivalent" and r["discordant"] == 0


def test_shifted_runs_excluded():
    rng = np.random.default_rng(2)
    b = rng.integers(0, 2, 400).astype(float)
    a = b.copy()
    a[np.where(b == 0)[0][:60]] = 1.0  # +15 pts
    r = paired_bootstrap(a, b)
    assert r["ci_lo"] > 0 and r["verdict"] == "better" and abs(r["delta"] - 0.15) < 1e-9
    r2 = paired_bootstrap(b, a)
    assert r2["ci_hi"] < 0 and r2["verdict"] == "worse"


def test_small_noisy_difference_inconclusive():
    rng = np.random.default_rng(3)
    b = rng.integers(0, 2, 400).astype(float)
    a = b.copy()
    flip = rng.choice(400, 80, replace=False)  # many discordant pairs, ~0 net shift
    a[flip] = 1 - a[flip]
    r = paired_bootstrap(a, b)
    assert r["verdict"] in ("inconclusive", "equivalent") and r["ci_lo"] < 0.02
    assert r["mde80"] > 0 and r["n_for_equiv"] > 0


def test_deterministic_seed():
    rng = np.random.default_rng(4)
    a, b = rng.random(100), rng.random(100)
    assert paired_bootstrap(a, b, seed=0) == paired_bootstrap(a, b, seed=0)


def test_single_bootstrap_ci():
    from mem27.eval.bootstrap import single_bootstrap
    x = np.array([1.0] * 70 + [0.0] * 30)
    r = single_bootstrap(x)
    assert abs(r["mean"] - 0.7) < 1e-12 and r["ci_lo"] < 0.7 < r["ci_hi"]
    # normal-approximation half-width 1.96*sqrt(.21/100) = 0.0898; the percentile CI should be close
    assert abs((r["ci_hi"] - r["ci_lo"]) / 2 - 0.0898) < 0.015
    c = single_bootstrap(np.ones(50))
    assert c["ci_lo"] == c["ci_hi"] == 1.0
    assert single_bootstrap(x, seed=0) == single_bootstrap(x, seed=0)
