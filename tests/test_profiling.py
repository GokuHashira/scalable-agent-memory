"""Build profiling, environment capture, and the linear scale projection."""

from mem27.retrieval.profiling import capture_env, measure_build, project_linear


def test_measure_build_records_nonnegative(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"x" * 1234)
    with measure_build(tmp_path) as profile:
        sum(range(10_000))
    assert profile["build_ms"] >= 0
    assert profile["peak_rss_mb"] > 0
    assert profile["disk_bytes"] == 1234


def test_capture_env_fields():
    env = capture_env(seed=0, cache="warm")
    for key in ("host", "cpu_model", "cpu_count", "gpu", "python", "libs", "blas_threads", "seed", "cache"):
        assert key in env
    assert env["seed"] == 0 and env["cache"] == "warm"


def test_project_linear_exact_on_line():
    points = [(1_000, 10.0), (2_000, 20.0), (4_000, 40.0)]
    proj = project_linear(points, targets=[100_000, 1_000_000])
    assert abs(proj[100_000] - 1_000.0) < 1e-6
    assert abs(proj[1_000_000] - 10_000.0) < 1e-6
