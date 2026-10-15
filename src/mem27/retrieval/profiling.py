"""Build-cost measurement and run-environment capture (every reported number must be traceable)."""

from __future__ import annotations

import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter

import numpy as np

TRACKED_LIBS = ("numpy", "rank-bm25", "faiss-cpu", "tiktoken", "sentence-transformers", "torch", "mem27")
BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024  # macOS bytes, Linux KiB


def disk_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


@contextmanager
def measure_build(index_dir: str | Path | None = None) -> Generator[dict[str, float]]:
    """Yields a dict filled on exit with build_ms, peak_rss_mb (process high-water mark) and disk_bytes."""
    profile: dict[str, float] = {}
    began = perf_counter()
    try:
        yield profile
    finally:
        profile["build_ms"] = (perf_counter() - began) * 1000.0
        profile["peak_rss_mb"] = peak_rss_mb()
        profile["disk_bytes"] = disk_bytes(index_dir) if index_dir is not None else 0


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _gpu() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return f"nvidia-smi failed: {exc}"
    return out.stdout.strip() or None


def _lib_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_env(seed: int, cache: str) -> dict:
    """Machine and software facts written to env.json next to every run (constitution V)."""
    return {
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libs": {name: _lib_version(name) for name in TRACKED_LIBS},
        "blas_threads": {var: os.environ.get(var) for var in BLAS_VARS},
        "seed": seed,
        "cache": cache,
        "captured_at": datetime.now(UTC).isoformat(),
    }


def project_linear(points: list[tuple[float, float]], targets: list[float]) -> dict[float, float]:
    """Least-squares line through (notes, cost) points, evaluated at the target note counts."""
    if len(points) < 2:
        raise ValueError("need at least two (notes, cost) points to project; measure more scales")
    xs, ys = np.array([p[0] for p in points], dtype=float), np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return {t: float(slope * t + intercept) for t in targets}
