"""Text encoders behind one interface: real bge embeddings on the cluster, a hashing stub for offline tests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

_TOKEN = re.compile(r"\w+")


class Encoder(Protocol):
    """Maps texts to L2-normalised float32 row vectors; queries and passages may be encoded differently."""

    name: str

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # empty texts stay zero vectors instead of becoming NaN
    return (matrix / norms).astype(np.float32)


class StubEncoder:
    """Deterministic hashed bag-of-words. No downloads, no GPU; only for tests and laptop smoke runs."""

    name = "stub-hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        return digest % self.dim, (1.0 if (digest >> 32) & 1 else -1.0)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                col, sign = self._bucket(token)
                matrix[row, col] += sign
        return _normalise(matrix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_passages(texts)


class BgeEncoder:
    """BAAI/bge-base-en-v1.5 via sentence-transformers: fp16 on GPU, fp32 on CPU, bge query instruction prefix."""

    def __init__(self, model: str, query_prefix: str, batch_size: int = 256, dtype: str = "fp16"):
        from sentence_transformers import SentenceTransformer  # ty: ignore[unresolved-import]  # optional `encode` extra

        self.name = model
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(model)
        self.device = str(self._model.device)
        if dtype == "fp16" and self.device.startswith("cuda"):
            self._model.half()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


def make_encoder(kind: str, config: dict[str, Any]) -> Encoder:
    """`kind` is "stub" (tests, laptop) or "bge" (cluster); unknown kinds fail loudly."""
    if kind == "stub":
        return StubEncoder()
    if kind == "bge":
        enc = config["encoder"]
        return BgeEncoder(enc["model"], enc["query_prefix"], enc["batch_size"], enc["dtype"])
    raise ValueError(f"unknown encoder kind {kind!r}; use 'stub' or 'bge'")
