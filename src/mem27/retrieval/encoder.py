"""Text encoders behind one interface: a real embedding model on the cluster, a hashing stub for offline tests."""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Callable, Sequence
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


def token_budget_batches(lengths: Sequence[int], max_tokens: int, max_batch: int) -> list[list[int]]:
    """Groups text indices so each padded batch (rows × longest row) stays within `max_tokens`.

    Texts are taken longest first, so long texts land in small batches and short ones in large batches. A single
    text longer than the budget gets a batch of its own (the model truncates it to max_seq_length).
    """
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    for idx in order:
        longest = lengths[current[0]] if current else lengths[idx]
        if current and (len(current) + 1 > max_batch or longest * (len(current) + 1) > max_tokens):
            batches.append(current)
            current = []
        current.append(idx)
    if current:
        batches.append(current)
    return batches


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def encode_with_backoff(
    encode: Callable[[list[str]], np.ndarray], texts: list[str], on_oom: Callable[[], None]
) -> np.ndarray:
    """Encodes `texts`, halving the batch on GPU out-of-memory until it fits; row order is preserved."""
    try:
        return encode(texts)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        on_oom()
        if len(texts) == 1:
            raise RuntimeError(
                "a single text does not fit in GPU memory; lower encoder.max_seq_length in configs/retrieval.yaml"
            ) from exc
    middle = len(texts) // 2
    return np.vstack(
        [encode_with_backoff(encode, texts[:middle], on_oom), encode_with_backoff(encode, texts[middle:], on_oom)]
    )


class SentenceTransformerEncoder:
    """A sentence-transformers model (default Qwen/Qwen3-Embedding-0.6B): fp16 on GPU, fp32 on CPU.

    Queries get the configured instruction prefix; passages are encoded as-is (Qwen3-Embedding convention).
    Inputs longer than `max_seq_length` tokens are truncated by the model's own tokenizer. Batches are sized by
    token count (`max_tokens_per_batch`), and a batch that still runs out of GPU memory is split and retried.
    """

    def __init__(
        self,
        model: str,
        query_prefix: str,
        batch_size: int,
        dtype: str,
        max_seq_length: int,
        max_tokens_per_batch: int,
    ):
        import torch  # ty: ignore[unresolved-import]  # optional `encode` extra
        from sentence_transformers import SentenceTransformer  # ty: ignore[unresolved-import]

        self.name = model
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_tokens_per_batch = max_tokens_per_batch
        self._torch = torch
        on_gpu = torch.cuda.is_available()
        model_kwargs = {"torch_dtype": torch.float16} if (dtype == "fp16" and on_gpu) else {}
        # sentence-transformers renamed tokenizer_kwargs → processor_kwargs; use whichever this version accepts.
        params = inspect.signature(SentenceTransformer.__init__).parameters
        padding_key = "processor_kwargs" if "processor_kwargs" in params else "tokenizer_kwargs"
        self._model = SentenceTransformer(model, model_kwargs=model_kwargs, **{padding_key: {"padding_side": "left"}})
        self._model.max_seq_length = max_seq_length
        self.device = str(self._model.device)

    def _token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self._model.tokenizer(texts, truncation=True, max_length=self.max_seq_length)
        return [len(ids) for ids in encoded["input_ids"]]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _free_gpu(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        batches = token_budget_batches(self._token_lengths(items), self.max_tokens_per_batch, self.batch_size)
        rows: list[np.ndarray] = [np.empty(0)] * len(items)
        for batch in batches:
            vectors = encode_with_backoff(self._encode_batch, [items[i] for i in batch], self._free_gpu)
            for position, idx in enumerate(batch):
                rows[idx] = vectors[position]
        return np.vstack(rows).astype(np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


def make_encoder(kind: str, config: dict[str, Any]) -> Encoder:
    """`kind` is "stub" (tests, laptop) or "model" (cluster: config encoder.model); unknown kinds fail loudly."""
    if kind == "stub":
        return StubEncoder()
    if kind == "model":
        enc = config["encoder"]
        return SentenceTransformerEncoder(
            enc["model"],
            enc["query_prefix"],
            enc["batch_size"],
            enc["dtype"],
            enc["max_seq_length"],
            enc["max_tokens_per_batch"],
        )
    raise ValueError(f"unknown encoder kind {kind!r}; use 'stub' or 'model'")
