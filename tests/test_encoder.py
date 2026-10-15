"""Token-budget batching and OOM back-off for the real encoder (no torch needed: fake encode functions)."""

import numpy as np
import pytest

from mem27.retrieval.encoder import encode_with_backoff, token_budget_batches


def test_batches_respect_token_budget_and_cover_all():
    lengths = [10, 2000, 50, 2000, 300, 5, 1500]
    batches = token_budget_batches(lengths, max_tokens=4096, max_batch=64)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for batch in batches:
        padded = max(lengths[i] for i in batch) * len(batch)  # padding makes every row as long as the longest
        assert padded <= 4096 or len(batch) == 1


def test_short_texts_share_large_batches():
    batches = token_budget_batches([10] * 100, max_tokens=4096, max_batch=64)
    assert [len(b) for b in batches] == [64, 36]


def test_single_text_longer_than_budget_gets_own_batch():
    assert token_budget_batches([9000, 10], max_tokens=4096, max_batch=64) == [[0], [1]]


class _FakeOOM(RuntimeError):
    pass


def test_backoff_halves_batch_on_oom_and_keeps_order():
    calls = []

    def encode(texts):
        calls.append(len(texts))
        if len(texts) > 2:
            raise _FakeOOM("CUDA out of memory. Tried to allocate 384.00 MiB")
        return np.array([[float(len(t))] for t in texts], dtype=np.float32)

    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    out = encode_with_backoff(encode, texts, on_oom=lambda: None)
    assert out[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert calls == [5, 2, 3, 1, 2]  # 5 fails → 2 ok, 3 fails → 1 ok, 2 ok


def test_backoff_reraises_non_oom_and_single_text_oom():
    def boom(texts):
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        encode_with_backoff(boom, ["a", "b"], on_oom=lambda: None)

    def always_oom(texts):
        raise _FakeOOM("CUDA out of memory")

    with pytest.raises(RuntimeError, match="lower encoder.max_seq_length"):
        encode_with_backoff(always_oom, ["a"], on_oom=lambda: None)
