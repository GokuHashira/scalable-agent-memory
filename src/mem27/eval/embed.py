"""bge-base-en-v1.5 embedding cache keyed by md5(text) (shared default model, §0)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class EmbCache:
    def __init__(self, root: str | Path, model_name: str = MODEL, device: str | None = None):
        self.root = Path(root)
        self.model_name = model_name
        self.device = device
        self._model = None
        self.keys: dict[str, int] = {}
        self.mat = np.zeros((0, 768), np.float16)
        if (self.root / "keys.json").exists():
            self.keys = {k: i for i, k in enumerate(json.loads((self.root / "keys.json").read_text()))}
            self.mat = np.load(self.root / "emb.npy")

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = 512
            if self._model.device.type == "cuda":
                self._model.half()
        return self._model

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float16)

    def add(self, texts: list[str], batch_size: int = 256) -> dict:
        todo = {}
        for t in texts:
            k = key(t)
            if k not in self.keys and k not in todo:
                todo[k] = t
        t0 = time.time()
        if todo:
            items = sorted(todo.items(), key=lambda kv: len(kv[1]))  # length-sorted batches
            embs = self.encode([t for _, t in items], batch_size)
            base = len(self.keys)
            for j, (k, _) in enumerate(items):
                self.keys[k] = base + j
            self.mat = np.concatenate([self.mat, embs]) if len(self.mat) else embs
            self.save()
        return {"new": len(todo), "seconds": time.time() - t0}

    def get(self, texts: list[str]) -> np.ndarray:
        return self.mat[[self.keys[key(t)] for t in texts]]

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.root / "emb.npy", self.mat)
        (self.root / "keys.json").write_text(json.dumps(sorted(self.keys, key=self.keys.get)))
