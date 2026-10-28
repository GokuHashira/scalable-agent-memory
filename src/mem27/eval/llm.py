"""OpenAI-compatible chat client for the vLLM reader/judge endpoints.

- Endpoint: a URL, a file with one URL per line (runs/vllm_endpoint.txt holds
  "http://host:port/v1"), or a directory of such files (one per server job).
  Several endpoints are load-balanced by least in-flight requests, with failover.
- Greedy by default; thread-pool batching; exponential-backoff retries.
- SQLite response cache keyed by sha256(model, params, messages) so reruns
  and re-judging never recompute an identical call. Use one cache file per
  concurrently running process (eval_runs names them by experiment tag).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence


def resolve_endpoints(endpoint: str) -> list[str]:
    p = Path(endpoint)
    if p.is_dir():
        lines = [l for f in sorted(p.glob("*.txt")) for l in f.read_text().split()]
    elif p.exists():
        lines = p.read_text().split()
    else:
        lines = endpoint.split(",")
    urls = [l.strip().rstrip("/") for l in lines if l.strip()]
    if not urls:
        raise ValueError(f"no endpoints in {endpoint}")
    return urls


def resolve_endpoint(endpoint: str) -> str:
    return resolve_endpoints(endpoint)[0]


class Cache:
    """Best-effort cache. A cache failure must never fail or repeat an LLM call: on any sqlite
    error (Oct 13: two processes on one file over BeeGFS -> "disk I/O error" / "malformed"),
    the cache disables itself for this process and every call goes to the server."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.lock = threading.Lock()
        self.broken = False
        # one writer process per cache file: WAL needs shared memory, which networked
        # filesystems (BeeGFS/Lustre) do not provide reliably
        try:
            self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=120)
            self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        except sqlite3.Error as e:
            self._disable(e)

    def _disable(self, e: Exception) -> None:
        if not self.broken:
            print(f"[cache] disabled {self.path}: {e!r}", flush=True)
        self.broken = True

    def get(self, k: str):
        if self.broken:
            return None
        with self.lock:
            try:
                r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
            except sqlite3.Error as e:
                self._disable(e)
                return None
        return json.loads(r[0]) if r else None

    def put(self, k: str, v) -> None:
        if self.broken:
            return
        with self.lock:
            try:
                self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
                self.db.commit()
            except sqlite3.Error as e:
                self._disable(e)


class ChatLLM:
    def __init__(self, endpoint: str, model: str | None = None, cache: str | Path | None = None,
                 max_workers: int = 64, timeout: float = 600.0, extra_body: dict | None = None,
                 refresh_s: float = 60.0):
        from openai import OpenAI

        self._OpenAI, self._api_key, self._timeout = OpenAI, os.environ.get("VLLM_API_KEY", "EMPTY"), timeout
        self.urls = resolve_endpoints(endpoint)
        self.clients = [OpenAI(base_url=u, api_key=self._api_key, timeout=timeout, max_retries=0) for u in self.urls]
        self.inflight = [0] * len(self.clients)
        self.dead: set[int] = set()  # endpoints that refused connections or whose file vanished (server job ended)
        self.lb_lock = threading.Lock()
        # An endpoint *directory* is re-read every `refresh_s` seconds: servers that registered after start are
        # added and servers whose file disappeared are dropped (Oct 14: long runs otherwise ignored new servers
        # and kept sending to expired ones).
        self._src = endpoint if Path(endpoint).is_dir() else None
        self._refresh_s, self._last_refresh = refresh_s, time.time()
        self.base_url = self.urls[0]
        self.client = self.clients[0]
        self.model = model or self.client.models.list().data[0].id
        self.cache = Cache(cache) if cache else None
        self.max_workers = max_workers
        self.extra_body = extra_body or {}

    def _refresh(self) -> None:
        """Call with lb_lock held."""
        if self._src is None or time.time() - self._last_refresh < self._refresh_s:
            return
        self._last_refresh = time.time()
        try:
            present = resolve_endpoints(self._src)
        except (ValueError, OSError):
            return
        for u in present:
            if u not in self.urls:
                self.urls.append(u)
                self.clients.append(self._OpenAI(base_url=u, api_key=self._api_key, timeout=self._timeout, max_retries=0))
                self.inflight.append(0)
        live = set(present)
        for i, u in enumerate(self.urls):
            if u not in live:
                self.dead.add(i)

    def _key(self, messages, params) -> str:
        blob = json.dumps({"m": self.model, "p": params, "x": self.extra_body, "msg": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
             seed: int = 0, retries: int = 6) -> dict:
        params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        key = self._key(messages, params)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                hit["cached"] = True
                return hit
        delay = 2.0
        for attempt in range(retries):
            with self.lb_lock:
                self._refresh()
                alive = [i for i in range(len(self.clients)) if i not in self.dead] or list(range(len(self.clients)))
                ci = min(alive, key=lambda i: (self.inflight[i], (i - attempt) % len(self.clients)))
                self.inflight[ci] += 1
            try:
                t0 = time.perf_counter()
                r = self.clients[ci].chat.completions.create(model=self.model, messages=messages,
                                                             extra_body=self.extra_body or None, **params)
                if not hasattr(r, "choices"):
                    # not an OpenAI-compatible server (Oct 14: a node's cgroup exporter answered on a port that
                    # llama-server failed to bind); retire the endpoint so retries go elsewhere
                    with self.lb_lock:
                        self.dead.add(ci)
                    raise ValueError(f"non-OpenAI response from {self.urls[ci]}: {str(r)[:60]!r}")
                out = {
                    "text": r.choices[0].message.content or "",
                    "reasoning": getattr(r.choices[0].message, "reasoning_content", None),
                    "finish_reason": r.choices[0].finish_reason,
                    "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                    "completion_tokens": r.usage.completion_tokens if r.usage else None,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "model": self.model,
                    "endpoint": self.urls[ci],
                }
            except Exception as e:  # noqa: BLE001 — network / server overload: back off and retry
                with self.lb_lock:
                    self.inflight[ci] -= 1
                    if type(e).__name__ in ("APIConnectionError", "APITimeoutError") and "refused" in repr(e).lower() \
                            or "Connection refused" in repr(getattr(e, "__cause__", "")):
                        self.dead.add(ci)
                if attempt == retries - 1:
                    return {"text": "", "error": repr(e), "model": self.model, "cached": False}
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                with self.lb_lock:
                    self.inflight[ci] -= 1
                if self.cache is not None:
                    self.cache.put(key, out)  # never raises; outside the try so it cannot trigger a retry
                out["cached"] = False
                return out
        raise AssertionError

    def map(self, batch: Sequence[list[dict]], progress: Callable[[int, int], None] | None = None,
            **kw) -> list[dict]:
        out: list[dict | None] = [None] * len(batch)
        done = 0
        with ThreadPoolExecutor(self.max_workers) as ex:
            futs = {ex.submit(self.chat, m, **kw): i for i, m in enumerate(batch)}
            for f in futs:
                out[futs[f]] = f.result()
                done += 1
                if progress and (done % 100 == 0 or done == len(batch)):
                    progress(done, len(batch))
        return out  # type: ignore[return-value]
