"""Shared helpers for the evaluator: note index, dates, tokens, history formatting."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from mem27.vschema import iter_jsonl, session_of, turn_of

TOKEN_CAP = 2048
MAX_RESULTS = 20


@lru_cache(maxsize=1)
def encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(encoder().encode(text, disallowed_special=()))


def truncate_tokens(text: str, n: int) -> str:
    ids = encoder().encode(text, disallowed_special=())
    return text if len(ids) <= n else encoder().decode(ids[:n])


def fmt_date(iso: str) -> str:
    """ISO -> LongMemEval display format '2023/05/20 (Sat) 02:21'."""
    try:
        return datetime.fromisoformat(iso).strftime("%Y/%m/%d (%a) %H:%M")
    except (TypeError, ValueError):
        return str(iso)


class NoteIndex:
    """note_id -> light metadata (no text unless keep_text)."""

    def __init__(self, notes_path: str | Path, keep_text: bool = False, histories: set | None = None):
        self.meta: dict[str, tuple] = {}
        self.text: dict[str, str] = {}
        self.by_history: dict[str, list[str]] = {}
        for n in iter_jsonl(notes_path):
            if histories is not None and n["history_id"] not in histories:
                continue
            nid = n["note_id"]
            self.meta[nid] = (n["history_id"], n["session_id"], n["turn_idx"], n["role"], n["ts"],
                              n.get("metadata", {}).get("date_raw"), n.get("valid_to"))
            self.by_history.setdefault(n["history_id"], []).append(nid)
            if keep_text:
                self.text[nid] = n["text"]

    def session(self, nid: str) -> str:
        m = self.meta.get(nid)
        return m[1] if m else session_of(nid)

    def turn(self, nid: str) -> int:
        m = self.meta.get(nid)
        return m[2] if m else turn_of(nid)


def days_before(ts: str, ref: str | None) -> str:
    """' (23 days before the current date)' -- offloads date arithmetic from the reader."""
    if not ref or not ts:
        return ""
    try:
        d = (datetime.fromisoformat(ref).date() - datetime.fromisoformat(ts).date()).days
    except ValueError:
        return ""
    if d == 0:
        return ", same day as the current date"
    unit = "day" if abs(d) == 1 else "days"
    return f", {d} {unit} before the current date" if d > 0 else f", {-d} {unit} after the current date"


def format_history(note_ids: list[str], passages: list[str], idx: NoteIndex | None,
                   order: str = "chrono", question_date: str | None = None, date_hints: bool = False) -> str:
    """Render a bundle for the reader.

    order="chrono": group by session, sessions sorted by date, turns in order,
                    '...' between non-adjacent turns (V's default wrapper).
    order="rank":   one block per passage in retrieval-rank order, each with its date.
    date_hints:     add "N days before the current date" to every session header.
    """
    if not note_ids:
        return "(no chat history retrieved)"
    rows = []
    for rank, (nid, text) in enumerate(zip(note_ids, passages)):
        m = idx.meta.get(nid) if idx else None
        sid = m[1] if m else session_of(nid)
        turn = m[2] if m else turn_of(nid)
        role = m[3] if m else "user"
        ts = m[4] if m else ""
        date = (m[5] if m and m[5] else fmt_date(ts)) if m else ""
        rows.append((ts, sid, turn, role, date, text, rank))
    if order == "rank":
        out = []
        for ts, sid, turn, role, date, text, rank in rows:
            body = text if role == "session" else f"{role}: {text}"
            hint = days_before(ts, question_date) if date_hints else ""
            out.append(f"### Excerpt {rank + 1} (session date: {date}{hint})\n{body}")
        return "\n\n".join(out)
    sessions: dict[str, list] = {}
    for r in rows:
        sessions.setdefault(r[1], []).append(r)
    blocks = []
    for k, (sid, rs) in enumerate(sorted(sessions.items(), key=lambda kv: (kv[1][0][0], kv[0]))):
        rs.sort(key=lambda r: r[2])
        lines, prev = [], None
        for ts, _, turn, role, date, text, _ in rs:
            if role == "session" or turn < 0:
                lines.append(text)
            else:
                if prev is not None and turn != prev + 1:
                    lines.append("...")
                lines.append(f"{role}: {text}")
            prev = turn
        hint = days_before(rs[0][0], question_date) if date_hints else ""
        blocks.append(f"### Session {k + 1} (date: {rs[0][4]}{hint})\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def load_prompt(path: str | Path) -> tuple[str, str]:
    text = Path(path).read_text()
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


def fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def collect_env(cache: str = "cold", seed: int = 0) -> dict:
    env = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "seed": seed,
        "cache": cache,
    }
    try:
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True,
                                    timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001 — no GPU on this node
        env["gpu"] = None
    libs = {}
    for lib in ("numpy", "torch", "vllm", "transformers", "sentence_transformers", "faiss", "rank_bm25", "tiktoken"):
        try:
            mod = __import__(lib)
            libs[lib] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    env["libs"] = libs
    return env


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
