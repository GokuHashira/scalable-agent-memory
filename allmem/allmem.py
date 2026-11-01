"""All-Mem (from-paper reimplementation) — NOT the authors' code.

Clean-room reimplementation of the core mechanism of

    All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution
    Can Lv, Heng Chang, Yuchen Guo, Shengyu Tao, Shiji Zhou. arXiv:2603.19595 (2026).

The authors' implementation is not public. This module reimplements, from the paper
text (Sec. 3 + Appendices B/C/D/F), the parts that matter for a LongMemEval QA baseline:

  * A topology-structured memory bank M = (V, E): each unit v = <c, s, K, z, t, a>
    where c = immutable raw evidence (source turn note_ids), s = regenerable summary,
    K = regenerable keywords, z = embedding of (s + K), t = timestamp,
    a in {0,1} = visibility indicator (the "visible surface" V+ = {v : a=1}).
  * Non-destructive consolidation via Agentic Topology Consolidation (ATC, Alg. 1):
    an LLM diagnoser emits confidence-scored split/merge/update proposals over a
    bounded top-k anchor neighborhood; proposals are gated (theta_op = 0.9),
    normalized (merge coherence gamma=0.7, arity 2..4, visibility), then executed
    serially in the fixed order SPLIT -> MERGE -> UPDATE with an applicability
    (touched-set) check. Operators are non-destructive: sources are *archived*
    (a<-0), never deleted, and version links (new -> old) preserve traceability.
  * Topology-aware retrieval (Sec. 3.4): Stage-1 top-k anchors over the visible
    surface; Stage-2 hop-bounded, budgeted typed-link expansion to archived evidence
    (<= L candidates); Stage-3 cosine re-rank, materialize top-K units' evidence.

Faithfulness caveats (documented honestly, see docs/midterm/allmem_reimpl_note.md):
  * The paper's *online* writing regenerates each unit descriptor with one LLM call
    per turn. To keep LLM calls bounded on ~490-turn histories (per the task brief),
    this reimpl does LLM-free online writing (unit summary = raw turn text) and runs
    the LLM diagnoser once per small *batch* of turns rather than once per turn.
    This is the one deliberate efficiency deviation; everything downstream (gating,
    normalization, non-destructive operators, typed-link retrieval) follows the paper.
  * Embeddings: the paper uses all-MiniLM-L6-v2; we use BAAI/bge-base-en-v1.5
    (the slice-wide embedder, cached per-turn) so retrieval matches the other plans.
  * Reader/judge is a local 7B (Qwen2.5-7B-Instruct via vLLM), not GPT-4o.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Config (faithful defaults from the paper where given)
# --------------------------------------------------------------------------- #
@dataclass
class AllMemConfig:
    # ATC gating (Appendix C.2/C.4): single high-confidence gate for all operators.
    theta: float = 0.9
    gamma_merge: float = 0.7          # merge semantic-coherence pre-check (Eq. 29)
    merge_min: int = 2                # merge arity bounds (diagnosis schema)
    merge_max: int = 4
    # Consolidation batching (our bounded-cost adaptation of online writing).
    batch_size: int = 12              # turns per ATC diagnosis call
    anchor_k_consolidate: int = 10    # existing visible nodes shown to the diagnoser
    max_ops_per_history: int = 80     # safety cap on executed operators
    # Topology-aware retrieval budgets (Sec. 3.4 / 4.3 best setting: K=16, k=10, L=40).
    anchor_k: int = 10                # Stage-1 visible anchors
    expand_L: int = 40                # Stage-2 candidate budget
    expand_H: int = 4                 # hop limit (H_0.95 = 4 in the paper)
    final_K: int = 16                 # Stage-3 materialized units
    # Reader evidence budget.
    max_evidence_turns: int = 24
    max_context_tokens: int = 2048
    # LLM sampling.
    diag_max_tokens: int = 512
    plan_max_tokens: int = 512
    reader_max_tokens: int = 96
    judge_max_tokens: int = 4
    model: str = "Qwen/Qwen2.5-7B-Instruct"


# --------------------------------------------------------------------------- #
# Memory unit + bank
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    uid: str
    summary: str
    keywords: list[str]
    evidence: list[str]               # immutable source note_ids (c_i)
    z: np.ndarray                     # normalized embedding (z_i)
    ts: str | None = None
    visible: int = 1                  # a_i
    # typed links
    ver_links: list[str] = field(default_factory=list)   # E_nu : new -> old (archived)
    sib_links: list[str] = field(default_factory=list)   # E_beta: sibling coherence
    sem_links: list[str] = field(default_factory=list)   # E_sigma: semantic anchors
    prev: str | None = None                              # E_tau : temporal predecessor


class MemoryBank:
    def __init__(self):
        self.units: dict[str, Unit] = {}
        self._next = 0

    def new_uid(self) -> str:
        self._next += 1
        return f"n{self._next}"

    def add(self, u: Unit):
        self.units[u.uid] = u

    def visible_units(self) -> list[Unit]:
        return [u for u in self.units.values() if u.visible == 1]

    def n_visible(self) -> int:
        return sum(1 for u in self.units.values() if u.visible == 1)


# --------------------------------------------------------------------------- #
# Prompts (Appendix C.1 diagnosis prompt + B.6 operator plan prompts, paraphrased)
# --------------------------------------------------------------------------- #
DIAG_SYS = (
    "You are a Senior Memory Archivist maintaining the integrity of a knowledge graph of "
    "memory nodes about one user. You do NOT edit the graph; you only propose topology edits "
    "as strict JSON. Be conservative and precision-oriented."
)
DIAG_USER_TMPL = """DATA

[Retrieved Context Nodes (existing memory)]:
{context_text}

[Target Input (new nodes just written)]:
{target_info}

STRICT DIAGNOSIS RULES
1) SPLIT (mitosis): propose SPLIT for a node ONLY if it contains multi-domain noise
   (several unrelated topics). DO NOT split chronological stories or break who-said-what.
   If the node is a single coherent statement, confidence MUST be 0.0.
2) MERGE (fusion): propose MERGE for 2-4 node_ids ONLY if they are factually redundant
   (same entity, same fact). DO NOT merge conflicting states from different times.
3) UPDATE (conflict resolution): propose UPDATE only if a new target node explicitly
   CONTRADICTS or SUPERSEDES an existing node (old_node_id=existing, new_node_id=target).
   Must be an explicit conflict, not a mere refinement.

Assign a calibrated confidence in [0,1] to each task; only truly clear cases should exceed 0.9.

Output ONLY this JSON object, no prose:
{{"split_tasks":[{{"node_id":"<id>","reason":"<short>","confidence":<float>}}],
 "merge_tasks":[{{"node_ids":["<id>","<id>"],"reason":"<short>","confidence":<float>}}],
 "update_tasks":[{{"old_node_id":"<id>","new_node_id":"<id>","reason":"<short>","confidence":<float>}}]}}
Use empty lists when nothing qualifies."""

MERGE_SYS = "You consolidate redundant memory notes into one canonical statement."
MERGE_USER_TMPL = """These memory notes are redundant (same fact about the user). Write ONE canonical
statement that captures their shared fact, then up to 6 keywords.

{members}

Output ONLY JSON: {{"summary":"<one sentence>","keywords":["..."]}}"""

UPDATE_SYS = "You refresh a memory note so it reflects the most current information."
UPDATE_USER_TMPL = """A newer note supersedes/contradicts an older one. Rewrite the CURRENT note so it
states the up-to-date fact, then up to 6 keywords. Do not mention the old value unless needed.

[CURRENT note]: {cur}
[OLD superseded note]: {old}

Output ONLY JSON: {{"summary":"<one sentence>","keywords":["..."]}}"""

SPLIT_SYS = "You separate a memory note that mixes unrelated topics into coherent sub-notes."
SPLIT_USER_TMPL = """This memory note mixes multiple unrelated topics. Group its content into 2 or more
coherent sub-notes (one topic each). For each sub-note give a summary and up to 5 keywords.

[NODE]: {node}

Output ONLY JSON: {{"subnotes":[{{"summary":"<...>","keywords":["..."]}}]}}"""

READER_SYS = (
    "You are answering questions about a user based ONLY on excerpts from their past "
    "conversations with an assistant. Use only the excerpts. If the answer is not present, "
    "reply exactly 'I don't know'. Answer in as few words as possible — just the fact."
)
JUDGE_SYS = (
    "You grade whether a predicted answer matches the gold answer for a question. "
    "Reply with a single word: CORRECT if the prediction conveys the same key fact as the "
    "gold answer (minor wording/format differences are fine), otherwise WRONG."
)


def reader_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n".join(f"- {p}" for p in passages)
    return f"Conversation excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def judge_prompt(question: str, gold: str, pred: str) -> str:
    return (f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}\n\n"
            f"Is the predicted answer correct? Reply CORRECT or WRONG.")


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible vLLM endpoint) with retry + robust JSON parse
# --------------------------------------------------------------------------- #
class LLM:
    def __init__(self, base_url="http://localhost:8001/v1", api_key="sk-local",
                 model="Qwen/Qwen2.5-7B-Instruct", timeout=120.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self._n = 0
        self._lock = threading.Lock()

    @property
    def n_calls(self) -> int:
        return self._n

    def chat(self, system: str, user: str, max_tokens: int, temperature: float = 0.0,
             retries: int = 4) -> str:
        with self._lock:
            self._n += 1
        last = None
        for a in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return (r.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(2.0 * (a + 1), 8.0))
        raise RuntimeError(f"LLM call failed after {retries} tries: {last}")


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict:
    """Best-effort: strip code fences, grab the first balanced {...}, json.loads."""
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = _JSON_RE.search(t)
    if m:
        frag = m.group(0)
        for cand in (frag, frag + "}", frag + "]}"):
            try:
                return json.loads(cand)
            except Exception:
                continue
    return {}


# --------------------------------------------------------------------------- #
# Embedding helpers
# --------------------------------------------------------------------------- #
_SBERT = None
_SBERT_LOCK = threading.Lock()
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_sbert(device: str | None = None):
    global _SBERT
    if _SBERT is None:
        with _SBERT_LOCK:
            if _SBERT is None:
                from sentence_transformers import SentenceTransformer
                _SBERT = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
                _SBERT.max_seq_length = 512
    return _SBERT


def embed_texts(texts: list[str], query: bool = False) -> np.ndarray:
    m = get_sbert()
    if query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
    with _SBERT_LOCK:
        v = m.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False)
    return v.astype(np.float32)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


_WORD = re.compile(r"[A-Za-z0-9']+")
_STOP = set("the a an and or of to in on at for with is are was were be been being i you he she "
            "it we they my your his her its our their me him them this that these those as by "
            "from have has had do does did not no so if then than too very can will would should "
            "could about into over under up down out".split())


def keywords_of(text: str, n: int = 6) -> list[str]:
    seen, out = set(), []
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 3 or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Build the per-history memory bank (online writing + batched ATC)
# --------------------------------------------------------------------------- #
def _render_node(u: Unit, max_chars: int = 220) -> str:
    s = u.summary.replace("\n", " ").strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return f"[{u.uid}] {s}"


def build_bank(notes: list[dict], turn_embs: np.ndarray, llm: LLM, cfg: AllMemConfig,
               stats: dict) -> MemoryBank:
    """notes: history turns ordered by (session, turn_idx); turn_embs: aligned bge rows."""
    bank = MemoryBank()
    n_ops = 0

    def similar_visible(z: np.ndarray, k: int, exclude: set[str]) -> list[str]:
        vis = [u for u in bank.visible_units() if u.uid not in exclude]
        if not vis:
            return []
        M = np.stack([u.z for u in vis])
        sims = M @ z
        idx = np.argsort(-sims)[:k]
        return [vis[i].uid for i in idx]

    def regen_embed(summary: str, kws: list[str]) -> np.ndarray:
        txt = summary + (" " + " ".join(kws) if kws else "")
        return _norm(embed_texts([txt])[0])

    prev_uid = None
    N = len(notes)
    for start in range(0, N, cfg.batch_size):
        batch = notes[start:start + cfg.batch_size]
        new_uids = []
        # ---- online writing: one provisional unit per turn (LLM-free) ----
        for j, note in enumerate(batch):
            gi = start + j
            z = turn_embs[gi].astype(np.float32)
            z = z / max(float(np.linalg.norm(z)), 1e-9)
            uid = bank.new_uid()
            role = note.get("role", "")
            text = note.get("text", "")
            summary = f"{role}: {text}" if role else text
            u = Unit(uid=uid, summary=summary, keywords=keywords_of(text),
                     evidence=[note["note_id"]], z=z, ts=note.get("ts"),
                     visible=1, prev=prev_uid)
            if prev_uid:
                pass  # temporal link recorded via .prev
            # sparse semantic attachments to current visible surface
            u.sem_links = similar_visible(z, 3, exclude={uid})
            bank.add(u)
            new_uids.append(uid)
            prev_uid = uid

        # ---- ATC diagnosis over this batch + top-k existing anchors ----
        centroid = _norm(np.mean([bank.units[x].z for x in new_uids], axis=0))
        ctx_ids = similar_visible(centroid, cfg.anchor_k_consolidate, exclude=set(new_uids))
        context_text = "\n".join(_render_node(bank.units[x]) for x in ctx_ids) or "(none yet)"
        target_info = "\n".join(_render_node(bank.units[x]) for x in new_uids)
        raw = llm.chat(DIAG_SYS,
                       DIAG_USER_TMPL.format(context_text=context_text, target_info=target_info),
                       max_tokens=cfg.diag_max_tokens)
        prop = parse_json(raw)

        # ---- confidence gating + normalization (Appendix C.2) ----
        # pre-gate proposal counts (diagnoser output before theta/coherence filtering)
        stats["proposed_split_raw"] = stats.get("proposed_split_raw", 0) + len(prop.get("split_tasks") or [])
        stats["proposed_merge_raw"] = stats.get("proposed_merge_raw", 0) + len(prop.get("merge_tasks") or [])
        stats["proposed_update_raw"] = stats.get("proposed_update_raw", 0) + len(prop.get("update_tasks") or [])
        split_q, merge_q, update_q = [], [], []
        for t in (prop.get("split_tasks") or []):
            try:
                c = float(t.get("confidence", 0))
            except Exception:
                c = 0.0
            nid = str(t.get("node_id", ""))
            # granular rejection accounting (why proposed splits don't execute)
            if nid not in bank.units:
                stats["split_rej_badid"] = stats.get("split_rej_badid", 0) + 1
            elif bank.units[nid].visible != 1:
                stats["split_rej_invisible"] = stats.get("split_rej_invisible", 0) + 1
            elif c < cfg.theta:
                stats["split_rej_lowconf"] = stats.get("split_rej_lowconf", 0) + 1
            if c >= cfg.theta and nid in bank.units and bank.units[nid].visible == 1:
                split_q.append(nid)
        for t in (prop.get("merge_tasks") or []):
            try:
                c = float(t.get("confidence", 0))
            except Exception:
                c = 0.0
            ids = [str(x) for x in (t.get("node_ids") or [])]
            ids = sorted({i for i in ids if i in bank.units and bank.units[i].visible == 1})
            if c >= cfg.theta and cfg.merge_min <= len(ids) <= cfg.merge_max:
                # semantic-coherence pre-check (Eq. 29)
                Z = np.stack([bank.units[i].z for i in ids])
                S = Z @ Z.T
                m = len(ids)
                coh = (S.sum() - m) / (m * (m - 1))
                if coh >= cfg.gamma_merge:
                    merge_q.append(ids)
        for t in (prop.get("update_tasks") or []):
            try:
                c = float(t.get("confidence", 0))
            except Exception:
                c = 0.0
            old = str(t.get("old_node_id", ""))
            new = str(t.get("new_node_id", ""))
            if (c >= cfg.theta and old in bank.units and new in bank.units
                    and old != new and bank.units[new].visible == 1):
                update_q.append((new, old))

        stats["proposed"] += len(split_q) + len(merge_q) + len(update_q)

        # ---- serial editing: SPLIT -> MERGE -> UPDATE with applicability ----
        touched: set[str] = set()

        for nid in split_q:
            if n_ops >= cfg.max_ops_per_history:
                break
            u = bank.units.get(nid)
            if not u or u.visible == 0 or nid in touched or len(u.evidence) < 2:
                continue  # singletons cannot be split (no multi-evidence to regroup)
            raw = llm.chat(SPLIT_SYS, SPLIT_USER_TMPL.format(node=_render_node(u, 600)),
                           max_tokens=cfg.plan_max_tokens)
            subs = (parse_json(raw) or {}).get("subnotes") or []
            subs = [s for s in subs if isinstance(s, dict) and s.get("summary")]
            if len(subs) < 2:
                continue
            # partition evidence round-robin across siblings (deterministic)
            sib_uids = []
            for si, s in enumerate(subs):
                ev = [e for ei, e in enumerate(u.evidence) if ei % len(subs) == si] or [u.evidence[si % len(u.evidence)]]
                kws = [str(k) for k in (s.get("keywords") or [])][:5]
                suid = bank.new_uid()
                nu = Unit(uid=suid, summary=str(s["summary"]), keywords=kws, evidence=ev,
                          z=regen_embed(str(s["summary"]), kws), ts=u.ts, visible=1,
                          ver_links=[nid])  # sibling -> archived original
                bank.add(nu)
                sib_uids.append(suid)
            for suid in sib_uids:
                bank.units[suid].sib_links = [x for x in sib_uids if x != suid]
            u.visible = 0  # archive original (non-destructive)
            touched.add(nid)
            touched.update(sib_uids)
            n_ops += 1
            stats["split"] += 1

        for ids in merge_q:
            if n_ops >= cfg.max_ops_per_history:
                break
            ids = [i for i in ids if bank.units.get(i) and bank.units[i].visible == 1 and i not in touched]
            if len(ids) < cfg.merge_min:
                continue
            members = "\n".join(_render_node(bank.units[i], 400) for i in ids)
            raw = llm.chat(MERGE_SYS, MERGE_USER_TMPL.format(members=members),
                           max_tokens=cfg.plan_max_tokens)
            o = parse_json(raw)
            summ = str(o.get("summary") or "").strip()
            if not summ:
                continue
            kws = [str(k) for k in (o.get("keywords") or [])][:6]
            ev = []
            for i in ids:
                ev.extend(bank.units[i].evidence)
            nuid = bank.new_uid()
            nu = Unit(uid=nuid, summary=summ, keywords=kws, evidence=list(dict.fromkeys(ev)),
                      z=regen_embed(summ, kws), ts=bank.units[ids[-1]].ts, visible=1,
                      ver_links=list(ids))  # representative -> archived sources
            bank.add(nu)
            for i in ids:
                bank.units[i].visible = 0
                touched.add(i)
            touched.add(nuid)
            n_ops += 1
            stats["merge"] += 1

        for new, old in update_q:
            if n_ops >= cfg.max_ops_per_history:
                break
            nu, ou = bank.units.get(new), bank.units.get(old)
            if not nu or not ou or nu.visible == 0 or new in touched or old in touched:
                continue
            raw = llm.chat(UPDATE_SYS,
                           UPDATE_USER_TMPL.format(cur=_render_node(nu, 400), old=_render_node(ou, 400)),
                           max_tokens=cfg.plan_max_tokens)
            o = parse_json(raw)
            summ = str(o.get("summary") or "").strip()
            if not summ:
                continue
            kws = [str(k) for k in (o.get("keywords") or [])][:6] or nu.keywords
            nu.summary = summ
            nu.keywords = kws
            nu.z = regen_embed(summ, kws)
            ou.visible = 0                 # archive superseded
            if old not in nu.ver_links:
                nu.ver_links.append(old)   # current -> old
            touched.add(old)
            touched.add(new)
            n_ops += 1
            stats["update"] += 1

    stats["n_units_total"] = len(bank.units)
    stats["n_units_visible"] = bank.n_visible()
    stats["n_ops"] = n_ops
    return bank


# --------------------------------------------------------------------------- #
# Topology-aware retrieval (Sec. 3.4)
# --------------------------------------------------------------------------- #
def retrieve(bank: MemoryBank, qz: np.ndarray, cfg: AllMemConfig) -> list[Unit]:
    vis = bank.visible_units()
    if not vis:
        # fall back to all units if nothing visible (shouldn't happen)
        vis = list(bank.units.values())
    M = np.stack([u.z for u in vis])
    sims = M @ qz
    order = np.argsort(-sims)[:cfg.anchor_k]
    anchors = [vis[i] for i in order]

    # Stage-2: hop-bounded typed-link expansion (favor version/sibling links).
    cand: dict[str, Unit] = {u.uid: u for u in anchors}
    frontier = list(anchors)
    for _hop in range(cfg.expand_H):
        if len(cand) >= cfg.expand_L:
            break
        nxt = []
        for u in frontier:
            for lid in (u.ver_links + u.sib_links):
                if lid in bank.units and lid not in cand:
                    cand[lid] = bank.units[lid]
                    nxt.append(bank.units[lid])
                    if len(cand) >= cfg.expand_L:
                        break
            if len(cand) >= cfg.expand_L:
                break
        frontier = nxt
        if not frontier:
            break

    # Stage-3: cosine re-rank, materialize top-K.
    cu = list(cand.values())
    Z = np.stack([u.z for u in cu])
    cs = Z @ qz
    top = np.argsort(-cs)[:cfg.final_K]
    return [cu[i] for i in top]


def materialize_evidence(units: list[Unit], note_text: dict[str, str],
                         cfg: AllMemConfig) -> tuple[list[str], list[str]]:
    """Return (passages, evidence_note_ids) under the evidence-turn/token budget."""
    passages, ev_ids, seen = [], [], set()
    from mem27.planner.refcommon import count_tokens
    tok = 0
    for u in units:
        for nid in u.evidence:
            if nid in seen:
                continue
            seen.add(nid)
            txt = note_text.get(nid)
            if not txt:
                continue
            c = count_tokens(txt)
            if tok + c > cfg.max_context_tokens or len(passages) >= cfg.max_evidence_turns:
                return passages, ev_ids
            passages.append(txt)
            ev_ids.append(nid)
            tok += c
    return passages, ev_ids


__all__ = ["AllMemConfig", "Unit", "MemoryBank", "LLM", "build_bank", "retrieve",
           "materialize_evidence", "reader_prompt", "judge_prompt", "parse_json",
           "embed_texts", "get_sbert", "READER_SYS", "JUDGE_SYS"]
