"""Plan registry.

Maps plan names -> retriever factory. The `ref:p0..p5` plans are Slice K's own
reference retrievers (stand-ins for G's canonical fixed plans). After integration,
flip FIXED_PLAN_IMPL to "g" and register G's classes under the same p0..p5 names so
oracle/planner code is unchanged (planner.md section 5).

The 6-rung cost ladder (planner.md section 3.2 names five retrievers but numbers
p0..p5; p5 is the hybrid+traversal top rung — the natural ceiling of the a2 cascade):
  p0  RefBM25           lexical, cheapest
  p1  RefDense          bge-base + FAISS flat
  p2  RefHybrid         RRF(k=60) of BM25 + Dense
  p3  RefSummaryFirst   extractive 3-sentence session summaries, then expand
  p4  RefTraversal      link-graph expansion, anchors from hybrid top-5
  p5  RefHybridTrav     hybrid anchors + traversal expansion (fullest/most expensive)
"""
from __future__ import annotations

import os

# "ref" = Slice K's reference retrievers (default). "g" = G's classes (alias stand-in).
# "gi"  = Slice K's genuinely-INDEPENDENT family (TF-IDF/dense+PRF/linear-hybrid/centroid/
#         PageRank/MMR) mapped onto the canonical ref:pK names — the faithful end-to-end
#         rehearsal of the planner §5 swap when G's real classes are unavailable.
# Overridable per-run with env MEM27_FIXED_PLAN_IMPL so run_plan/oracle need no code edit.
FIXED_PLAN_IMPL = os.environ.get("MEM27_FIXED_PLAN_IMPL", "ref")  # "ref" | "g" | "gi"

# Canonical plan ids, cheapest -> most expensive.
FIXED_PLANS = ["ref:p0", "ref:p1", "ref:p2", "ref:p3", "ref:p4", "ref:p5"]
PLAN_SHORT = {p: p.split(":")[1] for p in FIXED_PLANS}  # ref:p0 -> p0

# Planner plan ids
PLANNER_PLANS = ["a0", "a1", "a2"]

_REGISTRY: dict[str, object] = {}


def register(name: str, factory):
    _REGISTRY[name] = factory


def get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"plan {name!r} not registered; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def ensure_ref_retrievers():
    """Lazy import so schema-only consumers don't pull torch/faiss."""
    if FIXED_PLAN_IMPL == "ref" and "ref:p0" not in _REGISTRY:
        from mem27.planner import ref_retrievers  # noqa: F401  (self-registers)
    return FIXED_PLANS


def ensure_g_retrievers():
    """Register G's canonical plans under g:p0..p5 (lazy import). Returns [g:p0..p5]."""
    from mem27.planner import g_plans
    return g_plans.register_g()


def ensure_gi_retrievers():
    """Register the genuinely-independent gi:p0..p5 family (lazy). Returns [gi:p0..p5]."""
    from mem27.planner import g_plans_indep
    return g_plans_indep.register_gi()


def ensure_fixed_plans():
    """Register the active fixed-plan implementation under the canonical ref:pK names.

    Dispatches on FIXED_PLAN_IMPL (planner.md section 5):
      "ref" -> Slice K's reference retrievers self-register as ref:p0..p5.
      "g"   -> G's plans register as g:p0..p5, and the SAME instances are aliased
               onto ref:p0..p5 so oracle / run_plan (which iterate FIXED_PLANS)
               are unchanged.
    Returns FIXED_PLANS. Backward compatible: callers using ensure_ref_retrievers()
    and the ref:pK names keep working in the default "ref" mode.
    """
    if FIXED_PLAN_IMPL == "ref":
        return ensure_ref_retrievers()
    if FIXED_PLAN_IMPL == "g":
        ensure_g_retrievers()
        for p in FIXED_PLANS:                      # ref:pK -> g:pK alias
            _REGISTRY[p] = _REGISTRY["g:" + p.split(":")[1]]
        return FIXED_PLANS
    if FIXED_PLAN_IMPL == "gi":
        ensure_gi_retrievers()
        for p in FIXED_PLANS:                      # ref:pK -> gi:pK (independent impl)
            _REGISTRY[p] = _REGISTRY["gi:" + p.split(":")[1]]
        return FIXED_PLANS
    raise ValueError(f"unknown FIXED_PLAN_IMPL {FIXED_PLAN_IMPL!r}")
