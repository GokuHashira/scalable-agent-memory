"""Stand-in for G's canonical fixed-plan classes g:p0..p5 (planner.md section 5).

These prove the FIXED_PLAN_IMPL="g" registry swap and the retriever interface
contract *without* reimplementing retrieval: each g:pK is a thin wrapper that
delegates to the matching reference retriever ref:pK and tags its trace with
{"impl": "g-standin"}. Interface is identical to the ref retrievers:

    .retrieve(qtext, qvec, index) -> RankedResult   (.indices, .scores, .trace)

INTEGRATION (post-midterm): replace this module with G's real plan classes.
The only registry change needed is already wired: set
``registry.FIXED_PLAN_IMPL = "g"`` and G's classes register under g:p0..p5 here;
``registry.ensure_fixed_plans()`` then aliases them onto the canonical ref:pK
names so oracle / run_plan code is unchanged. G's real classes need not delegate
to the ref retrievers -- they only have to expose the same ``.retrieve`` method.
"""
from __future__ import annotations

from mem27 import registry
from mem27.planner.ref_retrievers import RankedResult, HistoryIndex  # interface


# g:pK <- ref:pK (stand-in only; real G plans are standalone).
_DELEGATE_OF = {f"g:p{i}": f"ref:p{i}" for i in range(6)}


class GPlan:
    """Stand-in G plan: delegates retrieval to a ref retriever, re-tags the trace."""

    def __init__(self, plan_id: str, delegate):
        self.plan_id = plan_id
        self.name = f"G{plan_id.split(':')[1]}"  # G -> Gp0..Gp5
        self._delegate = delegate

    def retrieve(self, qtext: str, qvec, index: HistoryIndex) -> RankedResult:
        r = self._delegate.retrieve(qtext, qvec, index)
        trace = dict(r.trace)
        trace["impl"] = "g-standin"
        trace["delegate_plan"] = self._delegate.plan_id
        return RankedResult(r.indices, r.scores, trace)


def register_g():
    """Register g:p0..p5 (stand-ins delegating to the ref retrievers)."""
    # Importing ref_retrievers self-registers ref:p0..p5 (lazy, same as
    # registry.ensure_ref_retrievers) so the delegates exist.
    from mem27.planner import ref_retrievers  # noqa: F401  (self-registers)
    for g_id, ref_id in _DELEGATE_OF.items():
        delegate = registry.get(ref_id)  # captured before any ref:pK aliasing
        registry.register(g_id, GPlan(g_id, delegate))
    return list(_DELEGATE_OF)
