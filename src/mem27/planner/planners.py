"""Planners a0 (always), a1 (rules), a2 (cascade). planner.md section 2/3.5.

A planner decides which fixed plan(s) to run for a question and returns a
PlanDecision that run_plan turns into a Bundle. a2 (cascade) may run several rungs;
its token cost is the cumulative cost of every rung it executed (cost-at-parity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mem27 import registry
from mem27.planner.ref_retrievers import RankedResult, HistoryIndex


@dataclass
class PlanDecision:
    final: RankedResult                 # ranked notes used for recall + final bundle
    cost_results: list                  # RankedResults whose bundles sum to the token cost
    chosen_plan: str                    # fixed plan id selected / final rung
    trace: dict = field(default_factory=dict)


# ---- a0: always one fixed plan -------------------------------------------
class A0Always:
    plan_id = "a0"

    def __init__(self, default_plan: str = "ref:p2"):
        self.default_plan = default_plan

    def decide(self, qtext, qvec, index) -> PlanDecision:
        r = registry.get(self.default_plan).retrieve(qtext, qvec, index)
        return PlanDecision(r, [r], self.default_plan,
                            {"planner": "a0", "always": self.default_plan})


# ---- a1: rules by question type ------------------------------------------
class A1Rules:
    plan_id = "a1"
    RULES = {
        "single-session-user": "ref:p2",
        "single-session-assistant": "ref:p2",
        "single-session-preference": "ref:p3",
        "temporal-reasoning": "ref:p4",
        "knowledge-update": "ref:p5",
        "multi-session": "ref:p2",
    }
    FALLBACK = "ref:p2"

    def decide(self, qtext, qvec, index, question_type=None) -> PlanDecision:
        plan = self.RULES.get(question_type, self.FALLBACK)
        r = registry.get(plan).retrieve(qtext, qvec, index)
        return PlanDecision(r, [r], plan,
                            {"planner": "a1", "question_type": question_type, "rule_plan": plan})


# ---- a2: cost cascade -----------------------------------------------------
def _cosine_margin(result: RankedResult, qvec: np.ndarray, index: HistoryIndex) -> float:
    """Confidence = cosine(query, top1) - cosine(query, top2). High => clear winner."""
    if index.emb is None or len(result.indices) == 0:
        return 0.0
    top1 = index.emb[result.indices[0]] @ qvec
    if len(result.indices) == 1:
        return float(top1)  # only one candidate -> treat its absolute sim as margin
    top2 = index.emb[result.indices[1]] @ qvec
    return float(top1 - top2)


class A2Cascade:
    plan_id = "a2"
    # dense is the best *cheap* rung (1ms, strong); escalate to hybrid then full traversal.
    RUNGS = ["ref:p1", "ref:p2", "ref:p5"]

    def __init__(self, tau_margin: float = 0.1):
        self.tau_margin = tau_margin

    def decide(self, qtext, qvec, index) -> PlanDecision:
        cost_results = []
        trace = {"planner": "a2", "tau_margin": self.tau_margin, "rungs": []}
        final = None
        chosen = self.RUNGS[-1]
        stop_reason = "exhausted"
        for i, rung in enumerate(self.RUNGS):
            r = registry.get(rung).retrieve(qtext, qvec, index)
            cost_results.append(r)
            conf = _cosine_margin(r, qvec, index)
            is_last = (i == len(self.RUNGS) - 1)
            trace["rungs"].append({"plan": rung, "confidence": round(conf, 4),
                                   "n": len(r.indices)})
            final = r
            chosen = rung
            if conf >= self.tau_margin:
                stop_reason = "confident"
                break
            if is_last:
                stop_reason = "exhausted"
                break
        trace["stop_reason"] = stop_reason
        trace["final_plan"] = chosen
        trace["rungs_run"] = len(cost_results)
        return PlanDecision(final, cost_results, chosen, trace)


# registries for planner discovery (run_plan constructs with params)
PLANNER_FACTORIES = {
    "a0": A0Always,
    "a1": A1Rules,
    "a2": A2Cascade,
}
