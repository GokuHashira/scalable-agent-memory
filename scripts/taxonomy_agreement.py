"""Agreement between hand labels of wrong answers and the rule taxonomy (scripts/taxonomy.py "labels").

Hand labels (jsonl: run, qid, label, note) use the same mode names with '/' sub-labels; both sides are mapped to
families (session retrieval / turn retrieval / version / reading / abstention). Labels starting with CHECK- are
reported separately (the hand reading could not decide them without the passage text).

  python scripts/taxonomy_agreement.py --manual runs/_tables/taxonomy_manual_labels_test.jsonl \
      --tax runs/_tables/taxonomy_lme_s_test.json --out runs/_tables/taxonomy_agreement_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import write_json  # noqa: E402
from mem27.vschema import iter_jsonl  # noqa: E402

FAMILY = {
    "retrieval-miss": "session-retrieval", "under-retrieval": "session-retrieval", "lexical-miss": "session-retrieval",
    "semantic-drift": "session-retrieval", "partial-retrieval": "session-retrieval",
    "turn-miss": "turn-retrieval", "turn-partial": "turn-retrieval", "truncation": "turn-retrieval",
    "wrong-version": "version",
    "false-abstention": "reading", "over-retrieval": "reading", "reader-error": "reading",
    "hallucinated-abstention": "abstention",
}


def family(label: str) -> str:
    base = label.split("/")[0].split(":")[0]
    return FAMILY.get(base, "other")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--tax", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rule = json.loads(Path(a.tax).read_text())["labels"]
    rows, fam_ok, exact_ok, n, checks = [], 0, 0, 0, []
    conf = Counter()
    for m in iter_jsonl(a.manual):
        key = f"{m['run']}|{m['qid']}"
        r = rule.get(key)
        if r is None:
            rows.append({**m, "rule": None, "note_rule": "not wrong under the final official label"})
            continue
        if m["label"].startswith("CHECK-"):
            checks.append({**m, "rule": r})
            continue
        n += 1
        fm, fr = family(m["label"]), family(r)
        fam_ok += fm == fr
        exact_ok += m["label"].split("/")[0] == r.split(":")[0]  # base mode (sub-labels ignored)
        conf[(fm, fr)] += 1
        rows.append({**m, "rule": r, "family_manual": fm, "family_rule": fr})
    out = {"n_compared": n, "family_agreement": fam_ok / max(1, n), "mode_agreement": exact_ok / max(1, n),
           "confusion_family": {f"{x}->{y}": c for (x, y), c in sorted(conf.items())}, "check_items": checks,
           "rows": rows}
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "check_items")}, indent=1))
    for c in checks:
        print("CHECK", c["run"], c["qid"], "->", c["rule"], "|", c["note"][:90])
    for r in rows:
        if r.get("rule") and r.get("family_manual") != r.get("family_rule"):
            print("DIFF", r["run"], r["qid"], r["label"], "->", r["rule"], "|", r["note"][:80])


if __name__ == "__main__":
    main()
