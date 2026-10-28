"""Markdown tables from scripts/taxonomy.py output: failure modes per run (count and share of wrong answers),
grouped into families, plus a mode x question-type table pooled over runs.

  python scripts/taxonomy_table.py --tax runs/_tables/taxonomy_lme_s_test.json --out runs/_tables/taxonomy_lme_s_test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from scripts.reproduce_table import SHORT, TYPES, to_md  # noqa: E402

FAMILIES = [
    ("retrieval: session level", ["retrieval-miss", "under-retrieval", "lexical-miss", "semantic-drift", "partial-retrieval"]),
    ("retrieval: turn level", ["turn-miss", "turn-partial", "truncation"]),
    ("version", ["wrong-version"]),
    ("reading (all answer turns present)", ["false-abstention", "over-retrieval", "reader-error:temporal",
                                            "reader-error:aggregation", "reader-error:extraction", "reader-error:preference"]),
    ("abstention questions", ["hallucinated-abstention"]),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tax", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = json.loads(Path(a.tax).read_text())
    runs = list(t["counts"])
    known = {m for _, ms in FAMILIES for m in ms}
    extra = sorted({m for r in runs for m in t["counts"][r]} - known)
    fams = FAMILIES + ([("other", extra)] if extra else [])
    rows = [["family", "mode"] + [f"{r} n (%)" for r in runs]]
    tot = {r: sum(t["counts"][r].values()) for r in runs}
    for fam, modes in fams:
        for m in modes:
            cells = [f"{t['counts'][r].get(m, 0)} ({100 * t['counts'][r].get(m, 0) / max(1, tot[r]):.0f}%)" for r in runs]
            rows.append([fam, m] + cells)
        fsum = [sum(t["counts"][r].get(m, 0) for m in modes) for r in runs]
        rows.append([f"**{fam}**", "**subtotal**"] + [f"**{s} ({100 * s / max(1, tot[r]):.0f}%)**" for s, r in zip(fsum, runs)])
    rows.append(["", "**wrong answers**"] + [f"**{tot[r]}**" for r in runs])
    md = to_md(rows)
    pooled = Counter()
    by_t = {}
    for r in runs:
        for qt, c in t["by_type"][r].items():
            for m, n in c.items():
                by_t.setdefault(m, Counter())[qt] += n
                pooled[m] += n
    trows = [["mode (pooled over runs)"] + [SHORT[x] for x in TYPES] + ["all"]]
    for fam, modes in fams:
        for m in modes:
            if pooled[m]:
                trows.append([m] + [str(by_t[m].get(x, 0)) for x in TYPES] + [str(pooled[m])])
    md += "\nBy question type (wrong answers pooled over the runs above):\n\n" + to_md(trows)
    Path(a.out + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
