"""Count sessions with no turns in raw LongMemEval files (cleaned vs original release), per file.

  python scripts/count_empty_sessions.py raw/longmemeval/longmemeval_s_cleaned.json raw/longmemeval_orig/longmemeval_s \
      --out runs/_tables/lme_empty_sessions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {}
    for f in a.files:
        data = json.loads(Path(f).read_text())
        n_sess = sum(len(q["haystack_sessions"]) for q in data)
        empty = sum(1 for q in data for s in q["haystack_sessions"] if len(s) == 0)
        hist = sum(1 for q in data if any(len(s) == 0 for s in q["haystack_sessions"]))
        out[Path(f).name] = {"n_questions": len(data), "n_sessions": n_sess, "n_sessions_nonempty": n_sess - empty,
                             "n_empty_sessions": empty, "n_histories_with_empty": hist}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
