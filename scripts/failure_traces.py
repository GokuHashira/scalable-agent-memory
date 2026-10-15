"""List failure-trace candidates (one plan finds all gold sessions, another none) from shared run dirs."""

import argparse
import json
from pathlib import Path

from mem27.retrieval.reports import failure_candidates
from mem27.schema import load_questions, read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="run directories")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", default="docs/midterm/failure_candidates.json")
    args = parser.parse_args(argv)
    runs = {Path(d).name.split("-")[0]: list(read_jsonl(Path(d) / "bundles.jsonl")) for d in args.runs}
    found = failure_candidates(runs, load_questions(args.questions))
    Path(args.out).write_text(json.dumps(found, indent=2))
    print(f"{len(found)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
