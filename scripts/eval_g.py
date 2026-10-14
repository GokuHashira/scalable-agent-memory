"""Evaluate run directories with G's local evidence metrics and write docs/midterm/retrieval_table.md."""

import argparse
import glob
from pathlib import Path

from mem27.retrieval.evalg import write_reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="run dirs or glob patterns, e.g. 'runs/p*-lme_s_g-*'")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--table", default="docs/midterm/retrieval_table.md")
    args = parser.parse_args(argv)
    run_dirs = sorted({Path(p) for pattern in args.runs for p in glob.glob(pattern) if Path(p).is_dir()})
    if not run_dirs:
        raise SystemExit(f"no run directories match {args.runs}")
    metrics = write_reports(run_dirs, Path(args.questions), Path(args.table))
    print(f"evaluated {len(metrics)} runs -> {args.table}")


if __name__ == "__main__":
    main()
