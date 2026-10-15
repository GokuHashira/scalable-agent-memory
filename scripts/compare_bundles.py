"""Compare two bundles.jsonl files on note ids and scores, ignoring timing (FR-019, SC-004). Exit 1 on mismatch."""

import argparse
import sys

from mem27.retrieval.reports import compare_bundle_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args(argv)
    result = compare_bundle_files(args.a, args.b)
    print(f"compared={result['compared']} identical={result['identical']} mismatched={len(result['mismatched'])}")
    for qid in result["mismatched"][:20]:
        print(f"  mismatch: {qid}")
    return 1 if result["mismatched"] or result["compared"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
