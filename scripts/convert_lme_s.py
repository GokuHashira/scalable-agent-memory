"""Convert LongMemEval_S into notes.jsonl / questions.jsonl (FR-001–003)."""

import argparse
import sys

from mem27.retrieval.longmemeval import ConversionError, convert_lme_s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument("--out", default="data/lme_s_g")
    parser.add_argument("--parent-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        counts = convert_lme_s(args.raw, args.out, args.parent_tokens)
    except ConversionError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 1
    print(" ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
