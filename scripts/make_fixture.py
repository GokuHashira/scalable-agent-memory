"""Write the deterministic offline fixture (stand-in for K's make_fixture.py)."""

import argparse

from mem27.retrieval.fixture import write_fixture


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="target number of turn notes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/fixture")
    args = parser.parse_args(argv)
    counts = write_fixture(args.out, args.n, args.seed)
    print(f"wrote {counts['notes']} notes and {counts['questions']} questions to {args.out}")


if __name__ == "__main__":
    main()
