"""Write docs/midterm/index_table.md from <dataset>/index/build_profile.json (FR-018)."""

import argparse
from pathlib import Path

from mem27.retrieval.reports import index_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="docs/midterm/index_table.md")
    args = parser.parse_args(argv)
    md = "# Index build cost\n\n" + index_table(Path(args.dataset) / "index" / "build_profile.json")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
