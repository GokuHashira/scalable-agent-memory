"""Fill a markdown report template from result files, so every number in the report regenerates.

Placeholders:
  {{table:runs/_tables/main_test_final.md}}          the file's contents (markdown table)
  {{json:runs/_tables/x.json:a.b.0.c:pct1}}           a value from a JSON file; key path uses '.' (list indices
                                                      allowed); format: pct1 (x100, 1 decimal), pct1s (signed),
                                                      f3, f2, f1, int, raw. Keys containing '.' can be quoted
                                                      with [..], e.g. groups.[supersede:explicit].fresh@5
  {{csv:runs/_tables/b.csv:row_key=val,other=val:col:fmt}}  a cell of a CSV row matched on columns
  {{file:docs/metrics.md:2:1}}                       a file's text, skipping its first 2 lines, headings demoted by 1
Missing files or keys render as **MISSING(...)** and are counted; exit status 1 if any are missing.

  python scripts/build_report.py --template docs/REPORT_V.template.md --out docs/REPORT_V.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PH = re.compile(r"\{\{(table|json|csv|file):([^}]+)\}\}")


def fmt(v, f: str) -> str:
    if v is None:
        return "–"
    if f == "pct1":
        return f"{100 * float(v):.1f}"
    if f == "pct1s":
        return f"{100 * float(v):+.1f}"
    if f == "f3":
        return f"{float(v):.3f}"
    if f == "f2":
        return f"{float(v):.2f}"
    if f == "f1":
        return f"{float(v):.1f}"
    if f == "int":
        return f"{int(round(float(v))):,}"
    return str(v)


def split_path(p: str) -> list[str]:
    out, buf, br = [], "", False
    for ch in p:
        if ch == "[" and not buf:
            br = True
        elif ch == "]" and br:
            br = False
        elif ch == "." and not br:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    out.append(buf)
    return out


def get_json(obj, path: str):
    for k in split_path(path):
        if isinstance(obj, list):
            obj = obj[int(k)]
        else:
            obj = obj[k]
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    missing = []
    cache: dict[str, object] = {}

    def sub(m: re.Match) -> str:
        kind, arg = m.group(1), m.group(2)
        try:
            if kind == "table":
                return Path(arg).read_text().rstrip("\n")
            if kind == "file":
                path, skip, demote = (arg.split(":") + ["0", "0"])[:3]
                lines = Path(path).read_text().splitlines()[int(skip):]
                return "\n".join(("#" * int(demote) + l) if l.startswith("#") else l for l in lines).strip("\n")
            if kind == "json":
                file, key, f = arg.split(":", 2) if arg.count(":") >= 2 else (*arg.split(":", 1), "raw")
                if file not in cache:
                    cache[file] = json.loads(Path(file).read_text())
                return fmt(get_json(cache[file], key), f)
            if kind == "csv":
                file, match, col, f = arg.split(":", 3)
                conds = dict(c.split("=", 1) for c in match.split(","))
                with open(file) as fh:
                    for row in csv.DictReader(fh):
                        if all(row.get(k) == v for k, v in conds.items()):
                            return fmt(row[col], f)
                raise KeyError(match)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{kind}:{arg} ({type(e).__name__}: {e})")
            return f"**MISSING({kind}:{arg})**"
        return m.group(0)

    text = PH.sub(sub, Path(a.template).read_text())
    Path(a.out).write_text(text)
    print(f"wrote {a.out}; {len(missing)} missing placeholders")
    for x in missing:
        print("  MISSING", x)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
