#!/usr/bin/env python
"""Check that analysis/table_<n>.py still matches the paper's table <n>.

Table numbers move whenever a table is added to the body, and a script whose
name no longer matches what it generates is worse than one that was never
numbered. Point this at the .tex and it reports the mapping.

    python tests/test_numbering.py --tex path/to/paper.tex
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = REPO_ROOT / "analysis"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tex", required=True, type=Path)
    args = p.parse_args()

    tex = io.open(args.tex, encoding="utf-8").read()
    tables = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
    figures = re.findall(r"\\label\{(fig:[^}]+)\}", tex)

    bad = []
    for i, label in enumerate(tables, 1):
        path = ANALYSIS / f"table_{i}.py"
        if not path.exists():
            bad.append(f"Table {i} ({label}): no {path.name}")
            continue
        m = re.search(r'"""Table (\d+) \((tab:[^)]+)\)\.', path.read_text(encoding="utf-8"))
        if not m or int(m.group(1)) != i or m.group(2) != label:
            got = f"{m.group(1)} / {m.group(2)}" if m else "no header"
            bad.append(f"{path.name}: paper says {i} / {label}, script says {got}")
        print(f"  Table {i:2d}  {label:26s} {path.name}")

    for i, label in enumerate(figures, 1):
        path = ANALYSIS / f"figure_{i}.py"
        status = "(no script)" if not path.exists() else path.name
        print(f"  Figure {i:d}  {label:26s} {status}")
        if path.exists():
            m = re.search(r'"""Figure (\d+) \((fig:[^)]+)\)\.',
                          path.read_text(encoding="utf-8"))
            if not m or int(m.group(1)) != i or m.group(2) != label:
                got = f"{m.group(1)} / {m.group(2)}" if m else "no header"
                bad.append(f"{path.name}: paper says {i} / {label}, script says {got}")

    if bad:
        print("\nMISMATCHED:")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)
    print(f"\n  {len(tables)} tables and {len(figures)} figures match the paper")


if __name__ == "__main__":
    main()
