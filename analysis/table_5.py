#!/usr/bin/env python
"""Table 5 (tab:budget_sweep). LongBench per task at B=256 and B=1024.

The budget sweep of Section 4.5, printed per task rather than pooled: the same
five base policies against themselves under RESCUE, at the two budgets above
the one the paper works at.

    python analysis/table_5.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longbench_grid import render  # noqa: E402

CAPTION = (r"LongBench at $B_{\text{total}}=256L$ and $1024L$, "
           r"Llama-3.1-8B-Instruct. $^{\dagger}$The average is over the $16$ "
           r"tasks, unweighted by document count.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--format", choices=["latex"], default="latex")
    a = p.parse_args()
    tex = render([(r"$B_{\text{total}}=256L$", "llama3_8b", 256),
                  (r"$B_{\text{total}}=1024L$", "llama3_8b", 1024)],
                 CAPTION, "tab:budget_sweep", placement="!t")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(tex + "\n", encoding="utf-8")
    else:
        print(tex)


if __name__ == "__main__":
    main()
