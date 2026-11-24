#!/usr/bin/env python
"""Collect the compression-curve scores straight from the OpenCompass run dirs.

scripts/evaluate.py also writes results/summaries/<name>.csv, but that file is
not usable here for two reasons: collect() is called once per template group
and opens the destination with "w", so the five raw-template tasks overwrite
the eleven chat-template ones; and it filters on the task list, where the
summary calls the last task "repobench-p" and the list calls it "repobench",
so that task is dropped either way. The run directories carry all sixteen.

    python followups/scripts/collect_curve.py
"""
from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs_sweep"
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p")


def scores(stem: str) -> dict[str, float]:
    """Every task score of one arm, read across both template groups."""
    out: dict[str, float] = {}
    for group in ("all-chat", "all-raw"):
        d = RUNS / f"{stem}-{group}"
        if not d.is_dir():
            continue
        for f in sorted(glob.glob(str(d / "*" / "summary" / "summary_*.csv")),
                        reverse=True):
            got = {}
            for row in csv.reader(open(f, encoding="utf-8")):
                if not row or not row[0].startswith("LongBench_"):
                    continue
                for cell in reversed(row):
                    if re.fullmatch(r"\d+\.\d+", (cell or "").strip()):
                        got[row[0].removeprefix("LongBench_")] = float(cell)
                        break
            if got:
                out.update(got)
                break
    return out


def mean(s: dict[str, float]) -> float | None:
    have = [s[t] for t in TASKS if t in s]
    return sum(have) / len(have) if len(have) == len(TASKS) else None


def main() -> None:
    rows = []
    for pol in ("snapkv", "h2o"):
        for b in (192, 512, 768):
            base = scores(f"llama3_8b-{pol}-b{b}-curve_b{b}")
            resc = scores(f"llama3_8b-rescue-b{b}-{pol}-curve_b{b}")
            if not base and not resc:
                continue
            mb, mr = mean(base), mean(resc)
            rows.append((pol, b, len(base), len(resc), mb, mr))

    w = max(len(f"{p}/b{b}") for p, b, *_ in rows) if rows else 10
    print(f"{'arm':<{w}}  {'base':>7} {'RESCUE':>7} {'gain':>6}   tasks")
    for pol, b, nb, nr, mb, mr in rows:
        g = f"{mr - mb:+.2f}" if (mb is not None and mr is not None) else "--"
        print(f"{pol}/b{b:<{w-len(pol)-2}}  "
              f"{mb if mb is not None else float('nan'):7.2f} "
              f"{mr if mr is not None else float('nan'):7.2f} {g:>6}   "
              f"{nb}/{len(TASKS)}, {nr}/{len(TASKS)}")

    out = Path(__file__).resolve().parent / "curve_scores.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["base", "budget", "arm", "task", "score"])
        for pol in ("snapkv", "h2o"):
            for b in (192, 512, 768):
                for arm, stem in (("base", f"llama3_8b-{pol}-b{b}-curve_b{b}"),
                                  ("rescue", f"llama3_8b-rescue-b{b}-{pol}-curve_b{b}")):
                    for t, v in sorted(scores(stem).items()):
                        wr.writerow([pol, b, arm, t, f"{v:.2f}"])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
