#!/usr/bin/env python3
"""Four-arm table for the delayed-eviction probe (the deferred-eviction control).

Undelayed numbers come from the release scores.csv; the delayed numbers are
read straight out of the run dirs, the same way collect_curve.py does it --
evaluate.py's own collect() overwrites one template group with the other.
"""
import csv, glob, re, sys
from pathlib import Path

ROOT = Path("/path/to/kv_cache")
REL = ROOT / "rescue-release/results/scores.csv"
RUNS = ROOT / "followups/scripts/delayed/runs"

# LongBench max_out_len, the axis the delay window competes against.
OUTLEN = {
    "2wikimqa": 32, "hotpotqa": 32, "musique": 32, "passage_count": 32,
    "passage_retrieval_en": 32, "triviaqa": 32,
    "lcc": 64, "multifieldqa_en": 64, "repobench": 64, "trec": 64,
    "narrativeqa": 128, "qasper": 128, "samsum": 128,
    "gov_report": 512, "multi_news": 512, "qmsum": 512,
}
TASKS = list(OUTLEN)


def undelayed(base, budget):
    out = {}
    for r in csv.DictReader(open(REL)):
        if r["model"] != "llama3_8b":
            continue
        if r["arm"] == "dense" and r["budget"] == "0":
            out.setdefault("dense", {})[r["task"]] = float(r["score"])
        elif r["base"] == base and r["budget"] == budget and r["arm"] in ("base", "rescue"):
            out.setdefault(r["arm"], {})[r["task"]] = float(r["score"])
    return out


def delayed(stem):
    out = {}
    for grp in ("all-chat", "all-raw"):
        for f in glob.glob(str(RUNS / f"{stem}-{grp}/*/summary/summary_*.csv")):
            for r in csv.DictReader(open(f)):
                t = r["dataset"].replace("LongBench_", "")
                t = "repobench" if t == "repobench-p" else t
                out[t] = float(list(r.values())[-1])
    return out


def mean(d):
    vals = [d[t] for t in TASKS if t in d]
    return sum(vals) / len(vals) if len(vals) == len(TASKS) else None


MARK = re.compile(r"Start inferencing \[.*?/LongBench_(\S+?)\]")


def degeneracy(stem):
    """Share of documents whose answer finished inside the delay window.

    OpenCompass writes one .out per template group -- all 11 (or 5) datasets
    run inside it, named after the first -- so split on the per-dataset
    "Start inferencing" marker instead of trusting the file name.
    """
    per = {}
    for grp in ("all-chat", "all-raw"):
        for log in glob.glob(str(RUNS / f"{stem}-{grp}/*/logs/infer/*/*.out")):
            task = None
            for line in open(log, errors="ignore"):
                m = MARK.search(line)
                if m:
                    task = m.group(1)
                    per.setdefault(task, [0, 0])
                elif task and line.startswith("[DELAY]"):
                    per[task][1] += 1
                    per[task][0] += int(line.rsplit("dense=", 1)[1].strip())
    return {t: tuple(v) for t, v in per.items() if v[1]}


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "8"
    pol = sys.argv[2] if len(sys.argv) > 2 else "snapkv"
    u = undelayed(pol, "128")
    cols = {
        f"{pol}": u["base"],
        f"{pol}+RESCUE": u["rescue"],
        f"{pol}/dly{w}": delayed(f"llama3_8b-{pol}-b128-dly{w}"),
        f"{pol}/dly{w}+RESCUE": delayed(f"llama3_8b-rescue-b128-{pol}-dly{w}"),
        "dense": u["dense"],
    }
    deg = degeneracy(f"llama3_8b-{pol}-b128-dly{w}")

    names = list(cols)
    print(f"{'task':<22}{'out':>5}" + "".join(f"{n:>22}" for n in names) + f"{'dense-finish':>14}")
    for t in sorted(TASKS, key=lambda x: (OUTLEN[x], x)):
        d, n = deg.get(t, (0, 0))
        share = f"{100*d/n:.0f}% ({d}/{n})" if n else "-"
        print(f"{t:<22}{OUTLEN[t]:>5}" + "".join(f"{cols[n_][t]:>22.2f}" for n_ in names) + f"{share:>14}")
    print(f"{'AVG':<22}{'':>5}" + "".join(f"{(mean(cols[n_]) or float('nan')):>22.2f}" for n_ in names))

    print("\nby output budget")
    for ol in (32, 64, 128, 512):
        sub = [t for t in TASKS if OUTLEN[t] == ol]
        row = [sum(cols[n_][t] for t in sub) / len(sub) for n_ in names]
        d = sum(deg.get(t, (0, 0))[0] for t in sub)
        n = sum(deg.get(t, (0, 0))[1] for t in sub)
        print(f"  out={ol:<4} n={len(sub)}" + "".join(f"{v:>22.2f}" for v in row)
              + f"{(100*d/n if n else 0):>13.0f}%")
