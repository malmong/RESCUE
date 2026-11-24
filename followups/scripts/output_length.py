#!/usr/bin/env python
"""Does RESCUE help where the answer is short? (where the gain is)

Per LongBench task: the median length of the model's own output and of the gold
answer, against the mean RESCUE gain over the 5 base policies. Reads the stored
predictions only -- no GPU.
"""
import ast, csv, glob, json, os, statistics as st
from collections import defaultdict
from pathlib import Path

RUNS = Path("/path/to/kv_cache/runs/opencompass_outputs")
S = Path("/path/to/kv_cache/rescue-release/results/scores.csv")
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench")
STEM = "llama31_8b_instruct-snapkv-b128tok-snapkvV_base"


def details(stem, task):
    d = RUNS / f"{stem}-longbench_{task}"
    for run in sorted(glob.glob(str(d / "*/")), reverse=True):
        f = glob.glob(os.path.join(run, "results", "*", "*.json"))
        if not f:
            continue
        det = json.load(open(f[0], encoding="utf-8")).get("details")
        if isinstance(det, dict):
            return det
    return None


gain = defaultdict(dict)
for r in csv.DictReader(open(S, encoding="utf-8")):
    if r["model"] == "llama3_8b" and r["budget"] == "128" and r["arm"] in ("base", "rescue"):
        gain[(r["arm"], r["base"])][r["task"]] = float(r["score"])
BASES = ("snapkv", "laprox", "lava", "rkv", "h2o")


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for p, i in enumerate(o):
            r[i] = p
        return r
    a, b = rk(x), rk(y)
    n = len(a)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** .5
    return num / den


rows = []
for t in TASKS:
    det = details(STEM, t)
    if not det:
        print("missing", t); continue
    pl, gl = [], []
    for v in det.values():
        if not isinstance(v, dict) or "predictions" not in v:
            continue
        pl.append(len(str(v["predictions"]).split()))
        r = v.get("references")
        if isinstance(r, str):
            try: r = ast.literal_eval(r)
            except Exception: r = [r]
        if isinstance(r, str): r = [r]
        if isinstance(r, dict): r = list(r.values())
        if isinstance(r, (list, tuple)) and r:
            gl.append(len(str(r[0]).split()))
    g = st.mean(gain[("rescue", b)][t] - gain[("base", b)][t] for b in BASES)
    rows.append((t, st.median(pl), st.median(gl) if gl else float("nan"), g))

rows.sort(key=lambda r: r[1])
print(f"{'task':24s} {'out(med w)':>11s} {'gold(med w)':>12s} {'gain':>7s}")
for t, p, gd, g in rows:
    print(f"{t:24s} {p:>11.0f} {gd:>12.0f} {g:>+7.2f}")
print(f"\nSpearman(output length, gain)     = {spearman([r[1] for r in rows], [r[3] for r in rows]):+.3f}")
print(f"Spearman(gold answer length, gain) = {spearman([r[2] for r in rows], [r[3] for r in rows]):+.3f}")
short = [r[3] for r in rows if r[1] <= 16]
long_ = [r[3] for r in rows if r[1] > 16]
print(f"\nmean gain, output <= 16 words (n={len(short)}): {st.mean(short):+.2f}")
print(f"mean gain, output  > 16 words (n={len(long_)}): {st.mean(long_):+.2f}")
