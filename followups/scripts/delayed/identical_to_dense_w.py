#!/usr/bin/env python
"""How often is "delayed eviction" actually measuring dense inference?

The deferred-eviction control defers the base policy's eviction by w decode steps. Where
the scored answer is produced inside that window, the arm never evicts
anything that matters, so its score is dense's score -- and the control is
not a compression method at that point. Counting EOS inside the window is a
bad proxy (TREC and LCC rarely emit EOS at all, yet TREC's delayed score
equals dense to the decimal). The direct test needs no re-scoring: compare
the post-processed generations themselves.

No GPU. Reads only the dumped eval details.
"""
import glob, json, os, sys
from pathlib import Path

ROOT = Path("/path/to/kv_cache")
OC = ROOT / "runs/opencompass_outputs"
DLY = ROOT / "followups/scripts/delayed/runs"
OUT = {"2wikimqa":32,"hotpotqa":32,"musique":32,"passage_count":32,
       "passage_retrieval_en":32,"triviaqa":32,
       "lcc":64,"multifieldqa_en":64,"repobench-p":64,"trec":64,
       "narrativeqa":128,"qasper":128,"samsum":128,
       "gov_report":512,"multi_news":512,"qmsum":512}

def details(path):
    d = json.load(open(path, encoding="utf-8"))
    det = d.get("details") or {}
    out = {}
    for k, v in det.items():
        if isinstance(v, dict) and "predictions" in v:
            out[str(k)] = v["predictions"]
    return d.get("score"), out

def newest(pattern, task_key):
    """Newest results json under a run-dir glob, for one task."""
    best = None
    for run in glob.glob(pattern):
        for f in glob.glob(os.path.join(run, "results", "*", f"*{task_key}.json")):
            m = os.path.getmtime(f)
            if best is None or m > best[0]:
                best = (m, f)
    return best[1] if best else None

POL = os.environ.get("IDENT_POL", "snapkv")
W = os.environ.get("IDENT_W", "8")
rows = []
for task, band in OUT.items():
    oc_task = task
    dense_f = newest(str(OC / f"llama31_8b_instruct-full-b0.3ratio-dense16-longbench_{oc_task}" / "*/"), oc_task)
    dly_f = newest(str(DLY / f"llama3_8b-{POL}-b128-dly{W}-all-*" / "*/"), oc_task)
    if not dense_f or not dly_f:
        print(f"  skip {task}: dense={bool(dense_f)} dly={bool(dly_f)}", file=sys.stderr)
        continue
    s_dense, p_dense = details(dense_f)
    s_dly, p_dly = details(dly_f)
    ids = sorted(set(p_dense) & set(p_dly), key=lambda x: int(x) if x.isdigit() else x)
    if not ids:
        print(f"  skip {task}: no shared ids", file=sys.stderr); continue
    same = sum(1 for i in ids if p_dense[i] == p_dly[i])
    rows.append((band, task, len(ids), same, s_dense, s_dly))

rows.sort()
print(f"{'task':20s}{'out':>5}{'docs':>6}{'dense와 동일':>13}{'비율':>8}   dense  delay")
for band, task, n, same, sd, sl in rows:
    print(f"{task:20s}{band:5d}{n:6d}{same:13d}{100*same/n:7.1f}%  {sd:7.2f}{sl:7.2f}")

print()
for b in (32, 64, 128, 512):
    sub = [r for r in rows if r[0] == b]
    n = sum(r[2] for r in sub); s = sum(r[3] for r in sub)
    if n: print(f"  출력 {b:3d}: {s:5d}/{n:5d} = {100*s/n:5.1f}% 가 dense와 동일")
n = sum(r[2] for r in rows); s = sum(r[3] for r in rows)
print(f"  전체    : {s:5d}/{n:5d} = {100*s/n:5.1f}%")
