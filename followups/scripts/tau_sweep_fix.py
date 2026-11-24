#!/usr/bin/env python
"""Corrected replay of the abstention-floor sweep (Table 18).

tau_sweep.py accumulates `y if live else x`, where x is the STANDALONE base
run's per-document score. For a document the selector itself rejected, the
selector run's own dumped output IS the base cache's output, so substituting
another job's generation is wrong whenever the two jobs diverge -- which they
do on a handful of documents per task. At tau = 1e-6 nothing newly abstains on
Qwen, yet the replay still moved Qwen3/H2O by +0.15 over 16 tasks, because five
PassageRetrieval documents alone swing that task by 2.50.

Fix: substitute the base run's output ONLY for documents the replay newly
abstains (the run accepted them, but max|KL| < tau). Documents the run itself
rejected keep the run's own score.

No GPU. Writes tau_sweep_fixed.csv next to this file.
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tau_sweep as T

OUT = Path(__file__).resolve().parent / "tau_sweep_fixed.csv"


def main():
    EV = T.evaluators()
    rows = []
    for (model, base), (bstem, sstem, ltmpl) in T.SPEC.items():
        for task in T.TASKS:
            if ltmpl is None:
                g = T.gate(sstem, task)
            else:
                tag = T._LALT.get((base, task), T._LLOG[base])
                g = T.gate(ltmpl.format(tag=tag), task)
            db, ds = T.details(bstem, task), T.details(sstem, task)
            if not (db and ds and g):
                print(f"skip {model}/{base}/{task}"); continue
            keys = sorted(set(db) & set(ds), key=lambda x: int(x) if x.isdigit() else x)
            if len(keys) != len(g):
                print(f"skip {model}/{base}/{task}: {len(keys)} vs {len(g)}"); continue
            ev = EV[task]()
            sb = [float(ev.score([db[k][0]], [db[k][1]])["score"]) for k in keys]
            ss = [float(ev.score([ds[k][0]], [ds[k][1]])["score"]) for k in keys]
            for tau in T.TAUS:
                tot = nfloor = nrej = 0.0
                for (k0, k1, acc), x, y in zip(g, sb, ss):
                    if not acc:
                        tot += y; nrej += 1.0
                    elif max(abs(k0), abs(k1)) >= tau:
                        tot += y
                    else:
                        tot += x; nfloor += 1.0
                rows.append((model, base, task, tau, len(keys),
                             sum(sb)/len(sb), tot/len(keys),
                             nfloor/len(keys), nrej/len(keys)))
            print(f"{model:10s} {base:7s} {task:22s} n={len(keys):4d} "
                  f"base={sum(sb)/len(sb):6.2f} tau1e-6={rows[-len(T.TAUS)][6]:6.2f} "
                  f"run={sum(ss)/len(ss):6.2f}")
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model","base","task","tau","n","score_base","score_rescue",
                    "floor_frac","reject_frac"])
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
