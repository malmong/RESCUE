import csv, statistics as st
from collections import defaultdict
from pathlib import Path
R = Path("/path/to/kv_cache/followups/scripts/tau_sweep.csv")
rows = list(csv.DictReader(open(R, encoding="utf-8")))
TAUS = sorted({float(r["tau"]) for r in rows})
MODELS = ["llama3_8b", "mistral_7b", "qwen3_8b"]

# per (model, tau): mean over 5 bases of (mean over 16 tasks of rescue-base)
print("=== mean gain over base, averaged over 5 policies x 16 tasks ===")
print(f"{'tau':>8s} " + " ".join(f"{m:>12s}" for m in MODELS) + f"{'abstain%':>10s}")
for tau in TAUS:
    line, absf = [], []
    for m in MODELS:
        per_base = defaultdict(list)
        for r in rows:
            if r["model"] == m and float(r["tau"]) == tau:
                per_base[r["base"]].append(float(r["score_rescue"]) - float(r["score_base"]))
                absf.append(float(r["abstain_frac"]))
        line.append(st.mean(st.mean(v) for v in per_base.values()))
    print(f"{tau:8.0e} " + " ".join(f"{d:>+12.2f}" for d in line)
          + f"{100*st.mean(absf):>9.1f}%")

print("\n=== abstention rate by model ===")
print(f"{'tau':>8s} " + " ".join(f"{m:>12s}" for m in MODELS))
for tau in TAUS:
    out = []
    for m in MODELS:
        v = [float(r["abstain_frac"]) for r in rows
             if r["model"] == m and float(r["tau"]) == tau]
        out.append(100 * st.mean(v))
    print(f"{tau:8.0e} " + " ".join(f"{d:>11.1f}%" for d in out))

print("\n=== per-policy gain at the shipped tau=1e-6 (sanity vs paper) ===")
for m in MODELS:
    s = []
    for b in ("snapkv", "laprox", "lava", "rkv", "h2o"):
        v = [(float(r["score_rescue"]), float(r["score_base"])) for r in rows
             if r["model"] == m and r["base"] == b and float(r["tau"]) == 1e-6]
        s.append(f"{b}={st.mean(x-y for x, y in v):+.2f}")
    print(f"  {m:11s} " + "  ".join(s))
