#!/usr/bin/env python
"""Does LU-KV's head-budget allocation have anything to redistribute at our
operating point?  (follow-up to the LU-KV comparison.)

LU-KV ships its profile as a (99, 32, 8) array of per-(layer, head) PRUNE
ratios indexed by global compression ratio.  Its allocator forces every head
to keep at least `min_keep_limit = max(round(0.01 * N), sink + window)`
tokens (step2_compute_curve.py:63, prune_threshold_int=99), so below ~1% of
the document per head there is nothing left to reallocate and the curve
collapses to uniform.

This measures, per evaluated document, which row of their table our budgets
land on and how much head differentiation survives there.  No GPU.
"""
import glob, json, os, statistics as st
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from kvbench.configs.models import MODELS

CURVE = Path("/path/to/kv_cache/followups/scripts/lukv_upstream/LU-KV/evaluation/"
             "curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy")
RUNS  = Path("/path/to/kv_cache/runs/opencompass_outputs")
STEM  = "llama31_8b_instruct-snapkv-b128tok-snapkvV_base"
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench")
BUDGETS = (128, 192, 256, 512, 768, 1024)
SINK, WIN = 4, 32

curve = np.load(CURVE)                      # [99, layer, head] prune ratio
assert curve.shape[0] == 99, curve.shape
keep_rate = 1.0 - curve                     # [99, layer, head]
# per-row head differentiation: spread of keep rate across heads inside a layer
spread = (keep_rate.max(axis=2) - keep_rate.min(axis=2)).mean(axis=1)   # [99]
# relative differentiation is the honest unit: an 0.06 spread at 2% mean is large
rel = spread / keep_rate.mean(axis=(1, 2))

tok = AutoTokenizer.from_pretrained(MODELS["llama31_8b_instruct"], local_files_only=True)

lens = []
for t in TASKS:
    f = None
    for run in sorted(glob.glob(str(RUNS / f"{STEM}-longbench_{t}" / "*/")), reverse=True):
        g = glob.glob(os.path.join(run, "predictions", "*", "*.json"))
        if g:
            f = g[0]; break
    if not f:
        print("missing", t); continue
    recs = json.load(open(f, encoding="utf-8"))
    lens += [len(tok(v["origin_prompt"], add_special_tokens=False)["input_ids"])
             for v in recs.values()]
lens = np.array([n for n in lens if n > 0])
print(f"{len(lens)} evaluated documents, median {int(np.median(lens))} tokens\n")

print("LU-KV's own table: how much head differentiation exists at each global ratio")
print(f"{'ratio':>7} {'mean keep':>10} {'head spread':>12} {'spread/mean':>12}")
for i in (79, 89, 94, 96, 97, 98):
    print(f"{(i+1)/100:7.2f} {keep_rate[i].mean():10.4f} {spread[i]:12.4f} {rel[i]:12.3f}")

print("\nWhere our budgets land on that table, per document")
print(f"{'B':>5} {'keep rate (med)':>16} {'ratio (med)':>12} {'row':>5} "
      f"{'docs at floor row 98':>22} {'spread/mean (med)':>18}")
for B in BUDGETS:
    kr = np.minimum(1.0, B / lens)                 # per-head keep fraction
    ratio = 1.0 - kr
    row = np.clip(np.round(ratio * 100).astype(int) - 1, 0, 98)
    at_floor = float((row >= 98).mean()) * 100
    print(f"{B:5d} {np.median(kr):16.5f} {np.median(ratio):12.4f} "
          f"{int(np.median(row)):5d} {at_floor:21.1f}% {np.median(rel[row]):18.3f}")

print("\nFloor: LU-KV keeps >= max(1% of N, sink+window) per head.")
for B in BUDGETS:
    floor = np.maximum(np.round(0.01 * lens), SINK + WIN)
    binds = float((B <= floor).mean()) * 100
    print(f"  B={B:5d}: our per-head budget is at or below LU-KV's floor on "
          f"{binds:5.1f}% of documents")
