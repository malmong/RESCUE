#!/usr/bin/env python
"""What compression ratio each token budget actually is (the compression curve).

The paper reports budgets in retained tokens per layer (B_total = 128L). The
A follow-up asks for a compression-ratio curve. The two are the same axis once
the prompt length is known, and every evaluated prompt is stored verbatim in
the OpenCompass predictions, so the mapping is measurable rather than assumed.

Ratio here is retained/original = B / N_prompt, per document, with the same
tokenizer the runs used. No GPU.
"""
import glob, json, os, statistics as st
from pathlib import Path
from transformers import AutoTokenizer
from kvbench.configs.models import MODELS

RUNS = Path("/path/to/kv_cache/runs/opencompass_outputs")
STEM = "llama31_8b_instruct-snapkv-b128tok-snapkvV_base"
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench")
BUDGETS = (128, 192, 256, 512, 768, 1024)

tok = AutoTokenizer.from_pretrained(MODELS["llama31_8b_instruct"], local_files_only=True)

lens, per_task = [], {}
for t in TASKS:
    d = RUNS / f"{STEM}-longbench_{t}"
    f = None
    for run in sorted(glob.glob(str(d / "*/")), reverse=True):
        g = glob.glob(os.path.join(run, "predictions", "*", "*.json"))
        if g:
            f = g[0]; break
    if not f:
        print("missing", t); continue
    recs = json.load(open(f, encoding="utf-8"))
    L = [len(tok(v["origin_prompt"], add_special_tokens=False)["input_ids"])
         for v in recs.values()]
    per_task[t] = L
    lens += L

print(f"{'task':24s} {'n':>5s} {'median tok':>11s} " +
      " ".join(f"{'B='+str(b):>8s}" for b in BUDGETS))
for t in TASKS:
    if t not in per_task: continue
    L = per_task[t]; m = st.median(L)
    print(f"{t:24s} {len(L):>5d} {m:>11.0f} " +
          " ".join(f"{100*b/m:>7.2f}%" for b in BUDGETS))

m = st.median(lens)
print(f"\n{'ALL 16 TASKS':24s} {len(lens):>5d} {m:>11.0f} " +
      " ".join(f"{100*b/m:>7.2f}%" for b in BUDGETS))
print("\nper-document ratio, pooled over all 18,750 evaluated documents:")
for b in BUDGETS:
    r = sorted(100 * b / n for n in lens)
    q = lambda p: r[int(p * (len(r) - 1))]
    print(f"  B={b:>4d}  median {st.median(r):6.2f}%   "
          f"p10 {q(.10):6.2f}%   p90 {q(.90):6.2f}%   "
          f"mean {st.mean(r):6.2f}%")
