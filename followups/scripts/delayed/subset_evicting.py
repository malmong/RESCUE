#!/usr/bin/env python
"""The deferred-eviction control, restricted to the documents where eviction actually happened.

The deferred-eviction control holds the uncompressed cache for w=8 decode
steps. Where the scored answer is produced inside that window, the arm never
evicts anything that reaches the metric and its generation is byte-identical
to dense -- which is 53% of all documents. Averaging over those is averaging
dense into a compression result.

This recomputes every arm on the complement: the documents where the delayed
arm's post-processed generation DIFFERS from dense. Scores come from
OpenCompass's own evaluator classes, not a reimplementation, and every arm is
first re-scored on the FULL set and checked against the number the run
reported -- if that check fails the task is dropped rather than reported.

No GPU. Reads only dumped eval details.
"""
import ast, csv, glob, json, os, sys
from pathlib import Path

ROOT = Path("/path/to/kv_cache")
sys.path.insert(0, str(ROOT / "opencompass"))
from opencompass.datasets.longbench.evaluators import (  # noqa: E402
    LongBenchF1Evaluator, LongBenchRougeEvaluator, LongBenchClassificationEvaluator,
    LongBenchCountEvaluator, LongBenchRetrievalEvaluator, LongBenchCodeSimEvaluator)

EVAL = {t: LongBenchF1Evaluator for t in
        ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa")}
EVAL.update({t: LongBenchRougeEvaluator for t in ("gov_report", "qmsum", "multi_news", "samsum")})
EVAL["trec"] = LongBenchClassificationEvaluator
EVAL["passage_count"] = LongBenchCountEvaluator
EVAL["passage_retrieval_en"] = LongBenchRetrievalEvaluator
EVAL["lcc"] = EVAL["repobench"] = LongBenchCodeSimEvaluator

OUT = {"2wikimqa":32,"hotpotqa":32,"musique":32,"passage_count":32,"passage_retrieval_en":32,
       "triviaqa":32,"lcc":64,"multifieldqa_en":64,"repobench":64,"trec":64,
       "narrativeqa":128,"qasper":128,"samsum":128,"gov_report":512,"multi_news":512,"qmsum":512}

OC = ROOT / "runs/opencompass_outputs"
DLY = ROOT / "followups/scripts/delayed/runs"

def want(model, arm, base, budget, task):
    for r in csv.DictReader(open(ROOT / "rescue-release/results/scores.csv")):
        if (r["model"] == model and r["arm"] == arm and r["base"] == base
                and r["budget"] == budget and r["task"] == task):
            return float(r["score"])
    return None

def _ref(v):
    """`references` is dumped as a repr of the dataset object, not the object.
    Iterating the raw string walks characters and silently produces ~1.0
    scores, which is what the full-set verification below is there to catch."""
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return [v]
    return v


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    det = {}
    for k, v in (d.get("details") or {}).items():
        if isinstance(v, dict) and "predictions" in v:
            det[str(k)] = {"predictions": v["predictions"],
                           "references": _ref(v.get("references"))}
    return float(d["score"]), det

def pick(globs, task_key, target):
    """Newest results json whose reported score matches `target` (or any, if None)."""
    cands = []
    for g in globs:
        for run in glob.glob(g):
            for f in glob.glob(os.path.join(run, "results", "*", f"LongBench_{task_key}*.json")):
                try:
                    s, det = load(f)
                except Exception:
                    continue
                if det and (target is None or abs(s - target) < 0.02):
                    cands.append((os.path.getmtime(f), f, s, det))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1:]

BASE = os.environ.get("SUBSET_BASE", "snapkv")
BASE_GLOB = {"snapkv": "llama31_8b_instruct-snapkv-b128tok-snapkvV_base-longbench_{t}/*/",
             "h2o":    "llama31_8b_instruct-h2o-b128tok-h2oV_base-longbench_{t}/*/"}
RESC_GLOB = {"snapkv": "*rfc-b128tok-lam1-noimpact-objRESCUE-recentsnapkv-scale_additive-fidelity0_1p1-*-longbench_{t}/*/",
             "h2o":    "*rfc-b128tok-lam1-noimpact-objRESCUE-recenth2o-scale_additive-fidelity0_1p1-*-longbench_{t}/*/"}
DLY_GLOB  = {"snapkv": "llama3_8b-snapkv-b128-dly8-all-*/*/",
             "h2o":    "llama3_8b-h2o-b128-dly8-all-*/*/"}
DLYR_GLOB = {"snapkv": "llama3_8b-rescue-b128-snapkv-dly8-all-*/*/",
             "h2o":    "llama3_8b-rescue-b128-h2o-dly8-all-*/*/"}


def arms_for(task):
    oc_t = task                       # opencompass dirs use "repobench"
    dly_t = "repobench-p" if task == "repobench" else task
    out = {}
    specs = [
        ("dense",   [str(OC / f"llama31_8b_instruct-full-b0.3ratio-dense16-longbench_{oc_t}/*/")],
         oc_t, want("llama3_8b", "dense", "", "0", task)),
        ("base",    [str(OC / BASE_GLOB[BASE].format(t=oc_t))],
         oc_t, want("llama3_8b", "base", BASE, "128", task)),
        ("rescue",  [str(OC / RESC_GLOB[BASE].format(t=oc_t))],
         oc_t, want("llama3_8b", "rescue", BASE, "128", task)),
        ("dly8",    [str(DLY / DLY_GLOB[BASE])], dly_t, None),
        ("dly8_r",  [str(DLY / DLYR_GLOB[BASE])], dly_t, None),
    ]
    for name, g, key, target in specs:
        got = pick(g, key, target)
        if got is None:
            return None, f"{name} missing (target={target})"
        out[name] = got
    return out, None

ARMS = ["base", "rescue", "dly8", "dly8_r", "dense"]
rows = []
for task in OUT:
    arms, err = arms_for(task)
    if err:
        print(f"  skip {task}: {err}", file=sys.stderr); continue
    ev = EVAL[task]()
    ids = None
    for a in ARMS:
        s = set(arms[a][2])
        ids = s if ids is None else (ids & s)
    ids = sorted(ids, key=lambda x: int(x) if str(x).isdigit() else x)
    if not ids:
        print(f"  skip {task}: no shared ids", file=sys.stderr); continue

    # verification: re-score the FULL set and demand the run's own number back
    ok, full = True, {}
    for a in ARMS:
        _, rep, det = arms[a]
        p = [det[i]["predictions"] for i in ids]
        r = [det[i]["references"] for i in ids]
        got = ev.score(p, r)
        got = float(list(got.values())[0]) if isinstance(got, dict) else float(got)
        full[a] = got
        if abs(got - rep) > 0.05:
            print(f"  DROP {task}/{a}: re-scored {got:.2f} != reported {rep:.2f}", file=sys.stderr)
            ok = False
    if not ok:
        continue

    # Two filters. "dly8" conditions on one of the arms being compared, which
    # biases the subset against that arm (diverging from dense usually means
    # doing worse); "base" conditions on the common reference every arm is
    # measured from. If the two tell the same story the selection is not
    # driving the result, so both are reported.
    d_dense = arms["dense"][2]
    subs = {}
    for key, ref in (("dly8", "dly8"), ("base", "base")):
        d = arms[ref][2]
        subs[key] = [i for i in ids if d_dense[i]["predictions"] != d[i]["predictions"]]
    parts = {}
    for key, sub in subs.items():
        if not sub:
            parts[key] = None; continue
        pp = {}
        for a in ARMS:
            det = arms[a][2]
            got = ev.score([det[i]["predictions"] for i in sub], [det[i]["references"] for i in sub])
            pp[a] = float(list(got.values())[0]) if isinstance(got, dict) else float(got)
        parts[key] = pp
    rows.append((OUT[task], task, len(ids), len(subs["dly8"]), full, parts["dly8"],
                 len(subs["base"]), parts["base"]))

rows.sort()

def block(title, idx_n, idx_p):
    print(f"\n===== {title}")
    print(f"{'band':>6}{'docs':>7}{'subset':>8}  |{'base':>7}{'+RESC':>7}{'dly8':>7}{'dly+R':>7}{'dense':>7}"
          f"   gap closed: +RESC / dly8 / dly+R")
    def agg(sel):
        n = sum(r[2] for r in sel); m = sum(r[idx_n] for r in sel)
        o = {}
        for a in ARMS:
            v = [r[idx_p][a] for r in sel if r[idx_p]]
            o[a] = sum(v)/len(v) if v else float('nan')
        return n, m, o
    def line(label, sel):
        n, m, o = agg(sel)
        g = o["dense"] - o["base"]
        pc = lambda a: 100*(o[a]-o["base"])/g if g else float('nan')
        print(f"{label:>6}{n:7d}{m:8d}  |" + "".join(f"{o[a]:7.2f}" for a in ARMS)
              + f"   {pc('rescue'):6.1f}% {pc('dly8'):6.1f}% {pc('dly8_r'):6.1f}%")
    for band in (32, 64, 128, 512):
        sel = [r for r in rows if r[0] == band]
        if sel: line(str(band), sel)
    line("ALL", rows)
    line(">=64", [r for r in rows if r[0] >= 64])

print(f"\n[base policy: {BASE}]")
print(f"{'task':20s}{'out':>5}{'docs':>6}{'dly8!=dense':>12}{'base!=dense':>12}")
for r in rows:
    print(f"{r[1]:20s}{r[0]:5d}{r[2]:6d}{r[3]:12d}{r[6]:12d}")
block("filter A: documents where the DELAYED arm diverged from dense", 3, 5)
block("filter B: documents where the BASE policy diverged from dense", 6, 7)
