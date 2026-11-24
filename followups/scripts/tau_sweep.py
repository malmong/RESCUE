#!/usr/bin/env python
"""Abstention-floor sensitivity, recomputed from the stored gate logs.

A follow-up question: how much does tau matter. The shipped value is 1e-6 and the
paper measures it on Llama only. Every gated run logged its own KL pair per
document, so raising the floor can be replayed offline: a document abstains at
tau when max(KL(lam=0), KL(lam=1)) < tau, and an abstaining document is served
the base cache, whose per-document score the base run already holds. No GPU.

Lowering the floor below a run's own value is NOT replayable -- an abstained
document has no correction output -- so the sweep only goes up from 1e-6.

    python followups/scripts/tau_sweep.py
"""
from __future__ import annotations

import ast, csv, glob, json, os, re, sys
from pathlib import Path

RUNS = Path("/path/to/kv_cache/runs/opencompass_outputs")
OC = Path("/path/to/kv_cache/opencompass")
OUT = Path(__file__).resolve().parent / "tau_sweep.csv"

TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench")
BASES = ("snapkv", "laprox", "lava", "rkv", "h2o")
TAUS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

L = "llama31_8b_instruct"
RFC = "-rfc-b128tok-lam1-noimpact-objRESCUE-"
STYLE = {"snapkv": "recentsnapkv", "laprox": "global_layer-recentlaprox",
         "lava": "recentlava", "rkv": "recentrkv", "h2o": "recenth2o"}

# (base stem, score stem, log stem) per model/policy.
SPEC = {}
_LB = {"snapkv": f"{L}-snapkv-b128tok-snapkvV_base",
       "laprox": f"{L}-laprox-b128tok-abl_baseline",
       "lava": f"{L}-lava-b128tok-p128_lava_base",
       "rkv": f"{L}-rkv-b128tok-rkvV_base",
       "h2o": f"{L}-h2o-b128tok-h2oV_base"}
_LLOG = {"snapkv": "fidp1_snapkv", "laprox": "fidp1_laprox",
         "lava": "p128_lava_fid", "rkv": "fidp1_rkv", "h2o": "fidp1_h2o"}
_LALT = {("rkv", "narrativeqa"): "p128_nq_rkv_fid",
         ("h2o", "narrativeqa"): "gatenq_h2o"}
for b in BASES:
    SPEC[("llama3_8b", b)] = (
        _LB[b],
        f"{L}{RFC}{STYLE[b]}-scale_additive-fidelity0_1p1-llfloor_{b}",
        f"{L}{RFC}{STYLE[b]}-scale_additive-fidelity0_1p1-" + "{tag}")
    SPEC[("mistral_7b", b)] = (
        f"mistral_7b_instruct_v03-{b}-b128tok-m32_base_{b}",
        f"mistral_7b_instruct_v03{RFC}{STYLE[b]}-scale_additive-fidelity0_1p1-m32_fid_{b}",
        None)
    SPEC[("qwen3_8b", b)] = (
        f"qwen3_8b-{b}-b128tok-qw_base_{b}",
        f"qwen3_8b{RFC}{STYLE[b]}-scale_additive-fidelity0_1p1-qwfl_{b}",
        None)


def evaluators():
    sys.path.insert(0, str(OC))
    from opencompass.datasets.longbench.evaluators import (
        LongBenchClassificationEvaluator, LongBenchCodeSimEvaluator,
        LongBenchCountEvaluator, LongBenchF1Evaluator,
        LongBenchRetrievalEvaluator, LongBenchRougeEvaluator)
    f1, rg = LongBenchF1Evaluator, LongBenchRougeEvaluator
    return {"qasper": f1, "multifieldqa_en": f1, "narrativeqa": f1,
            "hotpotqa": f1, "2wikimqa": f1, "musique": f1, "triviaqa": f1,
            "gov_report": rg, "qmsum": rg, "multi_news": rg, "samsum": rg,
            "trec": LongBenchClassificationEvaluator,
            "passage_count": LongBenchCountEvaluator,
            "passage_retrieval_en": LongBenchRetrievalEvaluator,
            "lcc": LongBenchCodeSimEvaluator,
            "repobench": LongBenchCodeSimEvaluator}


def details(stem: str, task: str):
    d = RUNS / f"{stem}-longbench_{task}"
    if not d.is_dir():
        return None
    for run in sorted(glob.glob(str(d / "*/")), reverse=True):
        files = glob.glob(os.path.join(run, "results", "*", "*.json"))
        if not files:
            continue
        det = json.load(open(files[0], encoding="utf-8")).get("details")
        if not isinstance(det, dict):
            continue
        out = {}
        for k, v in det.items():
            if not isinstance(v, dict) or "predictions" not in v:
                continue
            r = v.get("references")
            if isinstance(r, str):
                try:
                    r = ast.literal_eval(r)
                except Exception:
                    r = [r]
            if isinstance(r, str):
                r = [r]
            out[k] = (v["predictions"], r)
        if out:
            return out
    return None


PAT = re.compile(r"\[FIDELITY\] probe=\d+ lam0\.0=([0-9.eE+-]+)\s+"
                 r"lam1\.0=([0-9.eE+-]+)\s+->\s+([0-9.]+)")


def gate(stem: str, task: str):
    """(kl0, kl1, accepted) per document, in evaluation order."""
    d = RUNS / f"{stem}-longbench_{task}"
    if not d.is_dir():
        return []
    for run in sorted(glob.glob(str(d / "*/")), reverse=True):
        outs = sorted(glob.glob(os.path.join(run, "logs", "infer", "*", "*.out")))
        if not outs:
            continue
        m = PAT.findall(open(outs[-1], errors="ignore").read())
        if m:
            return [(float(a), float(b), float(c) > 0.5) for a, b, c in m]
    return []


def main():
    EV = evaluators()
    rows = []
    for (model, base), (bstem, sstem, ltmpl) in SPEC.items():
        for task in TASKS:
            if ltmpl is None:
                g = gate(sstem, task)
            else:
                tag = _LALT.get((base, task), _LLOG[base])
                g = gate(ltmpl.format(tag=tag), task)
            db, ds = details(bstem, task), details(sstem, task)
            if not (db and ds and g):
                print(f"skip {model}/{base}/{task} "
                      f"(base={bool(db)} sel={bool(ds)} gate={len(g)})")
                continue
            keys = sorted(set(db) & set(ds),
                          key=lambda x: int(x) if x.isdigit() else x)
            if len(keys) != len(g):
                print(f"skip {model}/{base}/{task}: {len(keys)} docs vs {len(g)} logs")
                continue
            ev = EV[task]()
            sb = [float(ev.score([db[k][0]], [db[k][1]])["score"]) for k in keys]
            ss = [float(ev.score([ds[k][0]], [ds[k][1]])["score"]) for k in keys]
            for tau in TAUS:
                tot = nab = 0.0
                for (k0, k1, acc), x, y in zip(g, sb, ss):
                    live = acc and max(abs(k0), abs(k1)) >= tau
                    tot += y if live else x
                    nab += 0.0 if live else 1.0
                rows.append((model, base, task, tau, len(keys),
                             sum(sb) / len(sb), tot / len(keys),
                             nab / len(keys)))
            print(f"{model:10s} {base:7s} {task:22s} n={len(keys):4d} "
                  f"base={sum(sb)/len(sb):6.2f} tau1e-6={rows[-len(TAUS)][6]:6.2f}")
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "base", "task", "tau", "n",
                    "score_base", "score_rescue", "abstain_frac"])
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
