#!/usr/bin/env python
"""Score every evaluated document individually, for the per-document analyses.

A LongBench cell is a mean over documents, and a correct mean is consistent
with many wrong per-document decisions that happen to cancel. Three tables and
one figure need the documents themselves: the oracle upper bound on any
selector of this form, the selector's decision quadrants, and the right panel
of the safety figure.

Each document is scored with the same official evaluator OpenCompass uses,
called on a one-element list, so a document's score here is exactly its
contribution to the cell. Output:

    results/per_document.csv
        model,base,task,doc,score_base,score_corr,score_selector,kl_margin

``kl_margin`` is KL(lambda=0) - KL(lambda=1) as the selector measured it:
positive means the correction looked better on the probe token. It is empty
where the run predates gate logging.

    python scripts/export_per_document.py --runs /path/to/opencompass_outputs
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rescue.models import opencompass_root  # noqa: E402

# The official LongBench metrics live in OpenCompass. They are imported when
# the work starts rather than at module scope, so --help and an import of this
# module work on a checkout that has not set OPENCOMPASS_ROOT yet.
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench")


def evaluators() -> dict:
    sys.path.insert(0, str(opencompass_root()))
    try:
        from opencompass.datasets.longbench.evaluators import (
            LongBenchClassificationEvaluator, LongBenchCodeSimEvaluator,
            LongBenchCountEvaluator, LongBenchF1Evaluator,
            LongBenchRetrievalEvaluator, LongBenchRougeEvaluator)
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the LongBench evaluators from {opencompass_root()}: {exc}\n"
            "Set OPENCOMPASS_ROOT to your OpenCompass checkout."
        ) from exc
    f1, rouge = LongBenchF1Evaluator, LongBenchRougeEvaluator
    return {
        "qasper": f1, "multifieldqa_en": f1, "narrativeqa": f1, "hotpotqa": f1,
        "2wikimqa": f1, "musique": f1, "triviaqa": f1,
        "gov_report": rouge, "qmsum": rouge, "multi_news": rouge, "samsum": rouge,
        "trec": LongBenchClassificationEvaluator,
        "passage_count": LongBenchCountEvaluator,
        "passage_retrieval_en": LongBenchRetrievalEvaluator,
        "lcc": LongBenchCodeSimEvaluator, "repobench": LongBenchCodeSimEvaluator,
    }



P = "llama31_8b_instruct"
RFC = "-rfc-b128tok-lam1-noimpact-objRESCUE-"
STYLE = {"snapkv": "recentsnapkv", "laprox": "global_layer-recentlaprox",
         "lava": "recentlava", "rkv": "recentrkv", "h2o": "recenth2o"}
BASE_STEM = {
    "snapkv": f"{P}-snapkv-b128tok-snapkvV_base",
    "laprox": f"{P}-laprox-b128tok-abl_baseline",
    "lava": f"{P}-lava-b128tok-p128_lava_base",
    "rkv": f"{P}-rkv-b128tok-rkvV_base",
    "h2o": f"{P}-h2o-b128tok-h2oV_base",
}
SEL_STEM = {
    "snapkv": f"{P}{RFC}recentsnapkv-scale_additive-fidelity0_1p1-fidp1_snapkv",
    "laprox": f"{P}{RFC}global_layer-recentlaprox-scale_additive-fidelity0_1p1-fidp1_laprox",
    "lava": f"{P}{RFC}recentlava-scale_additive-fidelity0_1p1-p128_lava_fid",
    "rkv": f"{P}{RFC}recentrkv-scale_additive-fidelity0_1p1-fidp1_rkv",
    "h2o": f"{P}{RFC}recenth2o-scale_additive-fidelity0_1p1-fidp1_h2o",
}
# Cells re-run under their own name after a fix; same flags, same checkpoint.
# h2o/narrativeqa had no complete gate log: llfloor_h2o was launched without
# RESCUE_GATE_LOG, and the partial logs that survived came from other arms.
# gatenq_h2o is that cell re-run under the shipped protocol with gate logging
# on; it reproduces llfloor_h2o's score exactly (30.59) and carries all 200
# margins. Every other cell's margins are read from the run named here.
ALT = {("rkv", "narrativeqa"): f"{P}{RFC}recentrkv-scale_additive-fidelity0_1p1-p128_nq_rkv_fid",
       ("h2o", "narrativeqa"): f"{P}{RFC}recenth2o-scale_additive-fidelity0_1p1-gatenq_h2o"}
# The selector SCORES come from the llfloor re-run, which is the protocol the
# paper reports (the abstention floor enabled on every backbone). The KL
# MARGINS do not exist in that run -- it was launched without gate logging --
# so they are read from SEL_STEM above, which is the same probe on the same
# documents and differs only in what it does when both candidates fall below
# the floor: 24 of 19,562 logged decisions. The two are joined per document.
FLOOR_STEM = {
    "snapkv": f"{P}{RFC}recentsnapkv-scale_additive-fidelity0_1p1-llfloor_snapkv",
    "laprox": f"{P}{RFC}global_layer-recentlaprox-scale_additive-fidelity0_1p1-llfloor_laprox",
    "lava": f"{P}{RFC}recentlava-scale_additive-fidelity0_1p1-llfloor_lava",
    "rkv": f"{P}{RFC}recentrkv-scale_additive-fidelity0_1p1-llfloor_rkv",
    "h2o": f"{P}{RFC}recenth2o-scale_additive-fidelity0_1p1-llfloor_h2o",
}


def corr_stem(base: str) -> str:
    return f"{P}{RFC}{STYLE[base]}-scale_additive-p128_corr_{base}"


def details(runs: Path, stem: str, task: str):
    """Per-document (prediction, references) from the newest completed run."""
    d = runs / f"{stem}-longbench_{task}"
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
            refs = v.get("references")
            if isinstance(refs, str):
                # OpenCompass writes references with repr(), i.e. Python literal
                # syntax rather than JSON -- single quotes, and apostrophes
                # inside the answers -- so json.loads cannot read them back.
                try:
                    refs = ast.literal_eval(refs)
                except Exception:
                    refs = [refs]
            if isinstance(refs, str):
                refs = [refs]
            out[k] = (v["predictions"], refs)
        if out:
            return out
    return None


def per_doc(task: str, det, EVAL) -> dict[str, float]:
    ev = EVAL[task]()
    return {k: float(ev.score([p], [r])["score"]) for k, (p, r) in det.items()}


def kl_margins(runs: Path, stem: str, task: str) -> list[float]:
    """KL(lambda=0) - KL(lambda=1) per document, in evaluation order."""
    d = runs / f"{stem}-longbench_{task}"
    for run in sorted(glob.glob(str(d / "*/")), reverse=True):
        outs = sorted(glob.glob(os.path.join(run, "logs", "infer", "*", "*.out")))
        if not outs:
            continue
        txt = open(outs[-1], errors="ignore").read()
        m = re.findall(
            r"\[FIDELITY\].*?lam0\.0=([0-9.eE+-]+)\s+lam1\.0=([0-9.eE+-]+)", txt)
        if m:
            return [float(a) - float(b) for a, b in m]
    return []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "per_document.csv")
    args = p.parse_args()

    EVAL = evaluators()
    rows = []
    for base in BASE_STEM:
        for task in TASKS:
            sel = ALT.get((base, task), SEL_STEM[base])   # gate logs live here
            db = details(args.runs, BASE_STEM[base], task)
            dc = details(args.runs, corr_stem(base), task)
            df = details(args.runs, FLOOR_STEM[base], task) or details(args.runs, sel, task)
            if not (db and dc and df):
                continue
            keys = sorted(set(db) & set(dc) & set(df),
                          key=lambda x: int(x) if x.isdigit() else x)
            if not keys:
                continue
            sb, sc, sf = (per_doc(task, db, EVAL), per_doc(task, dc, EVAL),
                          per_doc(task, df, EVAL))
            marg = kl_margins(args.runs, sel, task)
            for i, k in enumerate(keys):
                rows.append(("llama3_8b", base, task, k,
                             f"{sb[k]:.6f}", f"{sc[k]:.6f}", f"{sf[k]:.6f}",
                             f"{marg[i]:.9g}" if i < len(marg) else ""))
            print(f"  {base:8s} {task:21s} {len(keys):4d} documents", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "base", "task", "doc",
                    "score_base", "score_corr", "score_selector", "kl_margin"])
        w.writerows(rows)
    print(f"{len(rows)} documents -> {args.out}")


if __name__ == "__main__":
    main()
