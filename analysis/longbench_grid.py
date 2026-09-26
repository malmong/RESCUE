"""Shared renderer for the per-task LongBench grid (Table 5).

Both are the same object: sixteen task columns plus an unweighted average, with
each base policy printed above its RESCUE arm and the better of the pair in
bold. They differ only in what the blocks are -- backbones for Table 3, budgets
for Table 5 -- so the layout lives here once.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BASE_LABEL, Scores  # noqa: E402

COLS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
        "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
        "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench"]
POLICIES = ["snapkv", "laprox", "lava", "rkv", "h2o"]

HEADER = r"""\begingroup
\providecommand{\LBmissing}{\textemdash}
\providecommand{\LBhead}[1]{\normalfont\upshape #1}
\providecommand{\LBavg}[2]{\shortstack{#1\\{\scriptsize (#2)}}}
\providecommand{\LBblock}[1]{\multicolumn{18}{c}{#1}}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.95}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l*{16}{r}r@{}}
\toprule
\multirow{2}{*}{Method} & \multicolumn{3}{c}{Single Document QA}
 & \multicolumn{3}{c}{Multi Document QA}
 & \multicolumn{3}{c}{Summarization}
 & \multicolumn{3}{c}{Few-shot Learning}
 & \multicolumn{2}{c}{Synthetic}
 & \multicolumn{2}{c}{Code}
 & \multirow{2}{*}{Avg.$^{\dagger}$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\cmidrule(lr){11-13}\cmidrule(lr){14-15}\cmidrule(lr){16-17}
 & \LBhead{NrtvQA} & \LBhead{Qasper} & \LBhead{MF-en}
 & \LBhead{HotpotQA} & \LBhead{2WikiMQA} & \LBhead{MuSiQue}
 & \LBhead{GovRep} & \LBhead{QMSum} & \LBhead{MultiNews}
 & \LBhead{TREC} & \LBhead{TriviaQA} & \LBhead{SAMSum}
 & \LBhead{PCount} & \LBhead{PR-en} & \LBhead{Lcc} & \LBhead{RB-P}
 & \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}%
}
\endgroup"""


def _fmt(v: float | None, bold: bool) -> str:
    if v is None:
        return r"\LBmissing"
    # round() before formatting, so a value sitting exactly on a half (a 16-task
    # mean of two-decimal scores often does) rounds the same way here as in
    # every other script, which all go through round().
    v = round(v, 2)
    return (r"\textbf{%.2f}" % v) if bold else ("%.2f" % v)


def block(sc: Scores, title: str, model: str, budget: int,
          out: list[str]) -> None:
    out.append(r"\LBblock{%s} \\" % title)
    out.append(r"\midrule")
    for i, key in enumerate(POLICIES):
        base = [sc.get(model, "base", key, budget, t) for t in COLS]
        resc = [sc.get(model, "rescue", key, budget, t) for t in COLS]
        if any(v is None for v in base + resc):
            out.append("%% %s: missing cells, skipped" % BASE_LABEL[key])
            continue
        ab, ar = sum(base) / len(COLS), sum(resc) / len(COLS)
        out.append(BASE_LABEL[key])
        out.append("& " + " & ".join(_fmt(x, x > y + 1e-9)
                                     for x, y in zip(base, resc)))
        out.append("& %s \\\\" % _fmt(ab, ab > ar + 1e-9))
        out.append(r"\quad + RESCUE")
        out.append("& " + " & ".join(_fmt(y, y > x + 1e-9)
                                     for x, y in zip(base, resc)))
        out.append("& %s \\\\" % _fmt(ar, ar > ab + 1e-9))
        if i < len(POLICIES) - 1:
            out.append(r"\addlinespace[2pt]")


def render(blocks: list[tuple[str, str, int]], caption: str,
           label: str, placement: str = "p") -> str:
    sc = Scores()
    out = [r"\begin{table}[%s]" % placement, r"\centering",
           r"\caption{%s}" % caption, r"\label{%s}" % label, HEADER]
    for n, (title, model, budget) in enumerate(blocks):
        if n:
            out.append(r"\midrule")
        block(sc, title, model, budget, out)
    out += [FOOTER, r"\end{table}"]
    return "\n".join(out)
