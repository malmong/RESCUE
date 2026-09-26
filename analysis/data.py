"""Shared loading and statistics for every table and figure.

One file, ``results/scores.csv``, holds every LongBench cell the paper reports.
Nothing below touches a GPU or the evaluation harness, so the paper's numbers
can be checked from a clean checkout.
"""
from __future__ import annotations

import csv
import random
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rescue.models import LONGBENCH_TASKS  # noqa: E402

SCORES = REPO_ROOT / "results" / "scores.csv"

BASES = ("snapkv", "laprox", "h2o", "lava", "rkv")
BASE_LABEL = {"snapkv": "SnapKV", "laprox": "LaProx", "h2o": "H2O (evict-once)",
              "lava": "LAVa", "rkv": "R-KV"}
TASK_LABEL = {
    "narrativeqa": "NarrativeQA", "qasper": "Qasper",
    "multifieldqa_en": "MultiFieldQA-en", "hotpotqa": "HotpotQA",
    "2wikimqa": "2WikiMQA", "musique": "MuSiQue", "gov_report": "GovReport",
    "qmsum": "QMSum", "multi_news": "MultiNews", "trec": "TREC",
    "triviaqa": "TriviaQA", "samsum": "SAMSum", "passage_count": "PassageCount",
    "passage_retrieval_en": "PassageRetrieval-en", "lcc": "LCC",
    "repobench": "RepoBench-P",
}
MODEL_LABEL = {"llama3_8b": "Llama-3.1-8B-Instruct",
               "mistral_7b": "Mistral-7B-Instruct-v0.3",
               "qwen3_8b": "Qwen3-8B"}


def load() -> dict[tuple[str, str, str, int, str], float]:
    if not SCORES.exists():
        raise SystemExit(
            f"{SCORES} not found.\n"
            "Build it with:  python scripts/export_results.py --runs <opencompass outputs>"
        )
    out = {}
    with open(SCORES, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["model"], r["arm"], r["base"], int(r["budget"]), r["task"])] = float(r["score"])
    return out


class Scores:
    """Lookup over ``results/scores.csv`` that returns None for absent cells."""

    def __init__(self) -> None:
        self._d = load()

    def get(self, model: str, arm: str, base: str = "", budget: int = 128,
            task: str = "") -> float | None:
        return self._d.get((model, arm, base, budget, task))

    def sweep(self, model: str, arm: str, base: str = "", budget: int = 128
              ) -> dict[str, float]:
        """All tasks present for one (model, arm, base, budget)."""
        return {t: v for t in LONGBENCH_TASKS
                if (v := self.get(model, arm, base, budget, t)) is not None}

    def paired(self, model: str, a: str, b: str, base: str = "", budget: int = 128
               ) -> list[tuple[str, float, float]]:
        """Tasks where both arms have a score, so deltas are paired."""
        x, y = self.sweep(model, a, base, budget), self.sweep(model, b, base, budget)
        return [(t, x[t], y[t]) for t in LONGBENCH_TASKS if t in x and t in y]

    def cells(self, model: str, a: str, b: str, budget: int = 128
              ) -> list[tuple[str, str, float, float]]:
        """Every (base, task) cell where both arms scored, across all bases.

        Each base is compared against its own baseline: the rescue set is
        defined by that base's own misses, so a shared reference would mean
        nothing.
        """
        out = []
        for base in BASES:
            for t, u, v in self.paired(model, a, b, base, budget):
                out.append((base, t, u, v))
        return out


def bootstrap_ci(cells: list[tuple[str, str, float, float]], n: int = 10000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean delta and a 95% interval, resampling TASKS rather than cells.

    The cells reuse the same sixteen tasks across five policies, so they are
    not independent draws; resampling tasks moves all five policies together
    and is the interval the paper reports.
    """
    by_task: dict[str, list[float]] = {}
    for _base, task, u, v in cells:
        by_task.setdefault(task, []).append(v - u)
    tasks = sorted(by_task)
    flat = [d for t in tasks for d in by_task[t]]
    if not flat:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        picked = [rng.choice(tasks) for _ in tasks]
        vals = [d for t in picked for d in by_task[t]]
        means.append(st.mean(vals))
    means.sort()
    return (st.mean(flat), means[int(0.025 * n)], means[int(0.975 * n)])


def fmt(x: float | None, width: int = 0, signed: bool = False) -> str:
    if x is None:
        return r"\LBmissing".rjust(width)
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    return s.rjust(width)


def latex_header(caption: str, label: str, cols: str) -> str:
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule"
    )


LATEX_FOOTER = "\\bottomrule\n\\end{tabular}\n\\end{table}"


def write(out: Path | None, text: str) -> None:
    if out is None:
        print(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"-> {out}")
