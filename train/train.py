#!/usr/bin/env python
"""Train one residual scorer, end to end.

A scorer is specific to a (model, base policy) pair, because its target is that
base policy's own misses. Training is two stages:

  1. cache features   one pass per training document, capturing the 18 features
                      and the oracle future-attention target. This is the
                      expensive stage and it writes tens of GB, so its output
                      goes to $RESCUE_FEATURE_ROOT rather than into the repo.
  2. fit the MLP      minutes on one GPU, reading only the cache.

    python train/train.py --model llama3_8b --base snapkv

The trained scorers the paper reports are already in ``results/checkpoints/``;
running this is only necessary to reproduce training itself.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.registry import available  # noqa: E402

# The corpus mix the paper trains on: Natural Questions plus arXiv. Both are
# out of domain for LongBench, which is the point -- the scorer is not fitted
# to the evaluation distribution.
CORPORA = ["nq", "nq_extra", "arxiv"]
CORPUS_SET = "set1_nq520_arxiv"
BASES = ["snapkv", "laprox", "h2o", "lava", "rkv"]


def feature_root() -> Path:
    return Path(os.environ.get("RESCUE_FEATURE_ROOT", REPO_ROOT / "assets" / "train"))


def run(cmd: list[str], gpu: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")])
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=available())
    p.add_argument("--base", required=True, choices=BASES)
    p.add_argument("--gpu", default="0")
    p.add_argument("--budget-tokens", type=int, default=128,
                   help="the scorer's target is the base policy's miss set at this budget, "
                        "so a scorer for B=256 or B=1024 is a different scorer")
    p.add_argument("--skip-cache", action="store_true",
                   help="features are already cached; fit the MLP only")
    p.add_argument("--limit", type=int, help="documents per corpus (smoke tests)")
    args = p.parse_args()

    corpora_dir = Path(os.environ.get("RESCUE_CORPUS_DIR", feature_root() / "corpora"))
    if not args.skip_cache:
        missing = [c for c in CORPORA
                   if not (corpora_dir / f"{c.replace('nq_extra', 'nq_corpus_extra').replace('nq', 'nq_corpus').replace('arxiv', 'arxiv_corpus')}.jsonl").exists()]
        if missing:
            print(f"training corpora not found under {corpora_dir}.\n"
                  f"Build them first:  bash scripts/build_corpora.sh", file=sys.stderr)
            raise SystemExit(1)
        for corpus in CORPORA:
            cmd = [sys.executable, "train/cache_features.py",
                   "--corpus", corpus, "--model", args.model, "--base", args.base,
                   "--budget-tokens", str(args.budget_tokens), "--gpu", "0"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            run(cmd, args.gpu)

    run([sys.executable, "train/train_scorer.py",
         "--set", CORPUS_SET, "--model", args.model, "--base", args.base,
         "--budget-tokens", str(args.budget_tokens), "--gpu", "0"], args.gpu)

    print(f"\ncheckpoint -> results/checkpoints/  (--base {args.base}, --model {args.model})")


if __name__ == "__main__":
    main()
