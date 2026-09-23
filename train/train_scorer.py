#!/usr/bin/env python
"""Replay the recovered rescue_train_listwise_full.py training loop over cached
features. Same losses, same gradients, ~36x less GPU work.

Everything that defines the recipe is copied from the original exactly:
  MLP           18 -> 32 -> 16 -> 1, ReLU
  loss          -(target * log_softmax(logits)).sum(), target = uniform over rescue
  optimiser     Adam, lr 1e-3, ONE step per (document, layer)
  epochs        15
  split         random.Random(0).shuffle(samples); first 40 docs are validation
  norm stats    mean/std over the first 12 TRAIN docs, layers 0/8/16/24/31 only,
                std clamped at 1e-3
The only departure is where X and the rescue mask come from -- disk instead of a
fresh LLM forward -- which is bit-identical because the model is frozen.

Corpus sets: only the corpus mix changes between them, nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Pin the device BEFORE torch is imported, by reading --gpu out of argv
# directly. Something in this process's startup was taking a CUDA context on
# device 0 regardless of the device the tensors go to: with five trainings
# running on gpu1/2/4/5, gpu0 carried four 554 MiB shells on top of its own
# job, so `nvidia-smi` showed five processes on a card doing one thing. Making
# only the target card visible removes the possibility rather than chasing the
# call that took it. The harness still receives --gpu N in argv, which is what
# scheduler.sh reads to decide a card is occupied.
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", sys.argv[_i + 1]); break
    if _a.startswith("--gpu="):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", _a.split("=", 1)[1]); break

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "train" / "_original"))
sys.path.insert(0, str(REPO_ROOT))
ROOT = Path(os.environ.get("RESCUE_FEATURE_ROOT", REPO_ROOT / "assets" / "train"))
FEATURES = ROOT / "features"
CKPT_DIR = Path(os.environ.get("RESCUE_CKPT_DIR", REPO_ROOT / "results" / "checkpoints"))

N_FEAT = 18
EPOCHS = 15
LR = 1e-3
N_VAL_DOCS = 40
NORM_DOCS = 12
def _norm_layers(n_layers: int) -> list[int]:
    """Five evenly spaced layers, ENDING AT THE LAST ONE.

    This was the literal [0, 8, 16, 24, 31] -- correct for a 32-layer model and
    wrong for anything else. On Qwen3-8B (36 layers) it left layers 32-35 out of
    the normalisation statistics entirely, and those are exactly where v_norm
    and vwo_norm run away: normalised against stats that never saw them, their
    per-layer means land at z = +2.4 to +4.4, against a worst case of +1.7 on
    Llama. The MLP then scores those four layers off four-sigma inputs and
    per_head allocation spreads the damage across the budget. Measured on Qwen,
    MultiFieldQA-en: the correction alone cost -12.70 points against +2.76 on
    Llama, while training loss, the feature pipeline and the lambda=0 path were
    all verified healthy.

    For n_layers=32 this returns [0, 8, 16, 24, 31] exactly, so the Llama and
    Mistral checkpoints are unaffected.
    """
    if n_layers <= 1:
        return [0]
    # Quarter points plus the last layer -- the same rule the literal list
    # followed. Rounding (n-1)/4 instead gives 23 where the original had 24 and
    # would silently change the shipped Llama and Mistral checkpoints.
    return sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})

SETS = {
    "set1_nq_arxiv":        ["nq", "arxiv"],
    "set2_fewshot":         ["nq", "arxiv", "classify", "newsgroups"],
    "set3_code":            ["nq", "arxiv", "code"],
    "set4_fewshot_code":    ["nq", "arxiv", "classify", "newsgroups", "code"],
    # NQ extended to the full 520 the original used (n_train_docs:480 in its log).
    # About half of each NQ batch is skipped here for a too-short continuation,
    # so 520 raw docs is what restores ~260 usable -- the original's actual count.
    "set1_nq520_arxiv":     ["nq", "nq_extra", "arxiv"],
    # 진단용 분해: Qwen 만 NQ 에서 한 건도 skip 되지 않았다(260/260 대
    # Mistral 111/260). skip 조건은 "생성된 continuation < 16 토큰" 이므로
    # Qwen 의 NQ 감독신호만 장황한 생성물에서 나온 것이 된다. 코퍼스를
    # 갈라 학습해 그 차이가 회귀의 원인인지 본다.
    "diag_arxiv_only":      ["arxiv"],
    "diag_nq_only":         ["nq", "nq_extra"],
    "set2_nq520_fewshot":   ["nq", "nq_extra", "arxiv", "classify"],
}


class RescueMLP(nn.Module):
    """Verbatim from the original."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_docs(corpora):
    docs = []
    for c in corpora:
        d = FEATURES / c
        if not d.is_dir():
            raise SystemExit(f"missing cache for corpus {c}")
        for p in sorted(d.glob("*.pt")):
            docs.append((str(p), c))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="Reruns the recipe with a different document split and weight "
                         "init. seed 0 keeps the original split; note the shipped "
                         "checkpoints were trained before torch.manual_seed was set "
                         "here, so re-running seed 0 need not reproduce them exactly. "
                         "Any non-zero seed writes to a _seed<N> suffix, so it cannot "
                         "overwrite a shipped checkpoint.")
    ap.add_argument("--budget-tokens", type=int, default=128,
                    help="Which budget's feature cache to train on; tags the checkpoint.")
    ap.add_argument("--model", default="llama31_8b_instruct",
                    choices=["llama31_8b_instruct", "mistral_7b_instruct_v03", "qwen3_8b"],
                    help="Which model's feature cache to train on; tags the checkpoint.")
    ap.add_argument("--target", default="rescue", choices=["rescue", "gt"],
                    help="Which cached target to train on. 'gt' reads features_gt/ and "
                         "ignores --base, since top-B(oracle) is the same for every policy.")
    ap.add_argument("--base", default="laprox", choices=["laprox", "snapkv", "h2o", "lava", "rkv"],
                    help="Which base ranker's feature cache to train on. The rescue mask "
                         "in each cache is defined against that base's own evictions, so "
                         "a scorer is only valid for the base it was trained against.")
    args = ap.parse_args()
    global FEATURES
    if args.target == "gt":
        # The full-future target is policy-agnostic: top-B(oracle) does not
        # reference any base policy, so one cache and one scorer cover all of
        # them. This is the control for the paper's central claim.
        FEATURES = ROOT / "features_gt"
    elif args.base != "laprox":
        FEATURES = ROOT / f"features_{args.base}"
    if args.model != "llama31_8b_instruct":
        FEATURES = FEATURES.parent / f"{FEATURES.name}_{args.model}"
    if int(args.budget_tokens) != 128:
        FEATURES = FEATURES.parent / f"{FEATURES.name}_b{int(args.budget_tokens)}"
    # CUDA_VISIBLE_DEVICES was pinned to args.gpu at import time, so the
    # target card is the only one visible and it is index 0 here.
    dev = torch.device("cuda:0")
    corpora = SETS[args.set]

    paths = load_docs(corpora)
    # the original shuffles the per-document sample list with seed 0, then peels
    # off the first N_VAL_DOCS for validation
    random.Random(args.seed).shuffle(paths)
    torch.manual_seed(args.seed)
    val_paths, train_paths = paths[:N_VAL_DOCS], paths[N_VAL_DOCS:]
    print(f"[{time.ctime()}] {args.set}: corpora={corpora}  "
          f"train={len(train_paths)} val={len(val_paths)} docs", flush=True)

    # --- normalisation stats: first 12 train docs, 5 sampled layers (as in the original) ---
    # The layer list follows the model's own depth; see _norm_layers.
    n_layers_model = max(
        (li for pp, _ in train_paths[:NORM_DOCS] for li in torch.load(pp, map_location="cpu")["layers"]),
        default=31) + 1
    norm_layers = _norm_layers(n_layers_model)
    # RESCUE_NORM_ALL_LAYERS=1: fit the scaler on EVERY layer instead of five
    # sampled ones. The five-layer rule is inherited from the original and is
    # biased for both backbones -- on Llama the sampled std runs 1.5x the
    # all-layer std -- but the bias is far larger on Qwen3, whose layer 0 has
    # k_norm 146 against 21-25 everywhere else, pulling the sampled mean to
    # 47.9 against a true 28.5. Reading every layer costs nothing here: the
    # features are already on disk.
    if os.environ.get("RESCUE_NORM_ALL_LAYERS"):
        norm_layers = list(range(n_layers_model))
    print(f"  norm layers (of {n_layers_model}): "
          f"{'ALL' if len(norm_layers) == n_layers_model else norm_layers}", flush=True)
    norm_rows = []
    for p, _ in train_paths[:NORM_DOCS]:
        d = torch.load(p, map_location="cpu")
        for li, X in zip(d["layers"], d["X"]):
            if li in norm_layers:
                norm_rows.append(X)
    all_X = torch.cat(norm_rows, dim=0)
    feat_mean = all_X.mean(dim=0).to(dev)
    feat_std = all_X.std(dim=0).clamp_min(1e-3).to(dev)
    print(f"  norm stats from {all_X.shape[0]} rows", flush=True)

    mlp = RescueMLP().to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)

    def run_doc(path, train: bool):
        d = torch.load(path, map_location="cpu")
        losses = []
        for X, mask in zip(d["X"], d["rescue"]):
            n_rescue = int(mask.sum())
            if n_rescue == 0:
                continue
            Xn = ((X.to(dev) - feat_mean) / feat_std)
            target = mask.to(dev).float() / n_rescue
            if train:
                logits = mlp(Xn)
                loss = -(target * F.log_softmax(logits, dim=-1)).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                with torch.no_grad():
                    logits = mlp(Xn)
                    loss = -(target * F.log_softmax(logits, dim=-1)).sum()
            losses.append(float(loss.item()))
        return losses

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(EPOCHS):
        t0 = time.time()
        order = list(range(len(train_paths)))
        random.Random(epoch + 1000 * args.seed).shuffle(order)
        tr = []
        for i in order:
            tr.extend(run_doc(train_paths[i][0], True))
        va = []
        for p, _ in val_paths:
            va.extend(run_doc(p, False))
        tl = sum(tr) / max(1, len(tr))
        vl = sum(va) / max(1, len(va))
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl})
        print(f"  epoch {epoch+1}/{EPOCHS}  train {tl:.4f}  val {vl:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    best = min(history, key=lambda h: h["val_loss"])
    payload = {
        "kind": "rescue_scorer_v1",
        "raw_features": ["current_qk", "mean_qk", "max_qk", "var_qk", "slope_qk",
                         "k_norm", "v_norm", "vwo_norm", "age"],
        "all_features": ["current_qk", "mean_qk", "max_qk", "var_qk", "slope_qk",
                         "k_norm", "v_norm", "vwo_norm", "age"]
                        + [f"rk_{k}" for k in ["current_qk", "mean_qk", "max_qk", "var_qk",
                                                "slope_qk", "k_norm", "v_norm", "vwo_norm", "age"]],
        "recent_window": 32,
        "scaler_mean": feat_mean.cpu(),
        "scaler_scale": feat_std.cpu(),
        "mlp_weights": [l.weight.detach().cpu().clone() for l in mlp.net if isinstance(l, nn.Linear)],
        "mlp_biases": [l.bias.detach().cpu().clone() for l in mlp.net if isinstance(l, nn.Linear)],
        "train_corpora": corpora,
        "score_mode": "softmax",
        "softmax_temperature": 1.0,
        "history": history,
    }
    suffix = "_fullfuture" if args.target == "gt" else ("" if args.base == "laprox" else f"_{args.base}")
    if args.model != "llama31_8b_instruct":
        suffix += f"_{args.model}"
    if int(args.budget_tokens) != 128:
        suffix += f"_b{int(args.budget_tokens)}"
    if int(args.seed) != 0:
        suffix += f"_seed{int(args.seed)}"
    out = CKPT_DIR / f"rescue_original_{args.set}{suffix}.pt"
    torch.save(payload, out)
    (CKPT_DIR / f"{args.set}{suffix}_summary.json").write_text(json.dumps(
        {"set": args.set, "base": args.base, "target": args.target, "corpora": corpora, "n_train": len(train_paths),
         "n_val": len(val_paths), "best_epoch": best, "history": history}, indent=2))
    print(f"[{time.ctime()}] saved {out}  (best val {best['val_loss']:.4f} @ epoch {best['epoch']+1})",
          flush=True)


if __name__ == "__main__":
    main()
