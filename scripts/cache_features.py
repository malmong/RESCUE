#!/usr/bin/env python
"""Cache the ORIGINAL artifact recipe's per-(doc, layer) features + rescue mask.

The recovered rescue_train_listwise_full.py recomputes a full LLM forward for
every document on every one of its 15 epochs ("expected to take hours" -- ~43 h
for the largest corpus set here). Nothing it recomputes depends on the MLP being
trained: the model is frozen, so the features, the LaProx scores, the oracle
distribution and therefore the rescue mask are all deterministic per document.
Caching them once and replaying 15 epochs over the cache produces byte-identical
losses and gradients while doing 1/15th of the LLM work.

Everything numerical is imported from the recovered original rather than
reimplemented, so the cache cannot drift from the recipe it is meant to
reproduce. What is reproduced exactly:
  - candidates = all positions except sink(4) and the last OBS_WINDOW(32)
  - b_remaining = BUDGET_TOKENS - len(protected)      (= 92, not 60)
  - features = raw9 + rank9, per kv-head then head-averaged, ranks over the
    FULL candidate population
  - rescue = topB(oracle) \\ topB(laprox)
  - reference continuation generated greedily once per doc (as the original does)

Usage: cache_original_features.py --corpus nq --gpu 0 --shard 0/6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

# Paths are relative to the repo, and the feature cache goes wherever
# RESCUE_FEATURE_ROOT points (it grows to tens of GB, so it does not belong
# inside the checkout).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ROOT = Path(os.environ.get("RESCUE_FEATURE_ROOT", REPO_ROOT / "assets" / "train"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from rescue.future_contrib.extract import capture_sequence_chunked  # noqa: E402

# numerical core, imported verbatim from the recovered original
from rescue.models import available, load, DEFAULT_MODEL  # noqa: E402
from rescue import features as ORIGMOD  # the 18 features (raw9 + rank9)

SINK_TOKENS = ORIGMOD.SINK_TOKENS
OBS_WINDOW = ORIGMOD.OBS_WINDOW
BUDGET_TOKENS = ORIGMOD.BUDGET_TOKENS
cfg_rkv_window = 8          # BaselineConfig.rkv_window_size default
CFG_RKV_MIX = 0.07          # BaselineConfig.rkv_mix_lambda (official code default)
CFG_RKV_RETAIN = 0.1        # BaselineConfig.rkv_retain_ratio
ORACLE_HORIZON = ORIGMOD.ORACLE_HORIZON
MIN_HORIZON = ORIGMOD.MIN_HORIZON
RAW_ORDER = ORIGMOD.RAW_ORDER
ALL_FEATURES = ORIGMOD.ALL_FEATURES
MAX_SEQ_LEN = ORIGMOD.MAX_SEQ_LEN

CORPORA = {
    # name: (jsonl, prompt template, max_out_len)
    "nq":        ("nq_corpus.jsonl",        ORIGMOD.NQ_PROMPT_TMPL,    ORIGMOD.NQ_MAX_OUT_LEN),
    "nq_extra":  ("nq_corpus_extra.jsonl",  ORIGMOD.NQ_PROMPT_TMPL,    ORIGMOD.NQ_MAX_OUT_LEN),
    "arxiv":     ("arxiv_corpus.jsonl",     ORIGMOD.ARXIV_PROMPT_TMPL, ORIGMOD.ARXIV_MAX_OUT_LEN),
    # few-shot-shaped corpora: the demonstration block IS the context, so the
    # prompt is a thin wrapper, mirroring TREC's own LongBench template
    "classify":  ("classify_corpus.jsonl",
                  'Please determine the intent of the query below. Here are some examples.'
                  '\n\n{context}\n{input}', 64),
    "newsgroups": ("newsgroups_corpus.jsonl",
                   'Please determine the topic of the document below. Here are some examples.'
                   '\n\n{context}\n{input}', 64),
    # code completion, shaped like lcc's own template
    "code":      ("code_corpus.jsonl",
                  'Please complete the code given below. \n{context}Next line of code:\n', 64),
}
CORPUS_DIR = Path(os.environ.get("RESCUE_CORPUS_DIR", ROOT / "corpora"))
OUT_ROOT = Path(os.environ.get("RESCUE_FEATURE_ROOT", ROOT)) / "features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--model", default="llama3_8b", choices=available(),
                    help="Features are QK logits and K/V norms of THIS model, and the oracle "
                         "target is its own future attention, so a cache (and the scorer "
                         "trained from it) is valid only for the model that produced it. "
                         "Non-default models write to features_<...>_<model> and tag the "
                         "checkpoint, so the two can never be mixed up.")
    ap.add_argument("--target", default="rescue", choices=["rescue", "gt"],
                    help="'rescue' = top-B(oracle) MINUS top-B(base), the policy-conditioned "
                         "residual this paper proposes. 'gt' = top-B(oracle) itself, i.e. the "
                         "full future-importance target a policy-agnostic predictor would use. "
                         "The 'gt' target does not depend on --base at all, so ONE cache and ONE "
                         "scorer serve every base policy -- that independence is exactly the "
                         "control: same features, same architecture, same corpus, same budget, "
                         "differing only in whether the base policy's own hits are excluded.")
    ap.add_argument("--base", default="laprox", choices=["laprox", "snapkv", "h2o", "lava", "rkv"],
                    help="Which base ranker's mistakes the rescue set is defined against. "
                         "C_i is trained to predict tokens the ORACLE keeps and the BASE "
                         "evicts, so the target set differs per base -- a LaProx-trained "
                         "scorer bolted onto a SnapKV base would be correcting the wrong "
                         "ranker's errors. Features are identical across bases; only the "
                         "rescue mask changes.")
    ap.add_argument("--budget-tokens", type=int, default=BUDGET_TOKENS,
                    help="KV budget the rescue set is defined against. The target is "
                         "top-B(oracle) minus top-B(base) with B = budget - |protected|, "
                         "so a scorer trained at one budget is solving a different "
                         "problem at another: B is 92 at 128 tokens and 956 at 1024. "
                         "Caches for a non-default budget go to their own directory.")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    device = torch.device(f"cuda:{args.gpu}")

    fname, tmpl, max_out_len = CORPORA[args.corpus]
    rows = [json.loads(l) for l in (CORPUS_DIR / fname).open(encoding="utf-8") if l.strip()]
    rows = rows[si::sn]
    if args.limit:
        rows = rows[: args.limit]
    if args.target == "gt":
        # base-independent by construction -- one cache for all policies
        root = ROOT / "features_gt"
    else:
        root = OUT_ROOT if args.base == "laprox" else (ROOT / f"features_{args.base}")
    if args.model != DEFAULT_MODEL:
        root = root.parent / f"{root.name}_{args.model}"
    if int(args.budget_tokens) != BUDGET_TOKENS:
        root = root.parent / f"{root.name}_b{int(args.budget_tokens)}"
    out_dir = root / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.ctime()}] {args.corpus} shard {args.shard}: {len(rows)} docs on {device}", flush=True)
    MODEL_TYPE = {"llama3_8b": "llama", "mistral_7b": "mistral",
                  "qwen3_8b": "qwen3"}[args.model]
    tokenizer = AutoTokenizer.from_pretrained(str(load(args.model).path))
    model = AutoModelForCausalLM.from_pretrained(
        str(load(args.model).path), torch_dtype=torch.bfloat16,
        attn_implementation="eager").to(device)   # capture_sequence_chunked patches eager only
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    kv_repeat = num_heads // num_kv_heads
    # `or` not a getattr default: both configs DEFINE head_dim and set it to
    # None, so the default never fires. Llama survives because transformers
    # fills it in at config load; Mistral leaves it None and this crashed.
    head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // num_heads)
    scaling = head_dim ** -0.5

    from rescue.base_scores import get_o_proj_block as _shared_o_proj_block
    from rescue.base_scores import base_head_score as _shared_base_head_score

    o_cache: dict[int, list[torch.Tensor]] = {}

    def get_o_proj_block(layer_idx: int, kv_h: int) -> torch.Tensor:
        return _shared_o_proj_block(layer_idx, kv_h, model=model, num_heads=num_heads,
                                    kv_repeat=kv_repeat, head_dim=head_dim, o_cache=o_cache)

    def base_head_score(base, q_full, k_full, v_full, cand, cur_t, kv_h, o_block):
        return _shared_base_head_score(base, q_full, k_full, v_full, cand, cur_t, kv_h,
                                       o_block, kv_repeat=kv_repeat, scaling=scaling,
                                       cfg_rkv_window=cfg_rkv_window)

    def compute_doc_layer(boundary, layer_idx, q_full, k_full, v_full):
        """Identical to the original's nested compute_doc_layer."""
        cur_t = boundary - 1
        full_len = q_full.shape[1]
        horizon_end = min(cur_t + 1 + ORACLE_HORIZON, full_len)
        if horizon_end - (cur_t + 1) < MIN_HORIZON:
            return None
        protected = set(range(min(SINK_TOKENS, cur_t + 1)))
        protected.update(range(max(0, cur_t - OBS_WINDOW + 1), cur_t + 1))
        all_pos = torch.arange(cur_t + 1, device=device)
        cand_mask = torch.ones(cur_t + 1, dtype=torch.bool, device=device)
        for p in protected:
            cand_mask[p] = False
        cand_idx = all_pos[cand_mask]
        n_cand = int(cand_idx.numel())
        if n_cand < 32:
            return None
        b_remaining = max(1, min(int(args.budget_tokens) - len(protected), n_cand))

        per_F, per_R, per_feat = [], [], []
        for kv_h in range(num_kv_heads):
            F_h = ORIGMOD.oracle_head_distribution(q_full, k_full, cand_idx, cur_t + 1,
                                                   horizon_end, kv_h, kv_repeat, scaling)
            o_block = get_o_proj_block(layer_idx, kv_h)
            R_h = base_head_score(args.base, q_full, k_full, v_full, cand_idx,
                                  cur_t, kv_h, o_block)
            lw = ORIGMOD.recent_qk_logits(q_full, k_full, cand_idx, cur_t, OBS_WINDOW,
                                          kv_h, kv_repeat, scaling)
            w = lw.shape[0]
            xw = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2.0
            per_F.append(F_h)
            per_R.append(R_h)
            per_feat.append({
                "current_qk": lw[-1], "mean_qk": lw.mean(dim=0), "max_qk": lw.max(dim=0).values,
                "var_qk": lw.var(dim=0, unbiased=False),
                "slope_qk": (lw * xw.unsqueeze(1)).sum(dim=0) / (xw * xw).sum().clamp_min(1e-6),
                "k_norm": k_full[kv_h, cand_idx, :].float().norm(p=2, dim=-1),
                "v_norm": v_full[kv_h, cand_idx, :].float().norm(p=2, dim=-1),
                "vwo_norm": (v_full[kv_h, cand_idx, :].float() @ o_block).norm(p=2, dim=-1),
                "age": (cur_t - cand_idx.float()),
            })
        F_all = torch.stack(per_F, dim=0).mean(dim=0)
        R_all = torch.stack(per_R, dim=0).mean(dim=0)
        gt_top = set(F_all.topk(b_remaining).indices.tolist())
        if args.target == "gt":
            rescue = gt_top
        else:
            rescue = gt_top - set(R_all.topk(b_remaining).indices.tolist())
        if not rescue:
            return None
        fm = {k: torch.stack([d[k] for d in per_feat], dim=0).mean(dim=0) for k in per_feat[0]}
        # NOTE: base_score was tried as a 19th/20th feature and REVERTED --
        # it fit the rescue target better (val 6.35 vs 6.53) but collapsed the
        # eviction result on LaProx (qasper 34.82 -> 25.76, below the 27.85
        # baseline; multifieldqa_en 53.47 -> 45.61). Since the rule and corpus
        # must be shared across bases, a feature that breaks LaProx is
        # unusable whatever it does elsewhere.
        cols = [fm[k] for k in RAW_ORDER] + [ORIGMOD.rank_full_pop(fm[k]) for k in RAW_ORDER]
        X = torch.stack(cols, dim=-1)
        mask = torch.zeros(n_cand, dtype=torch.bool, device=device)
        mask[list(rescue)] = True
        return X, mask

    ok = skipped = failed = 0
    for i, r in enumerate(rows):
        doc_id = f"{args.corpus}_{si:02d}_{i:04d}"
        out_path = out_dir / f"{doc_id}.pt"
        if out_path.exists():
            ok += 1
            continue
        t0 = time.time()
        try:
            prompt = tmpl.format(context=r["context"], input=r.get("input", ""))
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            ids = tokenizer(text, return_tensors="pt", truncation=False,
                            add_special_tokens=False)["input_ids"]
            budget = max(1, MAX_SEQ_LEN - max_out_len)
            if int(ids.shape[1]) > budget:
                half = budget // 2
                ids = torch.cat([ids[:, :half], ids[:, -(budget - half):]], dim=1)
            ids = ids.to(device)
            boundary = int(ids.shape[1])

            # the original flips to sdpa for generation only, then restores eager
            prev_impl = model.config._attn_implementation
            try:
                model.config._attn_implementation = "sdpa"
                ref = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_out_len,
                    min_new_tokens=1, do_sample=False, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id)
            finally:
                model.config._attn_implementation = prev_impl
            cap = capture_sequence_chunked(model, ref, MODEL_TYPE, chunk_size=4096)

            Xs, masks, layers = [], [], []
            for layer_idx in range(num_layers):
                if layer_idx not in cap.qkv:
                    continue
                q, k, v = (t.to(device) for t in cap.qkv[layer_idx])
                out = compute_doc_layer(boundary, layer_idx, q, k, v)
                if out is None:
                    continue
                X, mask = out
                Xs.append(X.cpu())
                masks.append(mask.cpu())
                layers.append(layer_idx)
            del cap
            torch.cuda.empty_cache()
        except Exception as e:
            failed += 1
            print(f"[{time.ctime()}] {doc_id} FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
        if not Xs:
            skipped += 1
            continue
        torch.save({"X": Xs, "rescue": masks, "layers": layers, "doc_id": doc_id,
                    "source": args.corpus, "boundary": boundary,
                    "target": args.target, "base": args.base, "model": args.model,
                    "features": ALL_FEATURES}, out_path)
        ok += 1
        print(f"[{time.ctime()}] {doc_id} ({i+1}/{len(rows)}) layers={len(Xs)} "
              f"n_cand={Xs[0].shape[0]} ({time.time()-t0:.1f}s)", flush=True)

    print(f"[{time.ctime()}] DONE {args.corpus} {args.shard}: ok={ok} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
