from pathlib import Path
import os
"""Shared building blocks for the GT-vs-Observed-vs-Future coverage measurement
(paper Figure motivation, panel (b)). See coverage_phase_a.py / coverage_phase_b_lookaheadkv.py
/ coverage_aggregate.py for the actual pipeline.

Candidate definition (matches this project's own production eviction convention,
kvbench/baselines/scoring.py's BaselineConfig defaults):
  - protected prefix: [0, SINK_TOKENS)
  - protected recent: [boundary - RECENT_TOKENS, boundary)
  - candidates (the only positions any method's "importance" is compared over):
    [SINK_TOKENS, boundary - RECENT_TOKENS)
  - observation window (query source for every attention-based score): the last
    OBS_WINDOW real queries before boundary -- a SUBSET of the protected-recent
    region, not a separate protected zone.

All per-method scores are computed per-layer per-kv-head, then mean-reduced over
kv-heads (matching how each baseline's own real per-candidate score already
aggregates GQA groups) and mean-reduced over all 32 layers, giving ONE importance
scalar per candidate TOKEN per document -- matching panel (a)'s single-heatmap-line
visualization convention (not a per-layer/per-head breakdown).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvbench.baselines.scoring import (  # noqa: E402
    BaselineConfig,
    query_to_kv_attention,
    snapkv_head_token_scores,
    laprox_head_token_scores,
    lava_head_token_scores,
)
from kvbench.baselines.rkv import rkv_final_score  # noqa: E402
from kvbench.dense_eviction_hf import _h2o_chunked_prefill  # noqa: E402
from kvbench.learned.oracle_future import _oracle_head_distribution  # noqa: E402
from kvbench.learned.foresightkv_paper import ForesightKVJudgeScorer  # noqa: E402
from kvbench.future_contrib.extract import install_capture_hooks, SequenceCapture, capture_sequence_chunked  # noqa: E402

SINK_TOKENS = 4
RECENT_TOKENS = 64
OBS_WINDOW = 32
RKV_WINDOW = 8
# "how much observed history" family for the strength-gap figure: SnapKV's own
# sum-of-last-K-real-queries'-attention formula, evaluated at varying K. K=32
# is identical to the existing `snapkv` field (same formula, same window) so
# it is not stored again under a separate key.
RECENT_K_LIST = [1, 8, 16, 32, 64]
ORACLE_HORIZON = 256
MIN_HORIZON = 16
CONT_MAX_NEW = 300
CONT_MIN_NEW = ORACLE_HORIZON + 16

BUDGETS = [128, 256, 512, 1024]

TASK_TEMPLATES = {
    "qasper": (
        'You are given a scientific article and a question. Answer the question as '
        'concisely as you can, using a single phrase or sentence if possible. If the '
        'question cannot be answered based on the information in the article, write '
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", or '
        '"unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer '
        'the question based on the above article as concisely as you can, using a single '
        'phrase or sentence if possible. If the question cannot be answered based on the '
        'information in the article, write "unanswerable". If the question is a yes/no '
        'question, answer "yes", "no", or "unanswerable". Do not provide any explanation.'
        '\n\nQuestion: {input}\n\nAnswer:'
    ),
    # matches opencompass/configs/datasets/longbench/longbenchhotpotqa/*_gen_*.py exactly
    "hotpotqa": (
        'Answer the question based on the given passages. Only give me the answer and '
        'do not output any other words.\n\nThe following are given passages.\n{context}'
        '\n\nAnswer the question based on the given passages. Only give me the answer '
        'and do not output any other words.\n\nQuestion: {input}\nAnswer:'
    ),
    # matches opencompass/configs/datasets/longbench/longbenchgov_report/*_gen_*.py exactly
    # (no {input} placeholder -- gov_report has no question field, .format() just ignores
    # the unused `input` kwarg build_prompt_ids always passes)
    "gov_report": (
        'You are given a report by a government agency. Write a one-page summary of the '
        'report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.'
        '\n\nSummary:'
    ),
    # matches opencompass/configs/datasets/longbench/longbenchmultifieldqa_en/*_gen_*.py exactly
    "multifieldqa_en": (
        'Read the following text and answer briefly.\n\n{context}\n\nNow, answer the '
        'following question based on the above text, only give me the answer and do not '
        'output any other words.\n\nQuestion: {input}\nAnswer:'
    ),
}


def build_prompt_ids(tokenizer, context: str, question: str, max_seq_len: int, task: str = "qasper") -> torch.Tensor:
    text = TASK_TEMPLATES[task].format(context=context, input=question)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
    )
    enc = tokenizer(chat, return_tensors="pt", add_special_tokens=False, truncation=False)
    ids = enc["input_ids"]
    input_budget = max(1, int(max_seq_len) - CONT_MAX_NEW)
    if int(ids.shape[1]) > input_budget:
        half = input_budget // 2
        tail = input_budget - half
        ids = torch.cat([ids[:, :half], ids[:, -tail:]], dim=1)
    return ids


@torch.no_grad()
def generate_reference(model, prompt_ids: torch.Tensor, device) -> torch.Tensor:
    prompt_ids = prompt_ids.to(device)
    attn = torch.ones_like(prompt_ids)
    eos = model.config.eos_token_id
    pad = eos[0] if isinstance(eos, (list, tuple)) else eos
    prev = model.config._attn_implementation
    model.config._attn_implementation = "sdpa"
    try:
        out = model.generate(
            input_ids=prompt_ids, attention_mask=attn,
            max_new_tokens=CONT_MAX_NEW, min_new_tokens=min(CONT_MIN_NEW, CONT_MAX_NEW),
            do_sample=False, pad_token_id=pad,
        )
    finally:
        model.config._attn_implementation = prev
    return out


def rank_pct(x: torch.Tensor) -> torch.Tensor:
    """Scale-free normalization: each entry's rank among candidates, in [0, 1].
    Robust to R-KV's signed/unbounded scale and every other method's very
    different native units (attention-mass sums, value-norm-weighted sums,
    learned-judge logits)."""
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    order = x.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=x.device, dtype=torch.float32)
    return ranks / float(n - 1)


@dataclass
class DocResult:
    doc_id: str
    boundary: int
    cand_start: int
    cand_end: int
    prompt_ids: torch.Tensor  # [1, boundary] on CPU, for phase B reuse
    gt: torch.Tensor          # [n_cand] oracle future-attention share (sums to 1)
    h2o: torch.Tensor
    snapkv: torch.Tensor
    laprox: torch.Tensor
    lava: torch.Tensor
    rkv: torch.Tensor
    foresightkv: torch.Tensor
    recent: dict  # {k: torch.Tensor} for k in RECENT_K_LIST except OBS_WINDOW (=32, ==snapkv)


@torch.no_grad()
def compute_doc_scores(
    model, tokenizer, device, doc_id: str, context: str, question: str, max_seq_len: int,
    foresight_scorer: ForesightKVJudgeScorer, task: str = "qasper",
) -> DocResult | None:
    prompt_ids = build_prompt_ids(tokenizer, context, question, max_seq_len, task=task)
    boundary = int(prompt_ids.shape[1])
    cand_end = boundary - RECENT_TOKENS
    if cand_end - SINK_TOKENS < 32:
        return None  # document too short for a meaningful candidate range

    full_ids = generate_reference(model, prompt_ids, device)
    full_len = int(full_ids.shape[1])
    horizon_end = min(boundary + ORACLE_HORIZON, full_len)
    if horizon_end - boundary < MIN_HORIZON:
        return None

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    kv_repeat = num_heads // num_kv_heads
    scaling = head_dim ** -0.5
    cfg = BaselineConfig(policy="coverage", sink_tokens=SINK_TOKENS, recent_tokens=RECENT_TOKENS,
                          snapkv_obs_window=OBS_WINDOW)

    # ---- H2O partial (chunked, cumulative attention over [0, boundary-WIDE_WINDOW)) ----
    # WIDE_WINDOW = RECENT_TOKENS = max(RECENT_K_LIST): capturing this many real
    # trailing queries (instead of just OBS_WINDOW) additionally lets the
    # strength-gap "Recent-K" family (K in RECENT_K_LIST) be sliced straight out
    # of the SAME captured tensor for every K <= WIDE_WINDOW. h2o/snapkv/laprox/
    # lava/rkv below explicitly re-slice back down to their own original window
    # sizes (OBS_WINDOW=32 or RKV_WINDOW=8) so their values are UNCHANGED versus
    # capturing at width 32 directly -- only foresightkv legitimately sees more
    # context now (its own docstring says it accepts "whatever attention this
    # call observed", so a wider window is a valid variant, not a regression).
    WIDE_WINDOW = RECENT_TOKENS
    prev_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        past_prefix, _, h2o_partial = _h2o_chunked_prefill(
            model, prompt_ids[:, : boundary - WIDE_WINDOW].to(device), device,
        )

        # ---- observation-window forward: captures real Q/K/V for the last
        # WIDE_WINDOW real queries against the FULL [0, boundary) key/value cache,
        # both as raw attention weights (out.attentions) and as raw Q/K/V
        # (install_capture_hooks) for R-KV's own formula. ----
        cap = SequenceCapture()
        cleanup = install_capture_hooks(model, "llama", cap)
        obs_ids = prompt_ids[:, boundary - WIDE_WINDOW : boundary].to(device)
        attention_mask = torch.ones((1, boundary), dtype=torch.long, device=device)
        position_ids = torch.arange(boundary - WIDE_WINDOW, boundary, dtype=torch.long, device=device).view(1, -1)
        try:
            out = model(
                input_ids=obs_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_prefix, use_cache=True, output_attentions=True, return_dict=True,
            )
        finally:
            cleanup()
    finally:
        model.config._attn_implementation = prev_impl

    # ---- GT oracle: real future-attention over the SAME candidate range, from
    # a full teacher-forced pass over prompt+reference-continuation. Reuses
    # capture_sequence_chunked (the exact function OracleFutureScorer.
    # set_reference uses) rather than a hand-rolled chunk loop: it concatenates
    # each chunk's query along the sequence dim so cap_full.qkv ends up
    # ABSOLUTE-position-indexed over the WHOLE sequence (a naive per-chunk
    # capture only keeps the LAST call's queries, which are chunk-relative-
    # indexed -- confirmed as a real bug here: horizon_start/horizon_end are
    # absolute positions, so indexing a chunk-relative query tensor with them
    # silently slices out-of-range and returns an empty tensor). ----
    model.config._attn_implementation = "eager"
    try:
        cap_full = capture_sequence_chunked(model, full_ids.to(device), "llama", chunk_size=2048)
    finally:
        model.config._attn_implementation = prev_impl

    n_cand = cand_end - SINK_TOKENS
    cand_idx = torch.arange(SINK_TOKENS, cand_end, device=device)

    gt_layers, h2o_layers, snapkv_layers, laprox_layers, lava_layers, rkv_layers, fkv_layers = [], [], [], [], [], [], []
    recent_layers = {k: [] for k in RECENT_K_LIST if k != OBS_WINDOW}
    for layer_idx in range(num_layers):
        # GT
        q_full, k_full, _ = cap_full.qkv[layer_idx]  # already batch-squeezed: [heads/kv_heads, seq, hd]
        per_head_gt = []
        for h in range(num_kv_heads):
            per_head_gt.append(_oracle_head_distribution(
                q_full, k_full, cand_idx, boundary, horizon_end, h, kv_repeat, scaling,
            ))
        gt_layers.append(torch.stack(per_head_gt, dim=0).mean(dim=0))

        layer_attn_wide = out.attentions[layer_idx]  # [1, heads, WIDE_WINDOW, boundary]
        # observation-window capture already carries the FULL k/v the attention
        # module saw (past + new), i.e. [1, kv_heads, boundary, head_dim]
        _, k_win_full, v_win_full = cap.qkv[layer_idx]
        # re-slice down to the ORIGINAL OBS_WINDOW=32 for every method that was
        # defined against that width -- keeps h2o/snapkv/laprox/lava numerically
        # identical to capturing at width 32 directly (verified: summing this
        # window's own contribution below plus h2o_partial's now-shorter
        # [0, boundary-WIDE_WINDOW) accumulation still equals the full cumulative
        # sum over [0, boundary), same as before).
        layer_attn = layer_attn_wide[:, :, -OBS_WINDOW:, :]

        h2o_full = h2o_partial[layer_idx]  # [kv_heads, boundary-WIDE_WINDOW]
        h2o_from_window = query_to_kv_attention(layer_attn_wide, num_kv_heads).sum(dim=1)  # [kv_heads, boundary]
        h2o_layers.append((h2o_full[:, SINK_TOKENS:cand_end] + h2o_from_window[:, SINK_TOKENS:cand_end]).mean(dim=0))

        snap = snapkv_head_token_scores(layer_idx, layer_attn, v_win_full, cfg)
        snapkv_layers.append(snap[:, SINK_TOKENS:cand_end].mean(dim=0))

        lap = laprox_head_token_scores(model, layer_idx, layer_attn, v_win_full, cfg)
        laprox_layers.append(lap[:, SINK_TOKENS:cand_end].mean(dim=0))

        lav = lava_head_token_scores(layer_attn, v_win_full, OBS_WINDOW, cfg)
        lava_layers.append(lav[:, SINK_TOKENS:cand_end].mean(dim=0))

        q_win, k_win, _ = cap.qkv[layer_idx]
        rkv_score = rkv_final_score(
            q_win[0, :, -RKV_WINDOW:, :], k_win, RKV_WINDOW,
            mix_lambda=cfg.rkv_mix_lambda, kernel_size=cfg.rkv_kernel_size,
            retain_ratio=cfg.rkv_retain_ratio, retain_direction=cfg.rkv_retain_direction,
        )  # [kv_heads, boundary - RKV_WINDOW]
        rkv_layers.append(rkv_score[:, SINK_TOKENS:cand_end].mean(dim=0))

        # strength-gap "Recent-K" family: SAME snapkv formula (sum of last-K
        # real queries' attention, kernel-pooled), varying K only. K=OBS_WINDOW
        # is exactly the `snap` computed above, so it's skipped here.
        for k in recent_layers:
            attn_k = layer_attn_wide[:, :, -k:, :]
            rk = snapkv_head_token_scores(layer_idx, attn_k, v_win_full, cfg)
            recent_layers[k].append(rk[:, SINK_TOKENS:cand_end].mean(dim=0))

        positions = torch.arange(boundary, device=device).unsqueeze(0).expand(num_kv_heads, -1)
        # ForesightKVJudgeScorer.score_layer returns an EVICT-priority score
        # (higher = more evictable / less important, per its own docstring and
        # its official eval-time convention) -- negate to get an importance
        # score, matching dense_eviction_hf.py:975's `scores = -foresight_paper_scorer
        # .score_layer(...)` before that project's own top-k "keep highest" logic
        # runs on it. Missing this negation here silently made every
        # foresightkv-vs-GT overlap measurement pick out the LEAST important
        # candidates instead of the most important ones.
        fkv = -foresight_scorer.score_layer(k_win_full[0], v_win_full[0], positions, layer_idx, layer_attn_wide[0])
        fkv_layers.append(fkv[:, SINK_TOKENS:cand_end].mean(dim=0))

    cap_full.clear()
    cap.clear()

    return DocResult(
        doc_id=doc_id, boundary=boundary, cand_start=SINK_TOKENS, cand_end=cand_end,
        prompt_ids=prompt_ids.cpu(),
        gt=torch.stack(gt_layers, dim=0).mean(dim=0).cpu(),
        h2o=torch.stack(h2o_layers, dim=0).mean(dim=0).cpu(),
        snapkv=torch.stack(snapkv_layers, dim=0).mean(dim=0).cpu(),
        laprox=torch.stack(laprox_layers, dim=0).mean(dim=0).cpu(),
        lava=torch.stack(lava_layers, dim=0).mean(dim=0).cpu(),
        rkv=torch.stack(rkv_layers, dim=0).mean(dim=0).cpu(),
        foresightkv=torch.stack(fkv_layers, dim=0).mean(dim=0).cpu(),
        recent={k: torch.stack(v, dim=0).mean(dim=0).cpu() for k, v in recent_layers.items()},
    )
