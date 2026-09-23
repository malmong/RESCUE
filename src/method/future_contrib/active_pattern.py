"""Active-pattern history features for rfc objective "H".

Motivation (2026-08-24 oracle + Top-K diagnostic): legacy's Q/K
FutureQueryPredictor -- which compresses the recent-query window into a
single predicted future query via a learned MLP -- underperforms simply
using REAL recent attention as a proxy, at every layer measured, even after
matching training/eval horizon. Rather than replace recency with prediction,
this ENRICHES recency: instead of a single 32-token attention snapshot,
track each candidate's own observed activity pattern over a longer window --
recency (steps since last "active"), frequency (EMA of active-indicator),
and streak (current consecutive active run) -- and let a small model combine
these with Q/K compatibility, matching the objective already proven correct
by the oracle experiment (KL against real future attention).

"Active" at a given step = in the top ACTIVE_TOP_FRAC fraction of that
step's real attention mass (rank-based, robust to attention-scale
differences across layers/steps).

Two computation paths sharing the same "active" definition and decay
constant, but necessarily different data sources:
  - offline (training, compute_activity_stats_offline): replays a PAST
    window of real causal attention recomputed from full captured Q/K
    (src.method.future_contrib.extract.capture_sequence).
  - online (eval): incrementally updated per decode/prefill-observation step
    from LIVE attention HF actually produced, maintained as running state on
    DynamicEvictionState (gathered/appended alongside K/V, same lifecycle as
    h2o_scores -- see active_mask_from_attn/replay_activity, called directly
    from src/dense_eviction_hf.py).
"""
from __future__ import annotations

import torch

from src.utils.scoring import query_to_kv_attention

ACTIVE_TOP_FRAC = 0.2
ACTIVITY_DECAY = 0.9
ACTIVITY_WINDOW = 16  # past steps replayed for training / prefill initialization


def active_mask_from_attn(layer_attn: torch.Tensor, num_kv_heads: int, top_frac: float = ACTIVE_TOP_FRAC) -> torch.Tensor:
    """layer_attn: [num_heads or num_kv_heads, q_len, k_len] (real,
    causal-softmaxed attention). Returns bool [num_kv_heads, q_len, k_len]:
    True where that key was in the top `top_frac` fraction of attention mass
    for THAT SPECIFIC query step."""
    grouped = query_to_kv_attention(layer_attn, num_kv_heads)  # [kv_heads, q_len, k_len]
    k_len = grouped.shape[-1]
    k = max(1, int(round(top_frac * k_len)))
    threshold = grouped.topk(k, dim=-1).values[..., -1:]
    return grouped >= threshold


def replay_activity(
    active_mask: torch.Tensor,
    decay: float = ACTIVITY_DECAY,
    init_recency: torch.Tensor | None = None,
    init_freq: torch.Tensor | None = None,
    init_streak: torch.Tensor | None = None,
):
    """active_mask: bool [num_kv_heads, num_steps, num_candidates], steps in
    chronological order (oldest first). Replays step by step, returning
    (recency, freq, streak) AS OF THE LAST STEP, each [num_kv_heads,
    num_candidates]. init_* let this continue from prior running state (the
    eval-time incremental path calls this once per single new step, passing
    its own previous state in)."""
    num_kv_heads, num_steps, num_candidates = active_mask.shape
    device = active_mask.device
    recency = init_recency if init_recency is not None else torch.zeros(num_kv_heads, num_candidates, device=device)
    freq = init_freq if init_freq is not None else torch.zeros(num_kv_heads, num_candidates, device=device)
    streak = init_streak if init_streak is not None else torch.zeros(num_kv_heads, num_candidates, device=device)
    for tau in range(num_steps):
        act = active_mask[:, tau, :].float()
        active_bool = act.bool()
        recency = torch.where(active_bool, torch.zeros_like(recency), recency + 1)
        freq = decay * freq + (1 - decay) * act
        streak = torch.where(active_bool, streak + 1, torch.zeros_like(streak))
    return recency, freq, streak


def compute_activity_stats_offline(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    cand: torch.Tensor,
    window_start: int,
    window_end: int,
    num_kv_heads: int,
    kv_repeat: int,
    scaling: float,
    top_frac: float = ACTIVE_TOP_FRAC,
    decay: float = ACTIVITY_DECAY,
):
    """Training-time: replays REAL causal attention for queries in
    [window_start, window_end) (a PAST window, window_end <= t+1) onto
    `cand`, from full captured Q/K. Same per-head inner loop as
    src.method.future_contrib.train.future_attention_distribution, but keeps
    every step's own distribution un-aggregated (needed to threshold
    per-step, not just the horizon-averaged one). Returns (recency, freq,
    streak) as of window_end-1, each [num_kv_heads, len(cand)]."""
    device = q_full.device
    cand = cand.to(device)
    num_steps = window_end - window_start
    if num_steps <= 0:
        z = torch.zeros(num_kv_heads, cand.numel(), device=device)
        return z, z, z
    abs_q_pos = torch.arange(window_start, window_end, device=device).unsqueeze(1)
    key_pos_full = torch.arange(window_end, device=device).unsqueeze(0)
    causal_mask = key_pos_full > abs_q_pos
    active_mask = torch.zeros(num_kv_heads, num_steps, cand.numel(), dtype=torch.bool, device=device)
    for g in range(num_kv_heads):
        k_g = k_full[g, :window_end, :].float()
        acc = torch.zeros(num_steps, cand.numel(), device=device)
        for h_local in range(kv_repeat):
            h = g * kv_repeat + h_local
            q_h = q_full[h, window_start:window_end, :].float()
            scores = torch.matmul(q_h, k_g.transpose(0, 1)) * scaling
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            acc += attn.index_select(1, cand)
        dist = acc / kv_repeat  # [num_steps, num_candidates], per-step (NOT summed over steps)
        k_top = max(1, int(round(top_frac * cand.numel())))
        thresh = dist.topk(k_top, dim=-1).values[:, -1:]
        active_mask[g] = dist >= thresh
    return replay_activity(active_mask, decay=decay)
