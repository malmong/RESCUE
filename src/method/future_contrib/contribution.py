"""Output-contribution ground truth. Spec ref: sections 11-12.

C_{l,i,tau} = || sum_h A_h(tau,i) * V_h(i) @ W_O^(h) ||_2

A is the *real* full-causal-softmax attention weight (normalized over the
whole row, not just the candidate subset) -- candidates are just the columns
we bother to read out afterward. This is why the mask below is built against
the true key range [0, horizon_end), not against candidate_idx.

Memory: the naive approach (materialize [num_heads, seq, seq] attention) is
the O(seq^2) wall this project has hit before (see ForesightKV/LookaheadKV
training scripts). This implementation avoids that AND avoids ever
materializing a [horizon_len, num_candidates, hidden_size] tensor (an early
version of this file did, and timed out at ~160h/backbone for the spec's
recommended scale). The identity used instead (exact, not an approximation):

    ||sum_h a_h g_h||^2 = sum_h sum_h' a_h a_h' <g_h, g_h'>

So per (layer, eviction step) we precompute the candidate-wise head x head
Gram matrix Gram[i,h,h'] = <g_h(i), g_h(i')> ONCE (cost ~ candidates *
num_heads^2 * hidden, via a single batched matmul), then every future step
tau only needs a quadratic form a^T Gram a over num_heads (32) dims --
the hidden_size factor never touches the per-tau computation. The per-head
python loop that remains is only for the (unavoidable) causal softmax itself,
which is now a cheap [horizon_len, horizon_end] op with no candidate/hidden
accumulation inside it.
"""
from __future__ import annotations

import torch


def compute_output_contribution(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    o_proj_weight: torch.Tensor,
    candidate_idx: torch.Tensor,
    horizon_start: int,
    horizon_len: int,
    kv_repeat: int,
    scaling: float,
) -> torch.Tensor:
    """
    query: [num_heads, seq_len, head_dim], already rotary/qk-norm applied (real attention inputs).
    key/value: [num_kv_heads, seq_len, head_dim], NOT GQA-repeated.
    o_proj_weight: [hidden_size, num_heads * head_dim] (nn.Linear.weight layout).
    candidate_idx: 1-D long tensor of absolute cache positions, each < horizon_start.
    Returns C: [num_candidates, actual_horizon_len] (actual_horizon_len <= horizon_len,
    clipped at sequence end per spec section 13's H_t normalization).
    """
    device = query.device
    num_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[0]
    hidden_size = o_proj_weight.shape[0]
    num_candidates = int(candidate_idx.shape[0])

    horizon_end = min(horizon_start + horizon_len, seq_len)
    actual_horizon_len = horizon_end - horizon_start
    if actual_horizon_len <= 0 or num_candidates == 0:
        return query.new_zeros((num_candidates, max(0, actual_horizon_len)))

    cand_pos = candidate_idx.to(device=device, dtype=torch.long)
    q_h = query[:, horizon_start:horizon_end, :].float()  # [num_heads, H, d]
    wo = o_proj_weight.float()  # [hidden, num_heads*head_dim]

    # --- g_h(i) = V_kv(h)(i) @ W_O_block_h^T, batched over all heads at once ---
    v_cand = value[:, cand_pos, :].float()  # [num_kv_heads, num_candidates, head_dim]
    v_cand_rep = v_cand.repeat_interleave(kv_repeat, dim=0)  # [num_heads, num_candidates, head_dim]
    wo_blocks = wo.view(hidden_size, num_heads, head_dim).permute(1, 2, 0)  # [num_heads, head_dim, hidden]
    g_all = torch.bmm(v_cand_rep, wo_blocks)  # [num_heads, num_candidates, hidden]

    # --- Gram[i, h, h'] = g_h(i) . g_h'(i), batched over candidates ---
    g_ci = g_all.permute(1, 0, 2)  # [num_candidates, num_heads, hidden]
    gram = torch.bmm(g_ci, g_ci.transpose(1, 2))  # [num_candidates, num_heads, num_heads]

    # --- causal attention weights A[h, tau, i] (softmax needs the FULL row, not just candidates) ---
    abs_q_pos = torch.arange(horizon_start, horizon_end, device=device).unsqueeze(1)  # [H,1]
    key_pos_full = torch.arange(horizon_end, device=device).unsqueeze(0)  # [1,horizon_end]
    causal_mask = key_pos_full > abs_q_pos  # [H, horizon_end], True = disallowed

    attn_cand = torch.empty(num_heads, actual_horizon_len, num_candidates, device=device, dtype=torch.float32)
    for g in range(num_kv_heads):
        k_g = key[g, :horizon_end, :].float()  # [horizon_end, d]
        for h_local in range(kv_repeat):
            h = g * kv_repeat + h_local
            scores = torch.matmul(q_h[h], k_g.transpose(0, 1)) * scaling  # [H, horizon_end]
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)  # [H, horizon_end]
            attn_cand[h] = attn.index_select(1, cand_pos)  # [H, num_candidates]

    # --- quadratic form: C^2[i, tau] = a_i(tau)^T Gram[i] a_i(tau), a_i(tau) in R^num_heads ---
    a_ci = attn_cand.permute(2, 1, 0)  # [num_candidates, H, num_heads]
    tmp = torch.bmm(a_ci, gram)  # [num_candidates, H, num_heads]
    c2 = (tmp * a_ci).sum(dim=-1).clamp_min(0.0)  # [num_candidates, H] -- clamp guards fp roundoff near 0
    return c2.sqrt().contiguous()  # [num_candidates, H]


def compute_recent_score(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    o_proj_weight: torch.Tensor,
    candidate_idx: torch.Tensor,
    current_step: int,
    recent_window: int,
    kv_repeat: int,
    scaling: float,
    decay: float | None = None,
) -> torch.Tensor:
    """Spec section 25: S_i^recent(t) from the SAME physical quantity (output
    contribution) as the future target, over the observed window
    [current_step - recent_window + 1, current_step].
    decay=None -> simple moving average; decay in (0,1) -> exponential weights
    gamma_r^(t-tau).
    """
    obs_start = max(0, current_step - recent_window + 1)
    obs_len = current_step - obs_start + 1
    if obs_len <= 0:
        return query.new_zeros((int(candidate_idx.shape[0]),))
    # Reuse the same contribution kernel with the observation window playing
    # the role of "horizon": C[i, tau] for tau in [obs_start, current_step].
    c = compute_output_contribution(
        query, key, value, o_proj_weight, candidate_idx,
        horizon_start=obs_start, horizon_len=obs_len,
        kv_repeat=kv_repeat, scaling=scaling,
    )  # [num_candidates, obs_len]
    if c.shape[-1] == 0:
        return query.new_zeros((int(candidate_idx.shape[0]),))
    if decay is None:
        return c.mean(dim=-1)
    taus = torch.arange(obs_start, obs_start + c.shape[-1], device=c.device)
    weights = decay ** (current_step - taus).float()
    weights = weights / weights.sum().clamp_min(1e-8)
    return (c * weights.unsqueeze(0)).sum(dim=-1)


def recent_contribution_from_attn(
    attn: torch.Tensor,
    value: torch.Tensor,
    o_proj_weight: torch.Tensor,
    kv_repeat: int,
) -> torch.Tensor:
    """S_recent for deployment (spec section 25), computed from attention HF
    already produced (output_attentions=True) instead of recomputing Q@K --
    at real inference time we never cache historical Q, only K/V, so
    compute_recent_score's approach (which needs Q) isn't available here.
    Same exact Gram-matrix identity as compute_output_contribution, just fed
    a precomputed attention tensor instead of deriving it from query/key.

    attn: [num_heads, q_len, k_len] real (already causally-softmaxed) attention
    probabilities for whatever queries this call observed (a prefill
    observation window, or a single decode-step query).
    value: [num_kv_heads, k_len, head_dim] (NOT GQA-repeated).
    o_proj_weight: [hidden_size, num_heads*head_dim].
    Returns S_recent: [k_len] -- one score per currently-cached token, meaned
    over the observed query window (tau range), matching
    S_i^recent(t) = (1/R_t) sum_tau C_{i,tau}.
    """
    num_heads, q_len, k_len = attn.shape
    num_kv_heads = value.shape[0]
    hidden_size = o_proj_weight.shape[0]
    if q_len == 0 or k_len == 0:
        return attn.new_zeros((k_len,))

    wo = o_proj_weight.float()
    v_rep = value.float().repeat_interleave(kv_repeat, dim=0)  # [num_heads, k_len, head_dim]
    wo_blocks = wo.view(hidden_size, num_heads, wo.shape[-1] // num_heads).permute(1, 2, 0)  # [num_heads, head_dim, hidden]
    g_all = torch.bmm(v_rep, wo_blocks)  # [num_heads, k_len, hidden]

    g_ci = g_all.permute(1, 0, 2)  # [k_len, num_heads, hidden]
    gram = torch.bmm(g_ci, g_ci.transpose(1, 2))  # [k_len, num_heads, num_heads]

    a_ci = attn.float().permute(2, 1, 0)  # [k_len, q_len, num_heads]
    tmp = torch.bmm(a_ci, gram)  # [k_len, q_len, num_heads]
    c2 = (tmp * a_ci).sum(dim=-1).clamp_min(0.0)  # [k_len, q_len]
    return c2.sqrt().mean(dim=-1)  # [k_len]


def _naive_reference_contribution(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    o_proj_weight: torch.Tensor,
    candidate_idx: torch.Tensor,
    horizon_start: int,
    horizon_len: int,
    kv_repeat: int,
    scaling: float,
) -> torch.Tensor:
    """Maximally literal, unbatched transcription of spec section 11, used only
    to cross-check compute_output_contribution in unit tests."""
    num_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[0]
    horizon_end = min(horizon_start + horizon_len, seq_len)
    cand = candidate_idx.tolist()
    out = torch.zeros(len(cand), horizon_end - horizon_start)
    for tau_i, tau in enumerate(range(horizon_start, horizon_end)):
        for i_idx, i in enumerate(cand):
            acc = torch.zeros(o_proj_weight.shape[0])
            for h in range(num_heads):
                kv_h = h // kv_repeat
                keys_upto = key[kv_h, : tau + 1, :].float()
                q = query[h, tau, :].float()
                scores = (keys_upto @ q) * scaling
                attn = torch.softmax(scores, dim=0)
                a_hi = attn[i]
                v_i = value[kv_h, i, :].float()
                wo_block = o_proj_weight[:, h * head_dim:(h + 1) * head_dim].float()
                acc = acc + a_hi * (wo_block @ v_i)
            out[i_idx, tau_i] = acc.norm()
    return out
