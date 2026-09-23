"""R-KV (arXiv 2505.24133), implemented to match the OFFICIAL HuggingFace
backend code exactly (reference/official/R-KV/HuggingFace/rkv/{modeling,
compression/r1_kv,utils}.py), per explicit instruction: match the released
code's actual behavior/defaults, not the paper text (which describes a
periodic B_buffer-gated compression scheme -- the real HF code is simpler:
every forward call, if cache_len >= budget, compress back down to budget;
otherwise no-op. No separate periodic-buffer logic needed).

Score (per kv_head): Z_i = mix_lambda * attn_cache_i - (1-mix_lambda) * redundancy_i
  attn_cache: softmax(QK^T) of the last `window_size` real queries against
    candidates only (window excluded from the softmax denominator), GQA
    reduced by MAX across the query group, meaned over the window queries,
    maxpool(kernel_size) smoothed.
  redundancy: full pairwise cosine similarity among ALL cached keys
    (including the window), one link per row zeroed out per retain_direction
    (NOT a count -- exactly one exempted match, matching the shipped code
    literally, quirks included), column-meaned, softmax'd, then restricted
    to candidates.
Official defaults (R1KV.__init__): mix_lambda=0.07, window_size=8,
kernel_size=7, retain_ratio=0.1, retain_direction="last".

The redundancy term needs a full N x N similarity matrix, which is
infeasible to materialize at once for LongBench-scale prompts (tens of
thousands of candidates -> N^2 blows up memory). Computed in ROW CHUNKS
instead (same strategy as this project's own H2O _h2o_chunked_prefill):
mathematically IDENTICAL to the unchunked version, not an approximation --
each chunk only needs its own rows' full N columns to decide what to zero
and to accumulate the column-sum reduction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def rkv_redundancy_scores(
    key_states: torch.Tensor,
    retain_ratio: float = 0.1,
    retain_direction: str = "last",
    threshold: float = 0.5,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """key_states: [1, kv_heads, N, head_dim]. Returns [kv_heads, N] (NOT
    restricted to candidates yet -- matches official cal_similarity's full
    output before r1_kv.py's own [..., :-window_size] slice)."""
    _, kv_heads, n, head_dim = key_states.shape
    device = key_states.device
    k = key_states[0].float()  # [kv_heads, N, head_dim]
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    col_sum = torch.zeros(kv_heads, n, device=device, dtype=torch.float32)
    kk = max(1, int(n * retain_ratio))
    arange_n = torch.arange(n, device=device, dtype=torch.int32)
    # Cap the per-chunk intermediates instead of trusting a fixed row count:
    # sim_chunk (fp32) + indices (int32) are each kv_heads * c * n * 4 bytes, so
    # c must shrink as n grows or a long prompt OOMs at a chunk size that was
    # comfortable on a short one. 1.5 GiB per tensor keeps room for the dense
    # cache the selector holds alongside this.
    budget_elems = (1536 * 1024 * 1024) // 4
    max_c = max(1, budget_elems // max(1, kv_heads * n))
    chunk_size = max(1, min(int(chunk_size), int(max_c)))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c = end - start
        rows = k_norm[:, start:end, :]  # [kv_heads, c, head_dim]
        sim_chunk = torch.matmul(rows, k_norm.transpose(-1, -2))  # [kv_heads, c, N]
        # zero the diagonal (self-similarity) -- row (start+r) vs column (start+r)
        diag_rows = torch.arange(c, device=device)
        sim_chunk[:, diag_rows, start + diag_rows] = 0.0
        # arange as int32 and a SCALAR 0 in the where: the previous
        # `torch.zeros_like(idx_range)` materialised a second [kv_heads, c, n]
        # int64 tensor purely to hold zeros, and int64 doubled both of them.
        # At narrativeqa lengths (n ~ 3e4, c = 2048, kv_heads = 8) that pair was
        # ~7.8 GiB and OOM'd on top of the dense cache the fidelity selector
        # still holds. Values are unchanged: indices are non-negative and far
        # below 2**31, and `where(mask, idx, 0)` is what zeros_like encoded.
        idx_range = arange_n.view(1, 1, n).expand(kv_heads, c, n)
        mask = sim_chunk > threshold
        indices = torch.where(mask, idx_range, 0)
        # Matches official code literally (including its no-match-defaults-
        # to-index-0 quirk for "last"/"first" -- not "fixed" here, since the
        # instruction is exact parity with the shipped code, not a cleaner
        # reimplementation).
        if retain_direction == "last":
            retain = indices.amax(dim=-1)
        elif retain_direction == "first":
            retain = indices.amin(dim=-1)
        elif retain_direction == "last_percent":
            retain = torch.topk(indices, k=min(kk, n), dim=-1).values[..., 0]
        elif retain_direction == "first_percent":
            retain = torch.topk(indices, k=min(kk, n), dim=-1, largest=False).values[..., -1]
        else:
            raise ValueError(f"Unsupported retain_direction={retain_direction}")
        sim_chunk.scatter_(-1, retain.unsqueeze(-1).long(), 0.0)
        col_sum += sim_chunk.sum(dim=1)
    return torch.softmax(col_sum / float(n), dim=-1)


@torch.no_grad()
def rkv_attention_cache(
    query_window: torch.Tensor,
    key_states: torch.Tensor,
    kernel_size: int = 7,
) -> torch.Tensor:
    """query_window: [heads, window_size, head_dim] (FULL query-head
    resolution, NOT kv-head-reduced -- GQA reduction happens here via MAX,
    matching official compute_attention_scores(pooling="max")).
    key_states: [1, kv_heads, N, head_dim] -- attention computed against
    ALL cached keys (window included); caller restricts to candidates
    afterward, matching official r1_kv.py's own [..., :-window_size] slice
    done on the OUTPUT, not on key_states here.
    Returns [kv_heads, N]."""
    _, kv_heads, n, head_dim = key_states.shape
    q = query_window.float()  # [heads, window, head_dim]
    heads, window, _ = q.shape
    group = max(1, heads // kv_heads)
    k = key_states[0].float()  # [kv_heads, N, head_dim]
    q_grouped = q.view(kv_heads, group, window, head_dim)
    scores = torch.matmul(q_grouped, k.unsqueeze(1).transpose(-1, -2)) / (head_dim ** 0.5)  # [kv_heads, group, window, N]
    scores = scores.amax(dim=1)  # GQA: MAX over the query group -- [kv_heads, window, N]
    return scores


def rkv_final_score(
    query_window: torch.Tensor,
    key_states: torch.Tensor,
    window_size: int,
    mix_lambda: float = 0.07,
    kernel_size: int = 7,
    retain_ratio: float = 0.1,
    retain_direction: str = "last",
    threshold: float = 0.5,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Full R1KV.update_kv score, restricted to candidates (excludes the
    trailing window_size positions, which are always kept separately).
    Returns [kv_heads, N - window_size]."""
    _, kv_heads, n, _ = key_states.shape
    scores = rkv_attention_cache(query_window, key_states, kernel_size)  # [kv_heads, window, N]
    cand_n = n - window_size
    attn_weights_sum = torch.softmax(
        scores[:, :, :cand_n], dim=-1,
    ).mean(dim=1)  # [kv_heads, cand_n] -- softmax over candidates only (window excluded from denominator)
    attn_cache = F.max_pool1d(
        attn_weights_sum.unsqueeze(1), kernel_size=kernel_size, padding=kernel_size // 2, stride=1,
    ).squeeze(1)[:, :cand_n]
    redundancy = rkv_redundancy_scores(key_states, retain_ratio, retain_direction, threshold, chunk_size)
    redundancy = redundancy[:, :cand_n]
    return attn_cache * mix_lambda - redundancy * (1.0 - mix_lambda)
