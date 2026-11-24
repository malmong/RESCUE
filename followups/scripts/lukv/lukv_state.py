"""LU-KV's head-level budget allocation, as a ragged base inside this release.

A follow-up asked us to discuss LU-KV (ICML 2026).  Discussing it is cheap; the
question worth answering is whether it competes with RESCUE or composes with
it, and that needs a measurement.  The two act on different axes -- LU-KV
decides HOW MANY entries each (layer, kv_head) keeps and then delegates WHICH
ones to an existing rule ("Each head independently applies the heuristic
metric"; their experiments use SnapKV), while RESCUE changes which entries a
given budget retains.  So they can be stacked, and this file stacks them.

What is imported from upstream is exactly one artifact: the released
``*_avg_ratio.npy`` profile, a [99, n_layer, n_kv_head] table of per-head PRUNE
ratios indexed by global compression ratio.  Everything else -- the SnapKV
score, the protected window, the cache -- is this release's own, so the three
arms (uniform / LU-KV / LU-KV + RESCUE) differ in the allocation and nothing
else.

Two deviations from upstream, both deliberate and both stated in the reply:

  * Upstream lets its own table set the retained total
    (``total_keep_target = round(sum(ideal_keep_counts))``).  We rescale the
    table's shape to hit this release's budget exactly, so the LU-KV arm holds
    the same number of tokens as the uniform arm it is compared against.
    Without this the comparison is not at matched memory.
  * Upstream force-keeps sink and window by writing ``scores.max()`` into those
    slots.  We protect them explicitly instead, which is the same decision
    expressed as a constraint rather than as a score, and keeps the protected
    set identical to the uniform SnapKV arm's.

Single-shot only, like every ragged base here: run with
--evict-once-after-prefill (scripts/evaluate.py already does).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from rescue.eviction import RaggedKVCache, legacy_layers
from rescue.policies.scoring import (BaselineConfig, resolve_budget,
                                     snapkv_head_token_scores)

_DEFAULT_CURVE = ("/path/to/kv_cache/followups/scripts/lukv_upstream/LU-KV/evaluation/"
                  "curve_data/llama-3.1-8b/"
                  "snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy")

_CURVE_CACHE: dict[str, np.ndarray] = {}


def load_curve(path: str | None = None) -> np.ndarray:
    """[99, n_layer, n_kv_head] per-head PRUNE ratios, row i = global ratio (i+1)/100."""
    p = str(path or os.environ.get("LUKV_CURVE", _DEFAULT_CURVE))
    if p not in _CURVE_CACHE:
        if not Path(p).exists():
            raise FileNotFoundError(f"LU-KV budget curve not found: {p}")
        a = np.load(p)
        if a.ndim != 3 or a.shape[0] != 99:
            raise ValueError(f"expected a [99, L, H] curve, got {a.shape} from {p}")
        if not (0.0 <= a.min() and a.max() <= 1.0):
            raise ValueError(f"curve values are prune RATIOS; got range "
                             f"[{a.min():.4f}, {a.max():.4f}] from {p}")
        _CURVE_CACHE[p] = a
    return _CURVE_CACHE[p]


def allocate(curve: np.ndarray, seq_len: int, per_head_budget: int,
             num_layers: int, kv_heads: int, min_keep: int,
             total_budget: int) -> list[list[int]]:
    """Per-(layer, head) retained-token counts summing to exactly total_budget.

    The row is chosen by the compression ratio this budget implies on THIS
    document, which is how upstream indexes it (``target_idx =
    round(compression_ratio * 100) - 1``, clamped to [0, 98]).
    """
    if curve.shape[1] != num_layers or curve.shape[2] != kv_heads:
        raise ValueError(f"curve is [{curve.shape[1]}, {curve.shape[2]}] per row but the "
                         f"model has {num_layers} layers x {kv_heads} kv heads; "
                         "the profile must be regenerated for this model")
    ratio = 1.0 - (float(per_head_budget) / max(1, int(seq_len)))
    idx = int(round(ratio * 100)) - 1
    idx = max(0, min(98, idx))

    keep_rate = 1.0 - curve[idx]                      # [L, H]
    ideal = keep_rate.astype(np.float64) * float(seq_len)
    s = ideal.sum()
    if s <= 0:
        ideal = np.full_like(ideal, float(total_budget) / (num_layers * kv_heads))
    else:
        ideal = ideal * (float(total_budget) / s)     # match the uniform arm's total

    lo = float(min_keep)
    hi = float(seq_len)
    # Water-filling against both bounds: clamp, then redistribute the deficit
    # or surplus over the still-free cells, so the total is exact and no head
    # is handed fewer slots than its protected set.
    cur = np.clip(ideal, lo, hi)
    for _ in range(64):
        gap = float(total_budget) - cur.sum()
        if abs(gap) < 0.5:
            break
        free = (cur > lo + 1e-9) if gap < 0 else (cur < hi - 1e-9)
        if not free.any():
            break
        cur[free] += gap / float(free.sum())
        cur = np.clip(cur, lo, hi)

    base = np.floor(cur).astype(np.int64)
    rem = int(total_budget) - int(base.sum())
    if rem > 0:                                        # largest remainder, as upstream
        order = np.argsort(-(cur - base).flatten())
        room = (base < int(hi)).flatten()
        for j in order:
            if rem <= 0:
                break
            if room[j]:
                base.flat[j] += 1
                rem -= 1
    elif rem < 0:
        order = np.argsort((cur - base).flatten())
        room = (base > int(lo)).flatten()
        for j in order:
            if rem >= 0:
                break
            if room[j]:
                base.flat[j] -= 1
                rem += 1
    return base.tolist()


class LUKVState:
    """LU-KV head-budget allocation + SnapKV selection, optionally + RESCUE.

    Occupies the ragged-base slot: the generator hands ``self.rescue`` a
    RescueInjection for the --method rescue arm, exactly as it does for LAVa
    and R-KV, so the correction stacks on top with no new plumbing.
    """

    def __init__(self, cfg: BaselineConfig, model):
        if cfg.evict_during_decode:
            raise ValueError("lukv must be run with --evict-once-after-prefill "
                             "(decode-time re-pruning is not implemented)")
        self.cfg = cfg
        self.model = model
        self.prefill_retained = 0
        self.total_evicted = 0
        self.rescue = None
        self.curve = load_curve()
        self.alloc_log: list[list[int]] = []

    def initialize_after_prefill(self, past_key_values, attentions, seq_len: int):
        if not attentions:
            raise RuntimeError("lukv requires prefill output attentions. "
                               "Set model_kwargs.attn_implementation='eager'.")
        cache = RaggedKVCache.from_past(past_key_values, seq_len)
        layers = legacy_layers(past_key_values)
        num_layers = len(layers)
        if num_layers == 0:
            return cache
        window = max(1, int(self.cfg.snapkv_obs_window))
        sink = max(0, int(self.cfg.sink_tokens))

        layer_scores: list[torch.Tensor] = []
        for layer_idx, (key, value) in enumerate(layers):
            scores = snapkv_head_token_scores(
                layer_idx, attentions[layer_idx], value, self.cfg).to(key.device)
            if self.rescue is not None:
                cur_t = int(key.shape[2]) - 1
                pos = (torch.arange(int(key.shape[2]), device=key.device)
                       .view(1, -1).expand(int(key.shape[1]), -1))
                scores = self.rescue.combine(layer_idx, scores, key, value, pos, cur_t)
            layer_scores.append(scores)

        kv_heads = int(layers[0][1].shape[1])
        max_len = max((cache.layer_max_len(l) for l in range(len(cache))), default=0)
        per_head_budget = resolve_budget(max_len, self.cfg)
        min_keep = min(max_len, sink + window)
        total_budget = max(num_layers * kv_heads * min_keep,
                           min(cache.total_tokens(),
                               per_head_budget * num_layers * kv_heads))
        counts = allocate(self.curve, max_len, per_head_budget,
                          num_layers, kv_heads, min_keep, total_budget)
        self.alloc_log = counts

        neg = float("-inf")
        for layer_idx in range(num_layers):
            scores = layer_scores[layer_idx]
            kvh, cache_len = int(scores.shape[0]), int(scores.shape[1])
            prot_start = max(0, cache_len - window)
            n_sink = min(sink, cache_len)
            for h in range(kvh):
                keep = set(range(prot_start, cache_len)) | set(range(n_sink))
                k = int(counts[layer_idx][h]) if layer_idx < len(counts) else per_head_budget
                k = max(len(keep), min(cache_len, k))
                need = k - len(keep)
                if need > 0 and prot_start > n_sink:
                    cand = scores[h].detach().float().clone()
                    cand[prot_start:] = neg
                    cand[:n_sink] = neg
                    take = min(need, prot_start - n_sink)
                    if take > 0:
                        keep |= set(torch.topk(cand, k=take, largest=True)
                                    .indices.tolist())
                if not keep:
                    keep.add(int(torch.argmax(scores[h]).item()))
                key_t = cache.keys[layer_idx][h]
                slots = torch.tensor(sorted(keep), dtype=torch.long, device=key_t.device)
                self.total_evicted += cache.prune_head(layer_idx, h, slots)
        self.prefill_retained = cache.total_tokens()
        if os.environ.get("LUKV_DEBUG"):
            cnt = np.array(counts)
            print(f"[LUKV] N={max_len} B={per_head_budget} "
                  f"retained={self.prefill_retained} target={total_budget} "
                  f"ratio={self.prefill_retained/max(1,total_budget):.4f} "
                  f"head_counts min={cnt.min()} max={cnt.max()} mean={cnt.mean():.1f}",
                  flush=True)
        return cache

    def update_and_prune_after_decode(self, cache, ragged_attentions):
        return cache
