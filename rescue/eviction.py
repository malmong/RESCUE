from __future__ import annotations

from dataclasses import asdict
import importlib
import math
import os
import time
from types import SimpleNamespace
from typing import List

import torch

from rescue.policies.scoring import (
    BaselineConfig,
    h2o_decode_scores,
    h2o_prefill_scores,
    laprox_head_token_scores,
    laprox_indices,
    laprox_layer_token_scores,
    laprox_vwo_norm,
    lava_head_token_scores,
    legacy_layers,
    query_to_kv_attention,
    recent_absolute_positions,
    resolve_budget,
    select_common_topk_indices,
    snapkv_head_token_scores,
    snapkv_prefill_indices,
    start_recent_indices,
)
from rescue.policies.rkv import rkv_final_score
from rescue import LearnedScorer
from rescue.policies.foresightkv import ForesightPaperScorer
from rescue.policies.foresightkv_official import ForesightKVJudgeScorer
from rescue.policies.kvp import KVPPaperScorer
from rescue.oracle_future import OracleFutureScorer
from rescue.future_contrib.active_pattern import active_mask_from_attn, replay_activity
from rescue.future_contrib.contribution import recent_contribution_from_attn
from rescue.future_contrib.extract import SequenceCapture, install_capture_hooks
from rescue.future_contrib.query_state import QueryRingBuffer


def _call_with_query_capture(model, fn):
    """Runs fn() (a single self.model(...) forward call) while live-capturing
    each layer's real (rotary-applied) query tensor via the same monkeypatch
    rescue.future_contrib.extract.capture_sequence uses for training -- so
    rfc's eval-time recent-query state is built from bit-identical inputs to
    what FutureQueryPredictor was trained against. Returns (result,
    {layer_idx: query_tensor[num_heads, seq_len_this_call, head_dim]})."""
    capture = SequenceCapture()
    cleanup = install_capture_hooks(model, model.config.model_type, capture)
    try:
        result = fn()
    finally:
        cleanup()
    # capture.qkv[layer_idx] is (query, key, value), each [batch=1, num_heads,
    # seq_len_this_call, head_dim] -- install_capture_hooks captures pre-squeeze
    # (capture_sequence's own batch-dim squeeze happens in ITS wrapper, which
    # we bypass here since we only need query, not a full SequenceCapture).
    queries = {layer_idx: qkv[0][0] for layer_idx, qkv in capture.qkv.items()}
    return result, queries


def _rank_gate_combine(s_recent: torch.Tensor, s_future: torch.Tensor, lam: float) -> torch.Tensor:
    """Priority 7 -- predictor x LaProx confidence gating (rfc_combine_mode=
    "rank_gate"). Every S_total formulation tried so far is ADDITIVE
    (S_recent + lambda*S_future), on whatever raw scale each half happens to
    live on -- and every additive combination has made scores WORSE than
    S_recent alone, even lambda=0's exact LaProx match (33.71) collapsing to
    31-32 the moment lambda>0 mixes in ANY predictor, regardless of which
    predictor. That pattern implicates the additive combination itself, not
    predictor quality. This instead converts each half to a per-kv-head
    PERCENTILE RANK in [0,1] among this step's candidates (scale-invariant
    by construction, sidestepping the exact scale-mismatch that plagues
    every additive mix) and combines them MULTIPLICATIVELY: the predictor
    acts as a gate that scales recency up or down, rather than an
    independently-added term that can dominate/distort S_recent at the
    wrong scale. lam=0 still reduces to pure S_recent ranking (same
    ordering as S_recent alone, satisfying the same lambda=0 sanity
    property every other combine mode has)."""
    s_recent = s_recent.float()
    s_future = s_future.float()
    if s_recent.shape[0] == 1 and s_future.shape[0] > 1:
        s_recent = s_recent.expand(s_future.shape[0], -1)
    n = s_recent.shape[-1]
    if n <= 1:
        return s_recent
    recent_rank = s_recent.argsort(dim=-1).argsort(dim=-1).float() / (n - 1)
    future_rank = s_future.argsort(dim=-1).argsort(dim=-1).float() / (n - 1)
    return recent_rank * (1.0 + lam * future_rank)


def _rank_additive_combine(s_recent: torch.Tensor, s_future: torch.Tensor, lam: float) -> torch.Tensor:
    """rfc_combine_mode="rank_additive". Same motivation as _rank_gate_combine
    (S_recent's raw laprox magnitude and S_future's raw softmax-mixture
    magnitude live on very different scales -- measured ~4-7x apart on a real
    qasper document -- so an ADDITIVE mix of the raw values makes lambda do
    double duty as both a scale-correction factor and a genuine
    recent-vs-future preference weight, conflating the two), but keeps the
    ADDITIVE structure (S_recent + lambda*S_future) instead of switching to a
    multiplicative gate: each side is first converted to its own per-kv-head
    PERCENTILE RANK in [0,1] among this step's candidates (scale-invariant),
    and lambda multiplies the already-ranked future term, matching the
    combine_mode="additive" formula but on comparable normalized scales."""
    s_recent = s_recent.float()
    s_future = s_future.float()
    if s_recent.shape[0] == 1 and s_future.shape[0] > 1:
        s_recent = s_recent.expand(s_future.shape[0], -1)
    n = s_recent.shape[-1]
    if n <= 1:
        return s_recent
    recent_rank = s_recent.argsort(dim=-1).argsort(dim=-1).float() / (n - 1)
    future_rank = s_future.argsort(dim=-1).argsort(dim=-1).float() / (n - 1)
    return recent_rank + lam * future_rank


def _share_additive_combine(s_recent: torch.Tensor, s_future: torch.Tensor, lam: float) -> torch.Tensor:
    """rfc_combine_mode="share_additive". _rank_additive_combine's percentile-
    rank normalization fixes the S_recent/S_future scale mismatch but breaks
    rfc_allocation="global_layer"'s cross-layer budget pooling: that pooling
    (_prune_rfc_global_layer's norm_scores = layer_mean / layer_mean.sum())
    identifies which LAYERS deserve more of the global budget by how SKEWED/
    concentrated a layer's raw scores are (one standout candidate should win
    budget away from layers with only mediocre candidates) -- but rank is BY
    CONSTRUCTION uniformly distributed regardless of the original values'
    skew, so rank-transforming erases exactly the signal that pooling reads
    (confirmed: rank_additive/rank_gate score far BELOW even lambda=0, which
    should be impossible since lambda=0 must reduce to plain S_recent
    ranking). This instead normalizes each side by its OWN SUM over this
    step's candidates (both become a share-of-total distribution, summing to
    1) -- a scale-only correction that leaves each side's internal shape/
    skew completely intact, so it does NOT fight global_layer's pooling the
    way rank does, while still fixing the original raw-scale mismatch this
    combine mode was motivated by."""
    s_recent = s_recent.float().clamp_min(0.0)
    s_future = s_future.float().clamp_min(0.0)
    if s_recent.shape[0] == 1 and s_future.shape[0] > 1:
        s_recent = s_recent.expand(s_future.shape[0], -1)
    recent_share = s_recent / s_recent.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    future_share = s_future / s_future.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return recent_share + lam * future_share


def _kslot_combine(s_recent: torch.Tensor, s_future: torch.Tensor,
                   b_rem: int, k: int) -> torch.Tensor:
    """rfc_combine_mode="kslot". Reserve the base policy's own picks instead of
    re-ranking against them.

    Every additive mode here produces one score and takes top-B of it, so the
    correction competes with the base policy for all B slots and can evict
    entries the base ranked correctly. Measured, that is exactly what happens on
    the strongest base policy: applied unconditionally the correction drops R-KV's
    oracle coverage by 20.8 points while recovering only 0.7% of what R-KV missed.

    This mode instead gives the base policy the first (B - k) slots outright and
    lets the correction fill only the remaining k, chosen from candidates the base
    did NOT already take. The result is expressed as a synthetic score whose top-B
    is precisely that union, so nothing downstream has to change: reserved entries
    get the highest band, the correction's k picks the next, everything else the
    lowest, and ordering within each band is preserved so ties break as before.
    """
    n = int(s_recent.shape[-1])
    b_rem = max(1, min(int(b_rem), n))
    k = max(0, min(int(k), b_rem))
    keep_base = b_rem - k
    r = s_recent.float()
    f = s_future.float()
    # rank in [0,1) within each row, so the three bands can be stacked by integer offset
    def _pct(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort(dim=-1)
        out = torch.empty_like(x)
        idx = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).expand_as(x)
        out.scatter_(-1, order, idx)
        return out / max(1, x.shape[-1])
    rr, rf = _pct(r), _pct(f)
    if keep_base <= 0:
        return (rf + 1.0).contiguous()
    base_top = r.topk(keep_base, dim=-1).indices
    reserved = torch.zeros_like(r, dtype=torch.bool).scatter_(-1, base_top, True)
    # band 2: reserved (base's own picks)   band 1: correction's k picks   band 0: rest
    total = rr.clone()                       # band 0 baseline, keeps base ordering
    total = torch.where(reserved, rr + 2.0, total)
    if k > 0:
        f_masked = rf.masked_fill(reserved, -1.0)
        corr_top = f_masked.topk(k, dim=-1).indices
        chosen = torch.zeros_like(r, dtype=torch.bool).scatter_(-1, corr_top, True) & (~reserved)
        total = torch.where(chosen, rf + 1.0, total)
    return total.contiguous()


def _scale_additive_combine(s_recent: torch.Tensor, s_future: torch.Tensor, lam: float) -> torch.Tensor:
    """rfc_combine_mode="scale_additive". Fixes the defect share_additive still
    has: it divides s_recent by a PER-ROW (per-kv-head) sum, so every head's
    scores are forced to the same total before _prune_rfc_global_layer averages
    them over heads. LaProx's own path does the opposite order -- it averages
    the RAW per-head scores (scoring.py laprox_layer_token_scores ends in
    head_scores.mean(dim=0)) and only then normalizes once -- and
    mean_h(x_h/sum(x_h)) != mean_h(x_h)/sum(mean_h(x_h)) unless all heads carry
    equal mass. Per-head normalization therefore equalizes head influence,
    inflating heads with almost no attention mass up to the level of dominant
    ones. That reweighting is applied at EVERY lambda, lambda=0 included, which
    is why lambda=0 measured BELOW pure LaProx on 4 of 5 tasks (qasper -0.94,
    multifieldqa_en -1.15, gov_report -0.59, lcc -0.45) even though the
    correction is fully off there -- the attainable floor sat below baseline,
    so no lambda rule could reach parity, let alone a gain.

    This leaves s_recent completely alone and scales only the correction, by a
    single SCALAR (summed over heads as well as candidates, so relative head
    mass survives on both sides):

        S = s_recent + lambda * (sum(s_recent)/sum(s_future)) * s_future

    lambda=0 is then exactly s_recent, i.e. bit-identical to LaProx, which is
    what --rfc-recent-style laprox's help has always claimed. The raw-scale
    mismatch that motivated share_additive is still corrected, because the
    scalar puts s_future on s_recent's scale before lambda is applied."""
    s_recent = s_recent.float().clamp_min(0.0)
    s_future = s_future.float().clamp_min(0.0)
    if s_recent.shape[0] == 1 and s_future.shape[0] > 1:
        s_recent = s_recent.expand(s_future.shape[0], -1)
    if not torch.is_tensor(lam) and float(lam) == 0.0:
        return s_recent          # exact LaProx, no float noise from *0
    scale = s_recent.sum() / s_future.sum().clamp_min(1e-12)
    # Divide by (1+lambda) so this layer's TOTAL score mass is sum(s_recent)
    # whatever lambda is. rfc_allocation="global_layer" pools budget across
    # layers by comparing their summed scores, so without this a layer running
    # a larger lambda carries (1+lambda)x the mass of one running a smaller one
    # and wins budget purely from that scale difference -- not from deserving
    # it. That is why per-LAYER lambda collapsed before (trec 54.00 -> 45.50):
    # a scale artifact, not the adaptation itself. For a lambda that is uniform
    # across layers this is a no-op (every layer is divided by the same
    # constant, and within a layer a constant divisor cannot reorder anything),
    # so it changes no existing result while making per-layer lambda safe.
    return (s_recent + lam * scale * s_future) / (1.0 + lam)


_MEMLOG = bool(os.environ.get("RESCUE_MEMLOG"))


def _memlog(tag: str, seq_len: int | None = None) -> None:
    """Env-gated stage marker for the long-prompt OOMs. Prints LIVE allocated
    bytes, not reserved: the failing message on narrativeqa reported 92.6 GiB
    genuinely allocated, so the question is which stage still holds tensors,
    which reserved-memory figures cannot answer."""
    if not _MEMLOG:
        return
    a = torch.cuda.memory_allocated() / 1024 ** 3
    r = torch.cuda.reserved_memory() / 1024 ** 3 if hasattr(torch.cuda, "reserved_memory") \
        else torch.cuda.memory_reserved() / 1024 ** 3
    extra = f" seq={seq_len}" if seq_len is not None else ""
    print(f"[MEM] {tag:<28} alloc={a:6.2f}GiB reserved={r:6.2f}GiB{extra}", flush=True)
    if os.environ.get("RESCUE_MEMLOG_TENSORS"):
        # Which LIVE cuda tensors hold the memory, grouped by shape+dtype. The
        # allocated total says memory is leaking but not what is leaking, and
        # the per-layer growth (64 MiB/layer) matches several candidates, so
        # this names the object instead of inferring it.
        import gc as _gc
        from collections import Counter as _C
        agg = _C()
        for o in _gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    agg[(tuple(o.shape), str(o.dtype))] += o.numel() * o.element_size()
            except Exception:
                continue
        for (shape, dt), nb in agg.most_common(6):
            print(f"[MEMT]   {nb/1024**3:7.3f}GiB  {dt:<14} {shape}", flush=True)


def _h2o_chunked_prefill(model, input_ids: torch.Tensor, device, chunk_size: int = 512):
    """H2O's score (cumulative attention mass received by each KV token
    across the WHOLE prompt) needs the full attention matrix by definition --
    unlike SnapKV's windowed-observation trick, there's no short query window
    that gives the same answer. A single forward call with
    output_attentions=True over the whole prompt materializes a
    [num_heads, seq_len, seq_len] softmax tensor (OOM on long LongBench
    documents: confirmed crashing on all 3 backbones, e.g. "tried to
    allocate 7.75 GiB" on an ~8k-token document).

    This instead processes the prompt in fixed-size query chunks against a
    growing KV cache, accumulating the exact same sum via the distributive
    property of summation (sum over ALL query positions == sum over each
    chunk's query positions, added up) -- peak attention-tensor memory
    becomes O(chunk_size * seq_len) instead of O(seq_len^2), producing an
    IDENTICAL result to h2o_prefill_scores' single-shot computation
    (verified against it on short sequences in
    scratchpad/verify_h2o_chunked_prefill.py).

    Returns (past_key_values, last_position_logits, h2o_scores), where
    h2o_scores is list[num_layers] of [kv_heads, seq_len] tensors -- same
    shape/semantics as h2o_prefill_scores' own return value.
    """
    seq_len = int(input_ids.shape[1])
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    past = None
    accum: list[torch.Tensor | None] = [None] * num_layers
    last_logits = None
    # output_attentions=True makes HF collect EVERY layer's raw (query-head,
    # not kv-head-grouped) attention tensor into one tuple before this loop
    # body ever sees it -- so a fixed chunk_size that's safe early (cache
    # still short) OOMs once the cache has grown into the tens of thousands
    # of tokens: num_layers * num_heads * chunk_size * current_cache_len * 4
    # bytes (fp32 softmax) all resident simultaneously (confirmed: 32 layers *
    # 32 heads * 512-chunk * ~51k-token gov_report cache ~= 108GB, on a
    # 95GB GPU). Shrink the chunk adaptively so that product stays bounded
    # regardless of how long the document is.
    attn_mem_budget_bytes = 20 * 1024**3
    start = 0
    while start < seq_len:
        max_chunk = max(1, attn_mem_budget_bytes // (num_layers * num_heads * 4 * max(1, start + 1)))
        end = min(start + min(chunk_size, max_chunk), seq_len)
        chunk_ids = input_ids[:, start:end]
        attention_mask = torch.ones((1, end), dtype=torch.long, device=device)
        position_ids = torch.arange(start, end, dtype=torch.long, device=device).view(1, -1)
        out = model(
            input_ids=chunk_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        past = out.past_key_values
        last_logits = out.logits
        for layer_idx, layer_attn in enumerate(out.attentions):
            grouped = query_to_kv_attention(layer_attn, num_kv_heads)  # [kv_heads, chunk_len, end]
            chunk_sum = grouped.sum(dim=1).to(device)  # [kv_heads, end]
            prev = accum[layer_idx]
            if prev is None:
                accum[layer_idx] = chunk_sum
            else:
                pad = chunk_sum.shape[1] - prev.shape[1]
                if pad > 0:
                    prev = torch.cat([prev, torch.zeros(num_kv_heads, pad, device=device, dtype=prev.dtype)], dim=1)
                accum[layer_idx] = prev + chunk_sum
        start = end
    return past, last_logits, accum


def _slice_past_key_values(past_key_values, keep_positions: list[int]):
    if past_key_values is None:
        return None
    idx_device = None
    layers = legacy_layers(past_key_values)
    if layers:
        for key, _ in layers:
            if key is not None:
                idx_device = key.device
                break
    idx = torch.tensor(keep_positions, dtype=torch.long, device=idx_device)
    if isinstance(past_key_values, tuple):
        return tuple((k.index_select(2, idx), v.index_select(2, idx)) for k, v in past_key_values)
    if isinstance(past_key_values, list):
        return [(k.index_select(2, idx), v.index_select(2, idx)) for k, v in past_key_values]
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for layer_i in range(len(key_cache)):
            key_cache[layer_i] = key_cache[layer_i].index_select(2, idx.to(key_cache[layer_i].device))
            value_cache[layer_i] = value_cache[layer_i].index_select(2, idx.to(value_cache[layer_i].device))
        return past_key_values
    raise TypeError(f"Unsupported past_key_values type for pruning: {type(past_key_values)!r}")


def _gather_layer_cache(tensor: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    # tensor: [batch, kv_heads, seq, dim], keep: [kv_heads, new_seq]
    if tensor is None:
        return tensor
    keep = keep.to(tensor.device)
    if keep.ndim == 1:
        keep = keep.view(1, -1).expand(tensor.shape[1], -1)
    index = keep.view(1, keep.shape[0], keep.shape[1], 1).expand(tensor.shape[0], -1, -1, tensor.shape[-1])
    return tensor.gather(dim=2, index=index)


def _set_layer_cache(past_key_values, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
    if isinstance(past_key_values, tuple):
        layers = list(past_key_values)
        layers[layer_idx] = (key, value)
        return tuple(layers)
    if isinstance(past_key_values, list):
        past_key_values[layer_idx] = (key, value)
        return past_key_values
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        key_cache[layer_idx] = key
        value_cache[layer_idx] = value
        return past_key_values
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        layers[layer_idx].keys = key
        layers[layer_idx].values = value
        return past_key_values
    raise TypeError(f"Unsupported past_key_values type for layer update: {type(past_key_values)!r}")


def _append_layer_cache(past_key_values, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor):
    if hasattr(past_key_values, "append_layer"):
        return past_key_values.append_layer(layer_idx, key_new, value_new)
    key_old, value_old = legacy_layers(past_key_values)[layer_idx]
    key = torch.cat([key_old, key_new], dim=2)
    value = torch.cat([value_old, value_new], dim=2)
    return _set_layer_cache(past_key_values, layer_idx, key, value)


def _cache_len(past_key_values) -> int:
    if hasattr(past_key_values, "first_layer_max_len"):
        return int(past_key_values.first_layer_max_len())
    layers = legacy_layers(past_key_values)
    if not layers or layers[0][0] is None:
        return 0
    return int(layers[0][0].shape[2])


def _cache_lengths(past_key_values) -> list[int]:
    if hasattr(past_key_values, "lengths_by_layer"):
        return past_key_values.lengths_by_layer()
    return [int(key.shape[2]) for key, _ in legacy_layers(past_key_values) if key is not None]


def _cache_tokens_total(past_key_values) -> int:
    if hasattr(past_key_values, "total_tokens"):
        return int(past_key_values.total_tokens())
    return sum(_cache_lengths(past_key_values))


def _cache_head_lengths(past_key_values) -> list[list[int]]:
    if hasattr(past_key_values, "lengths_by_layer_head"):
        return past_key_values.lengths_by_layer_head()
    out: list[list[int]] = []
    for key, _ in legacy_layers(past_key_values):
        if key is None:
            continue
        out.append([int(key.shape[2]) for _ in range(int(key.shape[1]))])
    return out


def _cache_kv_slots_total(past_key_values) -> int:
    return sum(sum(layer) for layer in _cache_head_lengths(past_key_values))


def _cache_bytes(past_key_values) -> int:
    if hasattr(past_key_values, "bytes"):
        return int(past_key_values.bytes())
    total = 0
    for key, value in legacy_layers(past_key_values):
        if key is not None:
            total += int(key.numel() * key.element_size())
        if value is not None:
            total += int(value.numel() * value.element_size())
    return total


class RaggedKVCache:
    """Physical KV cache with independent sequence length per layer/KV head."""

    def __init__(self, keys: list[list[torch.Tensor]], values: list[list[torch.Tensor]], positions: list[list[torch.Tensor]]):
        self.keys = keys
        self.values = values
        self.positions = positions

    @classmethod
    def from_past(cls, past_key_values, seq_len: int):
        keys: list[list[torch.Tensor]] = []
        values: list[list[torch.Tensor]] = []
        positions: list[list[torch.Tensor]] = []
        for key, value in legacy_layers(past_key_values):
            layer_keys = []
            layer_values = []
            layer_positions = []
            kv_heads = int(key.shape[1])
            for h in range(kv_heads):
                layer_keys.append(key[:, h : h + 1, :seq_len, :].contiguous())
                layer_values.append(value[:, h : h + 1, :seq_len, :].contiguous())
                layer_positions.append(torch.arange(seq_len, device=key.device, dtype=torch.long))
            keys.append(layer_keys)
            values.append(layer_values)
            positions.append(layer_positions)
        return cls(keys, values, positions)

    def __len__(self) -> int:
        return len(self.keys)

    def layer_head_count(self, layer_idx: int) -> int:
        return len(self.keys[layer_idx])

    def first_layer_max_len(self) -> int:
        if not self.keys:
            return 0
        return max((int(k.shape[2]) for k in self.keys[0]), default=0)

    def lengths_by_layer_head(self) -> list[list[int]]:
        return [[int(k.shape[2]) for k in layer] for layer in self.keys]

    def lengths_by_layer(self) -> list[int]:
        return [sum(int(k.shape[2]) for k in layer) for layer in self.keys]

    def total_tokens(self) -> int:
        return sum(sum(int(k.shape[2]) for k in layer) for layer in self.keys)

    def bytes(self) -> int:
        total = 0
        for layer_keys, layer_values in zip(self.keys, self.values):
            for key, value in zip(layer_keys, layer_values):
                total += int(key.numel() * key.element_size())
                total += int(value.numel() * value.element_size())
        return total

    def layer_max_len(self, layer_idx: int) -> int:
        return max((int(k.shape[2]) for k in self.keys[layer_idx]), default=0)

    def append_layer(self, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor, absolute_pos: int | None = None):
        kv_heads = int(key_new.shape[1])
        for h in range(kv_heads):
            self.keys[layer_idx][h] = torch.cat([self.keys[layer_idx][h], key_new[:, h : h + 1, :, :]], dim=2)
            self.values[layer_idx][h] = torch.cat([self.values[layer_idx][h], value_new[:, h : h + 1, :, :]], dim=2)
            if absolute_pos is not None:
                pos = torch.tensor([int(absolute_pos)], device=key_new.device, dtype=torch.long)
                self.positions[layer_idx][h] = torch.cat([self.positions[layer_idx][h].to(key_new.device), pos], dim=0)
        return self

    def prune_head(self, layer_idx: int, kv_head: int, keep: torch.Tensor) -> int:
        key = self.keys[layer_idx][kv_head]
        value = self.values[layer_idx][kv_head]
        keep = keep.to(key.device).long()
        before = int(key.shape[2])
        self.keys[layer_idx][kv_head] = key.index_select(2, keep).contiguous()
        self.values[layer_idx][kv_head] = value.index_select(2, keep).contiguous()
        self.positions[layer_idx][kv_head] = self.positions[layer_idx][kv_head].to(key.device).index_select(0, keep).contiguous()
        return max(0, before - int(keep.numel()))


class DynamicEvictionState:
    def __init__(self, cfg: BaselineConfig, model, learned_scorer: LearnedScorer | None = None):
        self.cfg = cfg
        self.model = model
        self.learned_scorer = learned_scorer
        self.kvp_paper_scorer = None
        self.foresight_paper_scorer = None
        self.rfc_scorer = None
        # RescueScorer, or OracleFutureScorer for the headroom ceiling.
        self.unified_rfc_scorer = None
        self.external_future_scorer = None
        self.external_future_kind = None
        self.layer_head_gain = None
        # o_proj.weight is a static model parameter -- converting it to fp32 fresh
        # on every _learned_keep call (32 layers x every prefill/decode step) was
        # measured to dominate rfc's per-step cost (46.5ms/call average, ~672
        # calls for just prefill+20 decode steps); cache the conversion per layer.
        self._rfc_o_proj_fp32: dict[int, torch.Tensor] = {}
        self.positions: list[torch.Tensor] = []
        self.h2o_scores: list[torch.Tensor] = []
        # rfc_recent_style="h2o" only: per-layer cache size frozen right after
        # prefill's MLP+H2O-corrected global_layer eviction -- decode-time
        # maintenance then uses PURE H2O (no MLP re-invocation, which was
        # measured to badly hurt qasper: 23.0 vs 30.53 for the same checkpoint/
        # lambda, once vs every-decode-step) to hold each layer at this frozen
        # size, respecting global_layer's uneven per-layer prefill allocation
        # instead of H2O's own uniform-per-layer resolve_budget.
        self._h2o_decode_budget: list[int] | None = None
        # z_cache[layer_idx]: [cache_len, proj_dim] -- the frozen-projection feature
        # z_i cached ONCE per token at insertion time (spec section 4; real inference
        # can't recompute a past token's hidden state later, unlike training's
        # one-shot full-attention pass). Only populated for policy=="rfc". Not
        # per-kv-head: rfc's score is layer-level and shared across kv_heads (spec
        # section 2), so every head's eviction decision keeps the same slots and a
        # single reference index (head 0) is enough to gather/append it.
        self.z_cache: list[torch.Tensor] = []
        # q_cache[layer_idx]: a fixed-size ring buffer of the last
        # rfc_scorer.recent_window real query vectors (grouped to kv-heads),
        # feeding FutureQueryPredictor. Unlike z_cache/positions, this is NOT
        # indexed by cache slot and never participates in _apply_layer_keep's
        # gather -- queries aren't cached tokens, they're transient decision-
        # time state, so memory here is O(layers * recent_window * kv_heads *
        # head_dim), independent of sequence length. Only populated for
        # policy=="rfc".
        self.q_cache: list[QueryRingBuffer] = []
        # activity_recency/freq/streak[layer_idx]: [kv_heads, cache_len] running
        # state for rfc_objective=="H" (rescue.future_contrib.active_pattern) --
        # same gather/append lifecycle as h2o_scores, just 3 parallel tensors
        # instead of 1. Only populated for policy=="rfc" with objective "H".
        self.activity_recency: list[torch.Tensor] = []
        self.activity_freq: list[torch.Tensor] = []
        self.activity_streak: list[torch.Tensor] = []
        self.prefill_retained = 0
        self.total_evicted = 0
        # rfc_objective="RESCUE" only: absolute positions of chat-template
        # special/control tokens (<|start_header_id|>, <|end_header_id|>,
        # <|eot_id|>, ...) in the prompt -- found via direct inspection
        # (rescue_eviction_inspect.py) to sometimes get evicted by the
        # trace_chat-trained correction, corrupting the model's turn/role
        # structure and triggering repetition collapse. Treated as always-
        # protected, like sink/recent tokens, never scored for eviction.
        self.special_positions: set[int] = set()

    def _populate_q_cache(self, queries: dict[int, torch.Tensor], num_kv_heads: int) -> None:
        scorer = self.rfc_scorer or self.unified_rfc_scorer
        if scorer is None:
            return
        recent_window = scorer.recent_window
        for layer_idx, q in queries.items():
            while len(self.q_cache) <= layer_idx:
                self.q_cache.append(QueryRingBuffer(recent_window))
            self.q_cache[layer_idx].push(q, num_kv_heads)

    def _init_positions(self, past_key_values, seq_len: int):
        self.positions = []
        for key, _ in legacy_layers(past_key_values):
            kv_heads = int(key.shape[1])
            pos = torch.arange(seq_len, device=key.device, dtype=torch.long).view(1, -1).expand(kv_heads, -1).clone()
            self.positions.append(pos)

    def _apply_layer_keep(self, past_key_values, layer_idx: int, keep: torch.Tensor):
        layers = legacy_layers(past_key_values)
        key, value = layers[layer_idx]
        before = int(key.shape[2])
        if os.environ.get("RESCUE_KEEP_LOG"):
            # Which ABSOLUTE positions this layer retains. Two configurations
            # that implement the same policy must retain the same set; if they
            # diverge only here and there, the difference is numerical (a
            # different prefill kernel producing slightly different K/V) rather
            # than a difference in the eviction logic.
            pos = self.positions[layer_idx] if layer_idx < len(self.positions) else None
            if pos is not None:
                kept = pos.gather(dim=1, index=keep.to(pos.device))[0].tolist()
                print(f"[KEEP] layer={layer_idx} n={len(kept)} "
                      f"hash={hash(tuple(kept)) & 0xffffffff:08x}", flush=True)
        key_new = _gather_layer_cache(key, keep)
        value_new = _gather_layer_cache(value, keep)
        past_key_values = _set_layer_cache(past_key_values, layer_idx, key_new, value_new)
        pos = self.positions[layer_idx].to(keep.device)
        self.positions[layer_idx] = pos.gather(dim=1, index=keep)
        if self.h2o_scores:
            score = self.h2o_scores[layer_idx].to(keep.device)
            self.h2o_scores[layer_idx] = score.gather(dim=1, index=keep)
        for activity_list in (self.activity_recency, self.activity_freq, self.activity_streak):
            if activity_list and layer_idx < len(activity_list) and activity_list[layer_idx] is not None:
                activity_list[layer_idx] = activity_list[layer_idx].to(keep.device).gather(dim=1, index=keep)
        if self.z_cache and layer_idx < len(self.z_cache) and self.z_cache[layer_idx] is not None:
            z = self.z_cache[layer_idx].to(keep.device)
            if int(z.shape[0]) == before:
                self.z_cache[layer_idx] = z.index_select(0, keep[0])
            else:
                # Fell out of sync with the actual cache length -- e.g. rfc's
                # global_layer allocation decodes via RaggedLlamaLikeDecoder,
                # which doesn't produce hidden_states, so append_generated_position
                # never grows z_cache during decode while the real K/V cache
                # keeps growing/pruning. z_cache is only read by the OLD
                # rfc_scorer "learned" impact mode (unused by every objective
                # this project currently evaluates); drop it rather than crash.
                self.z_cache[layer_idx] = None
        self.total_evicted += max(0, before - int(key_new.shape[2]))
        return past_key_values

    def _populate_z_cache_from_hidden_states(self, hidden_states) -> None:
        """hidden_states: HF's output_hidden_states tuple, hidden_states[l] is the
        residual-stream INPUT to layer l (embedding output for l=0). Projects the
        WHOLE available sequence per layer and (re)builds self.z_cache from
        scratch -- called once after any forward that returned hidden_states for
        every currently-cached position (a full prefill covers all of them; a
        single decode step only covers the newest token, handled instead by
        append_generated_position)."""
        if self.rfc_scorer is None or hidden_states is None:
            return
        # HF's output_hidden_states tuple has num_layers+1 entries: index l is the
        # INPUT to layer l for l in [0, num_layers), and the last entry is the
        # final layer's output (input to nothing we evict) -- exclude it.
        num_layers = len(hidden_states) - 1
        self.z_cache = []
        for layer_idx in range(num_layers):
            h = hidden_states[layer_idx][0]  # [seq_len, hidden_size]
            self.z_cache.append(self.rfc_scorer.project(h))

    def initialize_after_prefill(self, past_key_values, attentions, hidden_states=None,
                                 precomputed_h2o_scores=None, probe_attentions=None):
        policy = self.cfg.normalized_policy()
        seq_len = _cache_len(past_key_values)
        self._init_positions(past_key_values, seq_len)
        # Per-DOCUMENT lambda state has to be cleared here, at the one hook that
        # runs once per document. Without this the warmup accumulator survives
        # across documents, so the value frozen on document 0 is reused for the
        # whole evaluation -- "doc scope" silently degenerating into "whichever
        # lambda the first document happened to pick".
        self._gate_overlaps, self._gate_lam = [], None
        self._adaptive_masses, self._adaptive_lam = [], None
        if policy == "h2o" or (policy == "rfc" and self.cfg.rfc_recent_style == "h2o"):
            if precomputed_h2o_scores is not None:
                # _h2o_chunked_prefill already computed these incrementally
                # (see its docstring) -- no full attention tensor exists to
                # fall back to here, unlike the old single-shot path.
                # A COPY of the list, not the list. Pruning replaces each layer's
                # entry with its gathered 128-token version, and the fidelity
                # selector builds this state once per candidate lambda from the
                # same precomputed scores -- sharing the list let the first
                # trial shrink it, so the second saw a full cache beside
                # 128-token scores (measured: s_recent 128 vs s_future 4957).
                # The tensors are still shared; only the list is private.
                self.h2o_scores = list(precomputed_h2o_scores)
            elif not attentions:
                raise RuntimeError("H2O requires output attentions. Set model_kwargs.attn_implementation='eager'.")
            else:
                self.h2o_scores = h2o_prefill_scores(attentions, past_key_values)
        if policy == "rfc":
            self._populate_z_cache_from_hidden_states(hidden_states)
            if self.cfg.rfc_objective == "H" and attentions:
                self.activity_recency, self.activity_freq, self.activity_streak = [], [], []
                for layer_idx, layer_attn in enumerate(attentions):
                    key_l, _ = legacy_layers(past_key_values)[layer_idx]
                    kv_heads = int(key_l.shape[1])
                    if layer_attn is None:
                        z = torch.zeros(kv_heads, seq_len, device=key_l.device)
                        self.activity_recency.append(z)
                        self.activity_freq.append(z.clone())
                        self.activity_streak.append(z.clone())
                        continue
                    # obs-window prefill gives attention from the last
                    # snapkv_obs_window individual queries onto the whole
                    # prompt -- exactly the per-step history replay_activity
                    # needs (see rescue.future_contrib.active_pattern).
                    active_mask = active_mask_from_attn(layer_attn, kv_heads)
                    recency, freq, streak = replay_activity(active_mask)
                    self.activity_recency.append(recency)
                    self.activity_freq.append(freq)
                    self.activity_streak.append(streak)
        return self.prune_after_prefill(past_key_values, attentions, hidden_states=hidden_states,
                                        probe_attentions=probe_attentions)

    def prune_after_prefill(self, past_key_values, attentions, hidden_states=None,
                            probe_attentions=None):
        if os.environ.get("RESCUE_KEEP_DUMP"):
            self._keepdump_doc = int(getattr(self, "_keepdump_doc", -1)) + 1
        policy = self.cfg.normalized_policy()
        if policy == "full":
            self.prefill_retained = _cache_len(past_key_values)
            return past_key_values
        if policy == "laprox" and self.cfg.laprox_allocation == "global_layer":
            if not attentions:
                raise RuntimeError("LaProx requires output attentions. Set model_kwargs.attn_implementation='eager'.")
            past_key_values = self._prune_laprox_global_layer(past_key_values, attentions)
            self.prefill_retained = _cache_tokens_total(past_key_values)
            return past_key_values
        if policy == "rfc" and self.cfg.rfc_allocation == "global_layer":
            past_key_values = self._prune_rfc_global_layer(past_key_values, attentions, hidden_states)
            self.prefill_retained = _cache_tokens_total(past_key_values)
            if self.cfg.rfc_recent_style == "h2o":
                self._h2o_decode_budget = [int(key.shape[2]) for key, _ in legacy_layers(past_key_values)]
            return past_key_values
        layers = legacy_layers(past_key_values)
        for layer_idx, (key, value) in enumerate(layers):
            if _MEMLOG and layer_idx % 4 == 0:
                _memlog(f"prune layer {layer_idx:02d}", int(key.shape[2]))
            budget = resolve_budget(int(key.shape[2]), self.cfg)
            if int(key.shape[2]) <= budget:
                continue
            if policy == "streamingllm":
                keep = start_recent_indices(self.positions[layer_idx], budget, self.cfg.sink_tokens)
            elif policy == "h2o":
                keep = self._h2o_keep(layer_idx, budget)
            elif policy == "snapkv":
                if not attentions:
                    raise RuntimeError("SnapKV requires prefill output attentions. Set model_kwargs.attn_implementation='eager'.")
                _pa = probe_attentions[layer_idx] if probe_attentions else None
                keep = snapkv_prefill_indices(attentions[layer_idx], key, budget, self.cfg,
                                              probe_attn=_pa,
                                              probe_weight=float(os.environ.get("RESCUE_PROBE_WEIGHT", "1")))
                if os.environ.get("RESCUE_KEEP_DUMP") and keep is not None:
                    # The same hook runs on the base policy, so "does RESCUE
                    # retain more of the oracle set than the base does" can be
                    # read directly. Scoring the correction's choices on their
                    # own is not enough: the base policy is the reference the
                    # number has to be against.
                    import json as _json
                    _kd = keep.detach().cpu()
                    with open(os.environ["RESCUE_KEEP_DUMP"], "a") as _fh:
                        _fh.write(_json.dumps({
                            "doc": int(getattr(self, "_keepdump_doc", 0)),
                            "n": int(key.shape[2]), "layer": int(layer_idx),
                            "keep": [sorted(map(int, _kd[h].tolist())) for h in range(_kd.shape[0])]
                                    if _kd.dim() == 2 else [sorted(map(int, _kd.tolist()))]}) + "\n")
            elif policy == "laprox":
                if not attentions:
                    raise RuntimeError("LaProx requires output attentions. Set model_kwargs.attn_implementation='eager'.")
                keep = laprox_indices(self.model, layer_idx, attentions[layer_idx], value, self.positions[layer_idx], budget, self.cfg)
            elif policy in {"kvp", "foresightkv", "lookaheadkv", "rfc"}:
                keep = self._learned_keep(layer_idx, key, value, budget, attentions, hidden_states)
                if os.environ.get("RESCUE_KEEP_DUMP") and keep is not None:
                    # Which cache positions this arm actually retained, per
                    # (doc, layer, kv_head). Two arms run on the same documents
                    # in the same order under greedy decoding, so dumping both
                    # and intersecting answers the question the cached-feature
                    # recall cannot: does the LEARNED scorer keep what the
                    # ORACLE keeps at inference, not just on training features.
                    import json as _json
                    _kd = keep.detach().cpu()
                    # Document fingerprint. A prune-call counter does not line
                    # up across arms, because they skip documents on different
                    # conditions (measured: 1682 lines against 576). The cache
                    # length is the same for the same document, so the two dumps
                    # are joined on that instead.
                    _rec = {"doc": int(getattr(self, "_keepdump_doc", 0)),
                            "n": int(key.shape[2]),
                            "layer": int(layer_idx),
                            "keep": [sorted(map(int, _kd[h].tolist()))
                                     for h in range(_kd.shape[0])] if _kd.dim() == 2
                                    else [sorted(map(int, _kd.tolist()))]}
                    with open(os.environ["RESCUE_KEEP_DUMP"], "a") as _fh:
                        _fh.write(_json.dumps(_rec) + "\n")
            else:
                keep = None
            if keep is not None:
                past_key_values = self._apply_layer_keep(past_key_values, layer_idx, keep)
        self.prefill_retained = _cache_len(past_key_values)
        return past_key_values

    def append_generated_position(self, past_key_values, absolute_pos: int, hidden_states=None):
        layers = legacy_layers(past_key_values)
        policy = self.cfg.normalized_policy()
        for layer_idx, (key, _) in enumerate(layers):
            kv_heads = int(key.shape[1])
            current = torch.full((kv_heads, 1), int(absolute_pos), device=key.device, dtype=torch.long)
            if layer_idx >= len(self.positions):
                self.positions.append(current)
            else:
                self.positions[layer_idx] = torch.cat([self.positions[layer_idx].to(key.device), current], dim=1)
            if self.h2o_scores:
                zeros = torch.zeros((kv_heads, 1), dtype=torch.float32, device=key.device)
                self.h2o_scores[layer_idx] = torch.cat([self.h2o_scores[layer_idx].to(key.device), zeros], dim=1)
            for activity_list in (self.activity_recency, self.activity_freq, self.activity_streak):
                if activity_list and layer_idx < len(activity_list) and activity_list[layer_idx] is not None:
                    zeros = torch.zeros((kv_heads, 1), dtype=torch.float32, device=key.device)
                    activity_list[layer_idx] = torch.cat([activity_list[layer_idx].to(key.device), zeros], dim=1)
            if policy == "rfc" and self.rfc_scorer is not None and hidden_states is not None:
                # hidden_states here is the single new token's per-layer input,
                # from this decode step's forward -- cache its projection now
                # since it will never be recomputable later (spec section 4).
                h_new = hidden_states[layer_idx][0]  # [1, hidden_size]
                z_new = self.rfc_scorer.project(h_new)  # [1, proj_dim]
                if layer_idx >= len(self.z_cache):
                    self.z_cache.append(z_new.to(key.device))
                else:
                    self.z_cache[layer_idx] = torch.cat([self.z_cache[layer_idx].to(key.device), z_new.to(key.device)], dim=0)

    def update_and_prune_after_decode(self, past_key_values, attentions, hidden_states=None):
        policy = self.cfg.normalized_policy()
        if policy in {"full", "snapkv"}:
            return past_key_values
        layers = legacy_layers(past_key_values)
        is_rfc_h2o_base = policy == "rfc" and self.cfg.rfc_recent_style == "h2o"
        if (policy == "h2o" or is_rfc_h2o_base) and not attentions:
            raise RuntimeError("H2O requires decode output attentions. Set model_kwargs.attn_implementation='eager'.")
        if (policy == "h2o" or is_rfc_h2o_base) and attentions:
            for layer_idx, layer_attn in enumerate(attentions):
                key, _ = layers[layer_idx]
                inc = h2o_decode_scores(layer_attn, int(key.shape[1]), key.device)
                self.h2o_scores[layer_idx][:, : inc.shape[1]] += inc[:, : self.h2o_scores[layer_idx].shape[1]]
        if is_rfc_h2o_base and self._h2o_decode_budget is not None:
            # Prefill already ran MLP+H2O once (global_layer eviction); from
            # here on, maintain each layer's FROZEN post-prefill size using
            # PURE H2O (no RESCUE MLP re-invocation -- see _h2o_decode_budget's
            # docstring for why not). Mirrors the plain policy=="h2o" per-layer
            # branch below (self._h2o_keep), just with a frozen per-layer
            # ceiling instead of resolve_budget's uniform one.
            for layer_idx, (key, value) in enumerate(layers):
                budget = self._h2o_decode_budget[layer_idx]
                if int(key.shape[2]) <= budget:
                    continue
                keep = self._h2o_keep(layer_idx, budget)
                past_key_values = self._apply_layer_keep(past_key_values, layer_idx, keep)
            return past_key_values
        if policy == "rfc" and self.cfg.rfc_objective == "H" and attentions and self.activity_recency:
            for layer_idx, layer_attn in enumerate(attentions):
                if layer_attn is None or layer_idx >= len(self.activity_recency):
                    continue
                key, _ = layers[layer_idx]
                kv_heads = int(key.shape[1])
                active_mask = active_mask_from_attn(layer_attn, kv_heads).to(key.device)  # [kv_heads, 1, cache_len]
                n = min(active_mask.shape[-1], self.activity_recency[layer_idx].shape[-1])
                recency, freq, streak = replay_activity(
                    active_mask[:, :, :n],
                    init_recency=self.activity_recency[layer_idx][:, :n].to(key.device),
                    init_freq=self.activity_freq[layer_idx][:, :n].to(key.device),
                    init_streak=self.activity_streak[layer_idx][:, :n].to(key.device),
                )
                self.activity_recency[layer_idx][:, :n] = recency
                self.activity_freq[layer_idx][:, :n] = freq
                self.activity_streak[layer_idx][:, :n] = streak
        if policy == "laprox" and self.cfg.laprox_allocation == "global_layer":
            if not attentions:
                raise RuntimeError("LaProx requires decode output attentions. Set model_kwargs.attn_implementation='eager'.")
            return self._prune_laprox_global_layer(past_key_values, attentions)
        if policy == "rfc" and self.cfg.rfc_allocation == "global_layer":
            return self._prune_rfc_global_layer(past_key_values, attentions, hidden_states)
        for layer_idx, (key, value) in enumerate(legacy_layers(past_key_values)):
            budget = resolve_budget(int(key.shape[2]), self.cfg)
            if int(key.shape[2]) <= budget:
                continue
            if policy == "streamingllm":
                keep = start_recent_indices(self.positions[layer_idx], budget, self.cfg.sink_tokens)
            elif policy == "h2o":
                keep = self._h2o_keep(layer_idx, budget)
            elif policy == "laprox":
                if not attentions:
                    raise RuntimeError("LaProx requires decode output attentions. Set model_kwargs.attn_implementation='eager'.")
                keep = laprox_indices(self.model, layer_idx, attentions[layer_idx], value, self.positions[layer_idx], budget, self.cfg)
            elif policy in {"kvp", "foresightkv", "lookaheadkv", "rfc"}:
                keep = self._learned_keep(layer_idx, key, value, budget, attentions, hidden_states)
            else:
                continue
            past_key_values = self._apply_layer_keep(past_key_values, layer_idx, keep)
        return past_key_values

    def _prune_laprox_global_layer(self, past_key_values, attentions):
        layers = legacy_layers(past_key_values)
        if not layers:
            return past_key_values
        current_total = _cache_tokens_total(past_key_values)
        if current_total <= 0:
            return past_key_values
        per_layer_ref = max(int(key.shape[2]) for key, _ in layers if key is not None)
        per_layer_budget = resolve_budget(per_layer_ref, self.cfg)
        total_budget = max(len(layers), min(current_total, per_layer_budget * len(layers)))
        candidates: list[tuple[float, int, int]] = []
        protected_by_layer: list[set[int]] = []
        layer_scores: list[torch.Tensor] = []
        for layer_idx, (key, value) in enumerate(layers):
            scores = laprox_layer_token_scores(self.model, layer_idx, attentions[layer_idx], value, self.cfg).to(key.device)
            denom = scores.sum().clamp_min(1e-12)
            norm_scores = scores / denom
            layer_scores.append(norm_scores)
            protected = recent_absolute_positions(self.positions[layer_idx], self.cfg.snapkv_obs_window)
            # Same chat-template control-token protection _prune_rfc_global_layer
            # applies. It used to live only in that path, which made rfc_lambda=0
            # differ from this baseline by exactly that much (qasper +0.02 but
            # multifieldqa_en -0.72) -- a difference that has nothing to do with
            # the correction under test. Protecting the control tokens is an
            # orthogonal improvement, so it belongs on BOTH sides of the
            # comparison; with it here, rfc_lambda=0 reduces to this function.
            if self.special_positions and getattr(self.cfg, "protect_special", True):
                protected = protected | self.special_positions
            protected_by_layer.append(protected)
            ref_positions = self.positions[layer_idx][0]
            for slot in range(int(key.shape[2])):
                abs_pos = int(ref_positions[slot].item())
                if abs_pos in protected:
                    continue
                candidates.append((float(norm_scores[slot].item()), layer_idx, slot))
        keep_slots = [set() for _ in layers]
        for layer_idx, protected in enumerate(protected_by_layer):
            ref_positions = self.positions[layer_idx][0]
            for slot in range(int(ref_positions.numel())):
                if int(ref_positions[slot].item()) in protected:
                    keep_slots[layer_idx].add(slot)
        protected_total = sum(len(x) for x in keep_slots)
        remaining = max(0, total_budget - protected_total)
        if remaining:
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, layer_idx, slot in candidates[:remaining]:
                keep_slots[layer_idx].add(slot)
        # Keep every layer non-empty so HuggingFace attention/cache code remains rectangular per layer.
        for layer_idx, slots in enumerate(keep_slots):
            if not slots:
                best = int(torch.argmax(layer_scores[layer_idx]).item())
                slots.add(best)
        for layer_idx, slots in enumerate(keep_slots):
            key, _ = legacy_layers(past_key_values)[layer_idx]
            slot_tensor = torch.tensor(sorted(slots), dtype=torch.long, device=key.device)
            keep = slot_tensor.view(1, -1).expand(key.shape[1], -1)
            past_key_values = self._apply_layer_keep(past_key_values, layer_idx, keep)
        return past_key_values

    def _prune_rfc_global_layer(self, past_key_values, attentions, hidden_states):
        """rfc's counterpart to _prune_laprox_global_layer -- matches LaProx's
        OWN default allocation exactly (this is what actually produced its
        33.71 qasper score, not the per-head/per-layer-fixed-budget path):
        within each layer, kv-heads share ONE common retained-position set
        (S_total averaged across heads); across layers, the *budget itself*
        is pooled (each layer's normalized scores compete in one global
        ranking), so different layers can retain different token counts for
        a fixed model-wide total. Reuses _rfc_score (identical scoring math
        to the per-head path) and _apply_layer_keep (no new cache machinery
        needed -- unlike LaProx's OTHER "global_head" mode, which pools
        across kv-heads too and genuinely needs RaggedKVCache, this stays
        rectangular within every layer)."""
        layers = legacy_layers(past_key_values)
        if not layers:
            return past_key_values
        current_total = _cache_tokens_total(past_key_values)
        if current_total <= 0:
            return past_key_values
        per_layer_ref = max(int(key.shape[2]) for key, _ in layers if key is not None)
        per_layer_budget = resolve_budget(per_layer_ref, self.cfg)
        total_budget = max(len(layers), min(current_total, per_layer_budget * len(layers)))
        candidates: list[tuple[float, int, int]] = []
        protected_by_layer: list[set[int]] = []
        layer_scores: list[torch.Tensor] = []
        use_floor = self.cfg.rfc_objective == "RESCUE" and float(self.cfg.rfc_rescue_floor_frac) > 0.0
        floor_candidates: list[tuple[float, int, int]] = []
        for layer_idx, (key, value) in enumerate(layers):
            scores = self._rfc_score(layer_idx, key, value, attentions, hidden_states)
            if scores is None:
                scores = torch.zeros(int(key.shape[1]), int(key.shape[2]), device=key.device)
            layer_mean = scores.mean(dim=0).clamp_min(0.0)  # [cache_len] -- head-shared, matches laprox_layer_token_scores
            denom = layer_mean.sum().clamp_min(1e-12)
            norm_scores = layer_mean / denom
            layer_scores.append(norm_scores)
            # LaProx's own global_layer protected window is snapkv_obs_window,
            # not recent_tokens -- match it exactly in "laprox" verification
            # style so rfc_lambda=0 reproduces LaProx's own score.
            protected_window = self.cfg.snapkv_obs_window if self.cfg.rfc_recent_style == "laprox" else self.cfg.recent_tokens
            protected = recent_absolute_positions(self.positions[layer_idx], protected_window)
            # The ONLY thing separating this path from _prune_laprox_global_layer
            # at lambda=0: that one protects the recent window alone, this also
            # force-keeps Llama-3's chat-template control tokens. It is applied
            # at every lambda, so "lambda=0 == LaProx" is false by exactly this
            # much (measured qasper +0.02 but multifieldqa_en -0.72 against the
            # pure-LaProx baseline). Set protect_special=False to drop it and
            # make the two paths genuinely identical at lambda=0.
            if self.special_positions and getattr(self.cfg, "protect_special", True):
                protected = protected | self.special_positions
            protected_by_layer.append(protected)
            ref_positions = self.positions[layer_idx][0]
            for slot in range(int(key.shape[2])):
                abs_pos = int(ref_positions[slot].item())
                if abs_pos in protected:
                    continue
                candidates.append((float(norm_scores[slot].item()), layer_idx, slot))
            if use_floor and attentions and layer_idx < len(attentions) and attentions[layer_idx] is not None:
                # Recent-only's OWN ranking (ignores the correction entirely) --
                # used below to reserve most of the budget as a "safety floor"
                # bounding how much of the budget a miscalibrated correction can
                # swap away from what recent-only alone would have kept.
                s_recent_only = laprox_head_token_scores(self.model, layer_idx, attentions[layer_idx], value, self.cfg).to(key.device)
                recent_layer_mean = s_recent_only.mean(dim=0).clamp_min(0.0)
                recent_norm = recent_layer_mean / recent_layer_mean.sum().clamp_min(1e-12)
                for slot in range(int(key.shape[2])):
                    abs_pos = int(ref_positions[slot].item())
                    if abs_pos in protected:
                        continue
                    floor_candidates.append((float(recent_norm[slot].item()), layer_idx, slot))
        keep_slots = [set() for _ in layers]
        for layer_idx, protected in enumerate(protected_by_layer):
            ref_positions = self.positions[layer_idx][0]
            for slot in range(int(ref_positions.numel())):
                if int(ref_positions[slot].item()) in protected:
                    keep_slots[layer_idx].add(slot)
        protected_total = sum(len(x) for x in keep_slots)
        if use_floor and floor_candidates:
            floor_budget = max(0, int(round(float(self.cfg.rfc_rescue_floor_frac) * total_budget)) - protected_total)
            floor_candidates.sort(key=lambda x: x[0], reverse=True)
            floor_selected: set[tuple[int, int]] = set()
            for _, layer_idx, slot in floor_candidates[:floor_budget]:
                keep_slots[layer_idx].add(slot)
                floor_selected.add((layer_idx, slot))
            protected_total += len(floor_selected)
            # Drop floor-claimed slots from the discretionary (combined-score)
            # pool -- they're already guaranteed, and leaving them in would let
            # them get "re-picked" there, wasting a discretionary slot that
            # should go to a genuinely different candidate.
            if floor_selected:
                candidates = [c for c in candidates if (c[1], c[2]) not in floor_selected]
        remaining = max(0, total_budget - protected_total)
        if remaining:
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, layer_idx, slot in candidates[:remaining]:
                keep_slots[layer_idx].add(slot)
        # Keep every layer non-empty so HuggingFace attention/cache code remains rectangular per layer.
        for layer_idx, slots in enumerate(keep_slots):
            if not slots:
                best = int(torch.argmax(layer_scores[layer_idx]).item())
                slots.add(best)
        for layer_idx, slots in enumerate(keep_slots):
            key, _ = legacy_layers(past_key_values)[layer_idx]
            slot_tensor = torch.tensor(sorted(slots), dtype=torch.long, device=key.device)
            keep = slot_tensor.view(1, -1).expand(key.shape[1], -1)
            past_key_values = self._apply_layer_keep(past_key_values, layer_idx, keep)
        return past_key_values

    def _h2o_keep(self, layer_idx: int, budget: int) -> torch.Tensor:
        positions = self.positions[layer_idx]
        scores = self.h2o_scores[layer_idx]
        recent = max(0, min(int(self.cfg.recent_tokens), int(budget)))
        protected = set()
        if recent:
            for h in range(positions.shape[0]):
                newest = positions[h].sort().values[-recent:]
                protected.update(int(x) for x in newest.tolist())
        heavy_budget = max(0, int(budget) - len(protected))
        if heavy_budget <= 0:
            return start_recent_indices(positions, budget, sink_tokens=0)
        from rescue.policies.scoring import select_topk_indices

        return select_topk_indices(scores, positions, budget, protected)

    def _rfc_base_score(self, layer_idx: int, key, value, attentions):
        """s_recent alone -- the base ranker's own opinion, with no correction
        mixed in. Used by the floor, which has to reserve what the BASE would
        have kept rather than what the combined score wants."""
        if not attentions or layer_idx >= len(attentions) or attentions[layer_idx] is None:
            return None
        style = self.cfg.rfc_recent_style
        try:
            if style == "laprox":
                return laprox_head_token_scores(self.model, layer_idx, attentions[layer_idx], value, self.cfg).to(key.device)
            if style == "snapkv":
                return snapkv_head_token_scores(layer_idx, attentions[layer_idx], value, self.cfg).to(key.device)
            if style == "h2o":
                return self.h2o_scores[layer_idx].to(key.device) if self.h2o_scores else None
        except Exception:
            return None
        return None

    def _learned_keep(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, budget: int, attentions, hidden_states) -> torch.Tensor:
        if self.learned_scorer is None:
            if self.cfg.normalized_policy() == "kvp" and self.kvp_paper_scorer is not None:
                scores = self.kvp_paper_scorer.score_layer(key, value, self.positions[layer_idx], layer_idx)
                # The official SamplerAgentPress library defaults n_sinks=0,
                # n_running_window=0, but THESE checkpoints' own saved config
                # (agent.pt's "config" dict, every layer/kv_head, all 3
                # backbones) records n_sinks=4, n_running_window=16 -- i.e.
                # they were actually trained assuming an external harness
                # always protects the first 4 + last 16 positions, so the
                # agent never learned to value sink tokens itself. Evicting
                # them anyway (as an earlier fix here did, chasing the
                # library's generic default instead of this checkpoint's
                # real training config) collapses generation into repeated-
                # token garbage -- confirmed on qasper predictions ("Ghost
                # Ghost Ghost...", "the the the the...").
                protected: set[int] = set()
                for h in range(self.positions[layer_idx].shape[0]):
                    pos_h = self.positions[layer_idx][h]
                    protected.update(int(x) for x in pos_h[pos_h < 4].tolist())
                    recent_n = min(16, pos_h.shape[0])
                    if recent_n:
                        protected.update(int(x) for x in pos_h.sort().values[-recent_n:].tolist())
                from rescue.policies.scoring import select_topk_indices

                return select_topk_indices(scores, self.positions[layer_idx], budget, protected)
            if self.cfg.normalized_policy() == "foresightkv" and self.foresight_paper_scorer is not None:
                layer_attn = None
                if attentions and layer_idx < len(attentions) and attentions[layer_idx] is not None:
                    layer_attn = attentions[layer_idx]
                if isinstance(self.foresight_paper_scorer, ForesightKVJudgeScorer):
                    raw_attn = layer_attn[0] if layer_attn is not None else None
                    # score_layer returns evict-priority (higher = more evictable,
                    # matching the official ForesightKV convention verified against
                    # reference/official/ForesightKV's topk_prob_sample, which
                    # samples DROP candidates from the highest scores). Every
                    # policy in this file's select_topk_indices call instead keeps
                    # the highest scorers, so this must be negated here or
                    # foresightkv keeps exactly the tokens its own judge says are
                    # safest to drop.
                    scores = -self.foresight_paper_scorer.score_layer(key[0], value[0], self.positions[layer_idx], layer_idx, raw_attn)
                    # Official ForesightKV hardcodes its recency-protected window
                    # to 128 (reference/official/ForesightKV/.../foresightkv.py:84,
                    # `self.length = 128`, independent of its `window_size` ctor
                    # arg) -- not this repo's shared --recent-tokens=64 default.
                    protected = recent_absolute_positions(
                        self.positions[layer_idx],
                        int(getattr(self.cfg, "foresight_recent_window", 128)))
                else:
                    attn_features = None
                    if layer_attn is not None:
                        attn_features = self._learned_attention_features(layer_idx, layer_attn, key)
                    scores = self.foresight_paper_scorer.score_layer(key, value, self.positions[layer_idx], layer_idx, attn_features)
                    protected = recent_absolute_positions(self.positions[layer_idx], self.cfg.recent_tokens)
                from rescue.policies.scoring import select_topk_indices

                return select_topk_indices(scores, self.positions[layer_idx], budget, protected)
            if self.cfg.normalized_policy() == "rfc" and (self.rfc_scorer is not None or self.unified_rfc_scorer is not None or self.external_future_scorer is not None):
                from rescue.policies.scoring import select_topk_indices

                scores = self._rfc_score(layer_idx, key, value, attentions, hidden_states)
                if scores is None:
                    # scorer state (q_cache/z_cache) not (yet) populated for this
                    # layer/step -- fall back to plain recency rather than crash
                    # (e.g. a call that never got hidden_states/queries, which
                    # shouldn't happen in the normal generate_one path but keeps
                    # this branch defensive).
                    return start_recent_indices(self.positions[layer_idx], budget, self.cfg.sink_tokens)
                # Follow the BASE's own protected-window convention, not this
                # file's shared --recent-tokens default. RESCUE is a correction
                # bolted onto an existing method, so lambda=0 has to reproduce
                # that method exactly -- SnapKV protects snapkv_obs_window (32,
                # snapkv_prefill_indices' obs window), H2O protects
                # recent_tokens (64, _h2o_keep), and only matching each makes
                # the gated-off case equal to the base it is correcting.
                protected_window = (self.cfg.snapkv_obs_window
                                    if self.cfg.rfc_recent_style == "snapkv"
                                    else self.cfg.recent_tokens)
                protected = recent_absolute_positions(self.positions[layer_idx], protected_window)
                # FLOOR, per_head counterpart of _prune_rfc_global_layer's.
                # Reserve part of the budget for the BASE's own top ranking so a
                # miscalibrated correction cannot swap it away. This is aimed at
                # the measured failure on the strong bases: on R-KV trec the
                # correction evicts the exact label strings the base was keeping
                # (share of predictions that are a verbatim label: 76% -> 62%),
                # and no lambda fixes it because every lambda trades trec
                # against qasper. Reserving the base's own picks bounds the
                # displacement instead of scaling it.
                floor_frac = float(getattr(self.cfg, "rfc_rescue_floor_frac", 0.0) or 0.0)
                if self.cfg.rfc_objective == "RESCUE" and floor_frac > 0.0:
                    base_only = self._rfc_base_score(layer_idx, key, value, attentions)
                    if base_only is not None:
                        n_floor = max(0, int(round(floor_frac * budget)) - len(protected))
                        if n_floor > 0:
                            pos_row = self.positions[layer_idx]
                            head_mean = base_only.float().mean(dim=0)
                            k = min(n_floor, int(head_mean.shape[-1]))
                            top_slots = head_mean.topk(k).indices.tolist()
                            ref = pos_row[0]
                            protected = set(protected) | {
                                int(ref[sl].item()) for sl in top_slots
                                if sl < int(ref.numel())}
                return select_topk_indices(scores, self.positions[layer_idx], budget, protected)
            raise ValueError(f"{self.cfg.normalized_policy()} requires a loaded learned scorer")
        attn_features = None
        if attentions and layer_idx < len(attentions) and attentions[layer_idx] is not None:
            attn_features = self._learned_attention_features(layer_idx, attentions[layer_idx], key)
        q_group = None
        if hidden_states is not None:
            q_group = self._current_q_group(layer_idx, hidden_states, key.device)
        scores = self.learned_scorer.score_layer(
            key=key,
            value=value,
            positions=self.positions[layer_idx],
            layer_id=layer_idx,
            attn_features=attn_features,
            q_group=q_group,
        )
        protected = recent_absolute_positions(self.positions[layer_idx], self.cfg.recent_tokens)
        from rescue.policies.scoring import select_topk_indices

        return select_topk_indices(scores, self.positions[layer_idx], budget, protected)

    def _lambda_is_zero(self) -> bool:
        """Is the lambda this layer will use exactly 0, knowable before the
        correction is computed?

        Knowing this early is what makes a declined correction free: the
        scorer's forward pass over every candidate in every layer is skipped
        rather than computed and multiplied by zero.

        An external future scorer supplies its own lambda, which is not visible
        from here, so that case falls through and the correction is computed as
        before.
        """
        if self.external_future_scorer is not None:
            return False
        sc = self.unified_rfc_scorer if self.unified_rfc_scorer is not None else self.rfc_scorer
        if sc is None or not hasattr(sc, "lambda_mix"):
            return False
        if float(sc.lambda_mix) != 0.0:
            return False
        # a gate can only push lambda further down, never up
        return str(getattr(self.cfg, "rfc_combine_mode", "")) == "scale_additive"

    def _rfc_score(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, attentions, hidden_states) -> torch.Tensor | None:
        """Computes rfc's S_total = rfc_recent_weight*S_recent + lambda*S_future
        for every (kv_head, cache_slot) in this layer, WITHOUT making an
        eviction decision -- extracted out of _learned_keep so both the
        default per-head top-k path and the global_layer (head-shared,
        cross-layer-ragged budget) path can share the identical scoring
        logic. Returns None if the scorer's running state isn't ready yet
        for this layer/step (caller should fall back to recency)."""
        num_kv_heads = int(value.shape[1])
        cache_len = int(key.shape[2])
        using_unified = self.unified_rfc_scorer is not None
        z_layer = self.z_cache[layer_idx] if layer_idx < len(self.z_cache) else None
        q_ring = self.q_cache[layer_idx] if layer_idx < len(self.q_cache) else None
        # z_cache is only genuinely read by the OLD "learned" impact branch
        # below (score_future's c_t) -- the no-impact and vwo-impact branches
        # both used to read z_layer[-1] too, but never actually USED the
        # result, so they never really needed z_cache to be ready. Gating
        # z_ready on it unconditionally silently degraded rfc's
        # global_layer allocation to plain recency after the first decode
        # step (RaggedLlamaLikeDecoder doesn't produce hidden_states, so
        # z_cache falls out of sync -- see _apply_layer_keep).
        needs_z_cache = (not using_unified) and self.cfg.rfc_use_impact and self.cfg.rfc_impact_mode != "vwo"
        z_ready = using_unified or not needs_z_cache or (z_layer is not None and int(z_layer.shape[0]) == cache_len)
        # external_future_scorer (KVP/ForesightKV plugged in as S_future)
        # never reads q_ring at all -- it wasn't populated in the first place
        # since _populate_q_cache only recognizes rfc_scorer/unified_rfc_scorer,
        # so requiring it here silently returned None on EVERY call, which
        # per_head allocation masks (falls back to plain recency, looking
        # superficially fine) but global_layer allocation does not (every
        # layer's score collapses to the same all-zero fallback, degenerate).
        if not z_ready or (q_ring is None and self.external_future_scorer is None):
            return None

        layer_attn = None
        if attentions and layer_idx < len(attentions) and attentions[layer_idx] is not None:
            layer_attn = attentions[layer_idx][0]  # [num_heads, q_len, k_len]
        if layer_attn is not None:
            kv_repeat = max(1, int(layer_attn.shape[0]) // num_kv_heads)
            if self.cfg.rfc_recent_style == "laprox":
                # LaProx's OWN exact formula (||attn||_2 * ||V W_O||, per
                # kv-head) instead of rfc's Gram-matrix true-contribution --
                # used only to make rfc_lambda=0 + rfc_allocation="global_layer"
                # reproduce LaProx's own score exactly, as a verification
                # baseline. Needs the UNSTRIPPED (batch-dim-included)
                # attentions[layer_idx]/value tensors -- laprox_head_token_scores
                # strips internally via query_to_kv_attention.
                s_recent = laprox_head_token_scores(self.model, layer_idx, attentions[layer_idx], value, self.cfg).to(key.device)
            elif self.cfg.rfc_recent_style == "snapkv":
                # SnapKV's OWN formula (sum of obs-window attention, no value
                # weighting, same kernel=7 smoothing pool) -- RESCUE ported
                # onto a SnapKV base instead of LaProx (Framing B: portable
                # correction module). Same obs-window attentions[layer_idx]
                # input as the laprox branch, just a different per-candidate
                # score formula.
                s_recent = snapkv_head_token_scores(layer_idx, attentions[layer_idx], value, self.cfg).to(key.device)
            elif self.cfg.rfc_recent_style == "h2o":
                # H2O's OWN formula (cumulative attention mass from EVERY
                # query position seen so far, not just the obs window) --
                # RESCUE ported onto an H2O base (Framing B). self.h2o_scores
                # was populated from a SEPARATE _h2o_chunked_prefill pass (see
                # generate_one) since obs-window attentions[layer_idx] can't
                # give this; already kept in sync with this layer's surviving
                # positions by _apply_layer_keep's gather, same as policy="h2o".
                s_recent = self.h2o_scores[layer_idx].to(key.device)
            else:
                o_proj_fp32 = self._rfc_o_proj_fp32.get(layer_idx)
                if o_proj_fp32 is None:
                    layer = getattr(getattr(self.model, "model", self.model), "layers", [])[layer_idx]
                    o_proj_fp32 = layer.self_attn.o_proj.weight.float()
                    self._rfc_o_proj_fp32[layer_idx] = o_proj_fp32
                s_recent = recent_contribution_from_attn(layer_attn, value[0], o_proj_fp32, kv_repeat).to(key.device)
        else:
            s_recent = torch.zeros(cache_len, device=key.device)

        k_cand = key[0]  # [num_kv_heads, cache_len, head_dim]
        if using_unified:
            kv_repeat = max(1, self.model.config.num_attention_heads // self.model.config.num_key_value_heads)
            o_proj_fp32 = self._rfc_o_proj_fp32.get(layer_idx)
            if o_proj_fp32 is None:
                layer = getattr(getattr(self.model, "model", self.model), "layers", [])[layer_idx]
                o_proj_fp32 = layer.self_attn.o_proj.weight.float()
                self._rfc_o_proj_fp32[layer_idx] = o_proj_fp32
            cur_t = int(self.positions[layer_idx][0, -1].item())
            activity = None
            if self.activity_recency and layer_idx < len(self.activity_recency):
                activity = (self.activity_recency[layer_idx], self.activity_freq[layer_idx], self.activity_streak[layer_idx])
            if self._lambda_is_zero():
                # lambda=0 means combine() returns s_recent untouched, so the
                # correction it would be multiplied by is never read. Computing
                # it anyway costs a full MLP pass over every candidate in every
                # layer -- 145 ms per document here, and the fidelity selector
                # pays it once per candidate, which is where most of its
                # "setup" time was going.
                s_future = k_cand.new_zeros((k_cand.shape[0], k_cand.shape[1]))
            else:
                s_future = self.unified_rfc_scorer.score(
                    q_ring, k_cand, value[0], self.positions[layer_idx], cur_t, o_proj_fp32, kv_repeat, layer_idx,
                    activity=activity, base_score=s_recent,
                ).to(key.device)
        elif self.external_future_scorer is not None:
            # Reverse experiment: KVP/ForesightKV's OWN future-prediction
            # mechanism plugged in as S_future, on top of the SAME laprox-
            # exact global-wise S_recent used for rfc's MLPs above -- tests
            # whether the S_recent fix generalizes beyond rfc's own
            # predictors to other baselines' future signals too.
            if self.external_future_kind == "KVP_FUTURE":
                s_future = self.external_future_scorer.score_layer(key, value, self.positions[layer_idx], layer_idx).to(key.device)
            else:  # FKV_FUTURE -- evict-priority convention, negate to match "higher = keep"
                s_future = -self.external_future_scorer.score_layer(key[0], value[0], self.positions[layer_idx], layer_idx, layer_attn).to(key.device)
        elif not self.cfg.rfc_use_impact:
            recent_q_flat = q_ring.flatten().to(key.device)
            s_future = self.rfc_scorer.score_reuse(recent_q_flat, k_cand, layer_idx).to(key.device)
        elif self.cfg.rfc_impact_mode == "vwo":
            recent_q_flat = q_ring.flatten().to(key.device)
            reuse_hat = self.rfc_scorer.score_reuse(recent_q_flat, k_cand, layer_idx)
            o_proj_fp32 = self._rfc_o_proj_fp32.get(layer_idx)
            if o_proj_fp32 is None:
                layer = getattr(getattr(self.model, "model", self.model), "layers", [])[layer_idx]
                o_proj_fp32 = layer.self_attn.o_proj.weight.float()
                self._rfc_o_proj_fp32[layer_idx] = o_proj_fp32
            kv_repeat = max(1, self.model.config.num_attention_heads // self.model.config.num_key_value_heads)
            impact_hat = self.rfc_scorer.score_vwo_impact(value[0], o_proj_fp32, kv_repeat)
            s_future = (reuse_hat * impact_hat.unsqueeze(0).to(reuse_hat.device)).to(key.device)
        else:
            c_t = z_layer[-1]
            recent_q_flat = q_ring.flatten().to(key.device)
            s_future = self.rfc_scorer.score_future(recent_q_flat, k_cand, z_layer, c_t, layer_idx).to(key.device)
        if self.external_future_scorer is not None:
            lam = float(self.cfg.rfc_lambda)
        else:
            active_scorer = self.unified_rfc_scorer if using_unified else self.rfc_scorer
            lam = float(active_scorer.lambda_mix)
        # s_recent is [cache_len] for the default "contribution" style
        # (position-level, head-shared) but already [kv_heads, cache_len]
        # for "laprox" style (laprox_head_token_scores is per-head) --
        # only unsqueeze when it's still 1D. Needed before the gate below,
        # which compares the two sides' top-B sets.
        s_recent_term = s_recent if s_recent.dim() == 2 else s_recent.unsqueeze(0)
        ppl_tau = float(getattr(self.cfg, "rfc_ppl_gate_tau", 0.0) or 0.0)
        if ppl_tau > 0.0 and not torch.is_tensor(lam):
            # PERPLEXITY GATE. One number per document, already computed during
            # prefill: the mean entropy of the model's own next-token
            # distribution over the prompt. Below tau the prompt is predictable
            # from local context, which is exactly where the recent-window base
            # score is already near-optimal and any correction can only perturb
            # it -- so lambda drops to 0, and with rfc_combine_mode=
            # "scale_additive" that is the untouched base score, i.e. the
            # baseline exactly, never a loss.
            pe = getattr(self, "prompt_entropy", None)
            if pe is not None:
                lam = lam if float(pe) >= ppl_tau else 0.0
                if os.environ.get("RESCUE_GATE_LOG"):
                    print(f"[PPLGATE] entropy={float(pe):.4f} lam={float(lam):.3f}", flush=True)
        if self.cfg.rfc_combine_mode == "rank_gate":
            total = _rank_gate_combine(s_recent_term, s_future, lam)
        elif self.cfg.rfc_combine_mode == "rank_additive":
            total = _rank_additive_combine(s_recent_term, s_future, lam)
        elif self.cfg.rfc_combine_mode == "share_additive":
            total = _share_additive_combine(s_recent_term, s_future, lam)
        elif self.cfg.rfc_combine_mode == "kslot":
            # lambda still gates the correction: lambda=0 reserves every slot for
            # the base policy, which reproduces it exactly, so the fidelity
            # selector works here unchanged.
            # b_rem is computed locally: the two places above that define it sit
            # inside the gate and adaptive-lambda branches, neither of which runs
            # in this configuration, so referencing theirs raised UnboundLocalError.
            b_rem_ks = (int(self.cfg.budget_tokens or 128)
                        - int(self.cfg.sink_tokens) - int(self.cfg.snapkv_obs_window))
            lam_f = float(lam.mean()) if torch.is_tensor(lam) else float(lam)
            k_eff = 0 if lam_f == 0.0 else int(self.cfg.rfc_kslot)
            total = _kslot_combine(s_recent_term, s_future, b_rem_ks, k_eff)
        elif self.cfg.rfc_combine_mode == "scale_additive":
            total = _scale_additive_combine(s_recent_term, s_future, lam)
        else:
            total = float(self.cfg.rfc_recent_weight) * s_recent_term + lam * s_future
        return total.contiguous()

    def _learned_attention_features(self, layer_idx: int, layer_attn: torch.Tensor, key: torch.Tensor) -> dict[str, torch.Tensor]:
        from rescue.policies.scoring import query_to_kv_attention

        kv_heads = int(key.shape[1])
        seq_len = int(key.shape[2])
        grouped = query_to_kv_attention(layer_attn, kv_heads).to(key.device).float()
        current = grouped[:, -1, :seq_len]
        obs_sum = grouped[:, :, :seq_len].sum(dim=1)
        obs_max = grouped[:, :, :seq_len].max(dim=1).values
        obs_mean = grouped[:, :, :seq_len].mean(dim=1)
        zeros = torch.zeros_like(current)
        ones = torch.ones_like(current)
        return {
            "current_attn_prob": current,
            "current_qk_logit": zeros,
            "attention_row_sum": ones,
            "hist_sum_all": obs_sum,
            "hist_max_all": obs_max,
            "hist_mean": obs_mean,
            "hist_variance": zeros,
            "last_attended_distance": zeros,
            "hit_count": (obs_sum > 0.01).float(),
            "hist_sum_last_1": current,
            "hist_sum_last_4": obs_sum,
            "hist_sum_last_16": obs_sum,
            "hist_sum_last_64": obs_sum,
            "hist_sum_last_256": obs_sum,
            "hist_max_last_256": obs_max,
        }

    def _current_q_group(self, layer_idx: int, hidden_states, device: torch.device) -> torch.Tensor | None:
        if hidden_states is None or layer_idx >= len(hidden_states):
            return None
        try:
            layer = getattr(getattr(self.model, "model", self.model), "layers", [])[layer_idx]
            hidden = hidden_states[layer_idx][:, -1:, :].to(next(layer.parameters()).device)
            attn = layer.self_attn
            cfg = self.model.config
            num_heads = int(cfg.num_attention_heads)
            kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
            head_dim = int(getattr(attn, "head_dim", cfg.hidden_size // num_heads))
            query = attn.q_proj(hidden).view(1, 1, num_heads, head_dim).transpose(1, 2)[0, :, 0, :]
            if num_heads % kv_heads != 0:
                return None
            group = num_heads // kv_heads
            return query.view(kv_heads, group, head_dim).detach().to(device)
        except Exception:
            return None


class RaggedLaProxState:
    """LaProx state for fully model-wide layer/head/token allocation."""

    def __init__(self, cfg: BaselineConfig, model):
        self.cfg = cfg
        self.model = model
        self.prefill_retained = 0
        self.total_evicted = 0

    def initialize_after_prefill(self, past_key_values, attentions, seq_len: int):
        if not attentions:
            raise RuntimeError("LaProx global_head requires output attentions. Set model_kwargs.attn_implementation='eager'.")
        cache = RaggedKVCache.from_past(past_key_values, seq_len)
        scores_by_layer = []
        for layer_idx, layer_attn in enumerate(attentions):
            value = legacy_layers(past_key_values)[layer_idx][1]
            scores_by_layer.append(laprox_head_token_scores(self.model, layer_idx, layer_attn, value, self.cfg))
        self._prune(cache, scores_by_layer)
        self.prefill_retained = cache.total_tokens()
        return cache

    def update_and_prune_after_decode(self, cache: RaggedKVCache, ragged_attentions):
        scores_by_layer = []
        for layer_idx, layer_attn_by_head in enumerate(ragged_attentions or []):
            scores_by_layer.append(self._decode_scores(layer_idx, layer_attn_by_head, cache))
        self._prune(cache, scores_by_layer)
        return cache

    def _decode_scores(self, layer_idx: int, layer_attn_by_head: list[torch.Tensor], cache: RaggedKVCache) -> list[torch.Tensor]:
        scores: list[torch.Tensor] = []
        layer = getattr(getattr(self.model, "model", self.model), "layers", [])[layer_idx]
        num_query_heads = int(self.model.config.num_attention_heads)
        kv_heads = int(getattr(self.model.config, "num_key_value_heads", num_query_heads))
        group = max(1, num_query_heads // max(1, kv_heads))
        head_dim = int(cache.values[layer_idx][0].shape[-1])
        blocks = [layer.self_attn.o_proj.weight.detach()[:, h * head_dim : (h + 1) * head_dim] for h in range(num_query_heads)]
        for kv_h, attn_h in enumerate(layer_attn_by_head):
            # Existing LaProx code groups query heads per KV head, then takes the
            # vector norm over query positions. With q_len=1 this is abs(sum).
            attn_score = attn_h.detach().float().sum(dim=0).norm(p=2, dim=0).to(cache.values[layer_idx][kv_h].device)
            val = cache.values[layer_idx][kv_h].detach().float()[0, 0]
            vwo = torch.zeros(val.shape[0], dtype=torch.float32, device=val.device)
            for r in range(group):
                qh = kv_h * group + r
                projected = val @ blocks[qh].float().t()
                vwo += projected.norm(p=2, dim=-1)
            scores.append(attn_score * (vwo / float(group)))
        return scores

    def _prune(self, cache: RaggedKVCache, scores_by_layer):
        current_total = cache.total_tokens()
        if current_total <= 0:
            return
        max_len = max((cache.layer_max_len(layer_idx) for layer_idx in range(len(cache))), default=0)
        per_head_budget = resolve_budget(max_len, self.cfg)
        head_count = sum(cache.layer_head_count(layer_idx) for layer_idx in range(len(cache)))
        total_budget = max(head_count, min(current_total, per_head_budget * head_count))
        candidates: list[tuple[float, int, int, int]] = []
        keep: list[list[set[int]]] = [
            [set() for _ in range(cache.layer_head_count(layer_idx))] for layer_idx in range(len(cache))
        ]
        for layer_idx in range(len(cache)):
            layer_scores = scores_by_layer[layer_idx]
            if isinstance(layer_scores, torch.Tensor):
                flat_scores = layer_scores.detach().float()
                denom = flat_scores.sum().clamp_min(1e-12)
                score_heads = [flat_scores[h, : cache.keys[layer_idx][h].shape[2]] / denom for h in range(flat_scores.shape[0])]
            else:
                flat_total = torch.stack([s.detach().float().sum() for s in layer_scores]).sum().clamp_min(1e-12)
                score_heads = [s.detach().float() / flat_total for s in layer_scores]
            for kv_h, scores in enumerate(score_heads):
                positions = cache.positions[layer_idx][kv_h]
                recent = max(0, min(int(self.cfg.snapkv_obs_window), int(positions.numel())))
                protected = set(int(x) for x in positions[-recent:].tolist()) if recent else set()
                for slot in range(int(positions.numel())):
                    if int(positions[slot].item()) in protected:
                        keep[layer_idx][kv_h].add(slot)
                    else:
                        candidates.append((float(scores[slot].item()), layer_idx, kv_h, slot))
        protected_total = sum(len(head_keep) for layer in keep for head_keep in layer)
        remaining = max(0, total_budget - protected_total)
        if remaining:
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, layer_idx, kv_h, slot in candidates[:remaining]:
                keep[layer_idx][kv_h].add(slot)
        for layer_idx in range(len(cache)):
            for kv_h in range(cache.layer_head_count(layer_idx)):
                if not keep[layer_idx][kv_h]:
                    scores = scores_by_layer[layer_idx][kv_h] if isinstance(scores_by_layer[layer_idx], torch.Tensor) else scores_by_layer[layer_idx][kv_h]
                    keep[layer_idx][kv_h].add(int(torch.argmax(scores).item()))
        for layer_idx in range(len(cache)):
            for kv_h in range(cache.layer_head_count(layer_idx)):
                key = cache.keys[layer_idx][kv_h]
                slots = torch.tensor(sorted(keep[layer_idx][kv_h]), dtype=torch.long, device=key.device)
                self.total_evicted += cache.prune_head(layer_idx, kv_h, slots)


class RescueInjection:
    """RESCUE bolted onto a RAGGED base (LAVa, R-KV) without touching how that
    base allocates budget.

    LaProx/SnapKV/H2O fit RFC's own rectangular prune paths, so RESCUE rides
    along there by being part of the score _rfc_score returns. LAVa and R-KV
    cannot: LAVa spends budget across layers in proportion to score entropy and
    then re-splits it across kv-heads (Algorithm 1), and R-KV takes an
    independent top-k per kv-head -- both leave heads with different lengths and
    need RaggedKVCache. Reimplementing those allocations inside RFC would be
    reimplementing the methods.

    So the correction is injected at the ONE line where each ragged state
    produces its per-(kv_head, token) score, and everything downstream --
    entropy budgeting, dynamic head budget, per-head top-k, the ragged cache --
    runs unchanged on the corrected ranking. lambda=0 leaves the base score
    untouched, exactly as scale_additive does elsewhere, so the gated-off case
    is still the base method itself.
    """

    def __init__(self, cfg, model, scorer, queries: dict, prompt_entropy=None):
        self.cfg = cfg
        self.model = model
        self.scorer = scorer
        self.prompt_entropy = prompt_entropy
        self.q_cache: list = []
        self._o_proj_fp32: dict = {}
        num_kv_heads = int(model.config.num_key_value_heads)
        window = scorer.recent_window
        for layer_idx, q in (queries or {}).items():
            while len(self.q_cache) <= layer_idx:
                self.q_cache.append(QueryRingBuffer(window))
            self.q_cache[layer_idx].push(q, num_kv_heads)

    def push_queries(self, queries: dict) -> None:
        """Decode-time counterpart of the constructor's initial fill: the
        correction scores against the most recent `recent_window` queries, so
        the ring has to keep advancing as tokens are generated."""
        if self.scorer is None:
            return
        num_kv_heads = int(self.model.config.num_key_value_heads)
        for layer_idx, q in (queries or {}).items():
            while len(self.q_cache) <= layer_idx:
                self.q_cache.append(QueryRingBuffer(self.scorer.recent_window))
            self.q_cache[layer_idx].push(q, num_kv_heads)

    def _lambda(self) -> float:
        lam = float(self.cfg.rfc_lambda)
        tau = float(getattr(self.cfg, "rfc_ppl_gate_tau", 0.0) or 0.0)
        if tau > 0.0 and self.prompt_entropy is not None:
            lam = lam if float(self.prompt_entropy) >= tau else 0.0
        return lam

    def combine(self, layer_idx: int, base_score, key_full, value_full, positions, cur_t: int):
        """base_score: [kv_heads, N] the ragged state was about to rank on.
        Returns the same shape, corrected. Any missing piece (no scorer, no
        queries for this layer) returns base_score untouched rather than
        silently ranking on a zero correction."""
        lam = self._lambda()
        if lam == 0.0 or self.scorer is None:
            return base_score
        if layer_idx >= len(self.q_cache) or self.q_cache[layer_idx] is None:
            return base_score
        o_proj = self._o_proj_fp32.get(layer_idx)
        if o_proj is None:
            layer = getattr(getattr(self.model, "model", self.model), "layers", [])[layer_idx]
            o_proj = layer.self_attn.o_proj.weight.float()
            self._o_proj_fp32[layer_idx] = o_proj
        kv_repeat = max(1, int(self.model.config.num_attention_heads) //
                        int(self.model.config.num_key_value_heads))
        k_cand = key_full[0] if key_full.dim() == 4 else key_full
        v_cand = value_full[0] if value_full.dim() == 4 else value_full
        n = int(base_score.shape[-1])
        # Deliberately NOT wrapped in try/except: a correction that silently
        # falls back to the base score is indistinguishable from a gate that
        # decided to switch off, and would be reported as "no effect" instead
        # of as the bug it is.
        s_future = self.scorer.score(
            self.q_cache[layer_idx], k_cand, v_cand, positions, cur_t,
            o_proj, kv_repeat, layer_idx, base_score=base_score,
        ).to(base_score.device)
        # base_score may cover fewer positions than the cache (R-KV drops its
        # trailing window); align on the leading N, which is where every
        # candidate lives.
        if s_future.shape[-1] > n:
            s_future = s_future[..., :n]
        elif s_future.shape[-1] < n:
            raise RuntimeError(
                f"correction produced {s_future.shape[-1]} scores for {n} ranked "
                f"positions at layer {layer_idx}; refusing to rank on a padded "
                f"or truncated correction")
        if s_future.shape[0] == 1 and base_score.shape[0] > 1:
            s_future = s_future.expand(base_score.shape[0], -1)
        # Honour rfc_combine_mode here too. This path serves the ragged bases
        # (R-KV, LAVa), which never reach _learned_keep's combine, so hardcoding
        # scale_additive meant --rfc-combine-mode was silently ignored for exactly
        # the base policy the kslot mode exists to protect: every k produced an
        # identical score because none of them was ever applied.
        if str(getattr(self.cfg, "rfc_combine_mode", "")) == "kslot":
            b_rem_ks = (int(self.cfg.budget_tokens or 128)
                        - int(self.cfg.sink_tokens) - int(self.cfg.snapkv_obs_window))
            k_eff = 0 if lam == 0.0 else int(getattr(self.cfg, "rfc_kslot", 16))
            out = _kslot_combine(base_score.float(), s_future.float(), b_rem_ks, k_eff)
        else:
            out = _scale_additive_combine(base_score.float(), s_future.float(), lam)
        # FLOOR for the ragged bases. _learned_keep's version cannot reach here
        # (LAVa and R-KV run their own state classes), so the reservation is
        # applied to the score itself: lift the base's own top-N above every
        # corrected value, which every downstream allocation -- LAVa's entropy
        # budget, R-KV's per-head top-k -- then keeps by construction.
        floor_frac = float(getattr(self.cfg, "rfc_rescue_floor_frac", 0.0) or 0.0)
        if floor_frac > 0.0:
            budget = int(self.cfg.budget_tokens or 128)
            n_floor = max(0, int(round(floor_frac * budget)))
            if n_floor > 0:
                k = min(n_floor, n)
                idx = base_score.float().mean(dim=0).topk(k).indices
                ceiling = out.max().detach() + 1.0
                out = out.clone()
                out[:, idx] = ceiling
        if os.environ.get("RESCUE_INJECT_LOG"):
            b = max(1, int(self.cfg.budget_tokens or 128) - int(self.cfg.snapkv_obs_window))
            k = min(b, n)
            before = set(base_score[0].topk(k).indices.tolist())
            after = set(out[0].topk(k).indices.tolist())
            print(f"[INJECT] layer={layer_idx} lam={lam:g} n={n} k={k} "
                  f"changed={k - len(before & after)}", flush=True)
        return out


class RaggedLaVaState:
    """LAVa (paper-faithful, implemented exactly as published -- no sink-
    token protection, see lava_head_token_scores' docstring): per-layer
    entropy-proportional budget allocation (Eq 6-7) + per-layer "dynamic
    head budget" (Algorithm 1: flatten every kv_head's scores within a
    layer into ONE pool and take a single global top-B_l regardless of
    which head a candidate came from). Unlike every fixed-per-head-budget
    baseline here, this means different kv_heads in the SAME layer can end
    up retaining different token COUNTS and different POSITIONS -- needing
    RaggedKVCache, same as LaProx's "global_head" mode (RaggedLaProxState).

    Single-shot only: prunes once, right after prefill. LAVa's own paper
    does progressive per-layer recompression DURING prefill purely to
    bound peak memory (never materializing all L layers' uncompressed KV
    at once) -- since a token's score never depends on what's currently
    evicted, repeatedly re-selecting top-B from a shrinking budget with a
    fixed score is mathematically identical to selecting the final top-B
    in one pass. This project only measures downstream accuracy, never
    peak prefill memory, so a single post-prefill pass reproduces
    byte-identical retained-token decisions without the memory-engineering
    complexity. Decode-time re-pruning was never implemented (no published
    need for it) -- __init__ raises unless evict_during_decode is False,
    i.e. this MUST be run with --evict-once-after-prefill.
    """

    def __init__(self, cfg: BaselineConfig, model):
        if cfg.evict_during_decode:
            raise ValueError("lava must be run with --evict-once-after-prefill (decode-time re-pruning is not implemented for lava)")
        self.cfg = cfg
        self.model = model
        self.prefill_retained = 0
        self.total_evicted = 0
        self.rescue = None      # set by generate_one for --rfc-recent-style lava

    def initialize_after_prefill(self, past_key_values, attentions, seq_len: int):
        if not attentions:
            raise RuntimeError("lava requires output attentions. Set model_kwargs.attn_implementation='eager'.")
        cache = RaggedKVCache.from_past(past_key_values, seq_len)
        layers = legacy_layers(past_key_values)
        num_layers = len(layers)
        window = max(1, int(self.cfg.snapkv_obs_window))

        layer_scores: list[torch.Tensor] = []  # per layer: [kv_heads, cache_len]
        layer_entropy: list[float] = []
        for layer_idx, (key, value) in enumerate(layers):
            scores = lava_head_token_scores(attentions[layer_idx], value, window, self.cfg).to(key.device)
            if self.rescue is not None:
                cur_t = int(key.shape[2]) - 1
                pos = torch.arange(int(key.shape[2]), device=key.device).view(1, -1).expand(int(key.shape[1]), -1)
                scores = self.rescue.combine(layer_idx, scores, key, value, pos, cur_t)
            layer_scores.append(scores)
            # Algorithm 1 only ever computes s_{l,h}[i] for i not-in [N-w, N]
            # in the first place -- Eq 6-7's entropy must be taken over that
            # SAME candidate-only population, not the full (window-included)
            # score tensor (the recent window tends to score very high,
            # which would skew the entropy/budget-allocation calculation).
            cand_only = scores[:, : max(0, scores.shape[1] - window)]
            flat = cand_only.flatten().clamp_min(0.0)
            denom = flat.sum().clamp_min(1e-12)
            share = flat / denom
            ent = -(share * torch.log(share.clamp_min(1e-12))).sum() / max(1, flat.numel())
            layer_entropy.append(float(ent.item()))

        entropy_sum = sum(layer_entropy) or 1.0
        max_len = max((cache.layer_max_len(l) for l in range(len(cache))), default=0)
        per_layer_budget = resolve_budget(max_len, self.cfg)
        # B_l is a SLOT count for the whole layer, because Algorithm 1 spends it
        # from one pool shared by every kv_head. resolve_budget returns a
        # PER-HEAD token budget (the unit every other baseline here uses), so a
        # layer's share is that times kv_heads. Without the factor, b_l (128)
        # was compared against protected_total (kv_heads * window = 256) below
        # and `remaining` clamped to 0: no candidate was ever selected and every
        # head kept only its last `window` tokens -- 32 of them, sink included
        # in the eviction, which collapsed generation into repeated single
        # tokens (qasper 0.65). At budget 1024 the same bug silently ran the
        # model at an effective 128 tokens/head.
        kv_heads = int(layers[0][1].shape[1]) if layers else 1
        total_budget = max(num_layers, min(cache.total_tokens(),
                                           per_layer_budget * num_layers * kv_heads))
        layer_budgets = [max(1, int(round(total_budget * e / entropy_sum))) for e in layer_entropy]

        for layer_idx in range(num_layers):
            scores = layer_scores[layer_idx]
            kv_heads, cache_len = scores.shape
            protected_start = max(0, cache_len - window)
            keep_by_head: list[set[int]] = [set(range(protected_start, cache_len)) for _ in range(kv_heads)]
            protected_total = sum(len(s) for s in keep_by_head)
            b_l = layer_budgets[layer_idx]
            remaining = max(0, b_l - protected_total)
            # Dynamic head budget (Algorithm 1): flatten ALL heads' candidate
            # scores into one pool and take a single global top-`remaining`,
            # regardless of which head each one came from. Vectorized (a
            # Python-level double loop + per-element .item() here was
            # measured to be prohibitively slow for realistic cache_len).
            if remaining and protected_start > 0:
                cand_scores = scores[:, :protected_start].reshape(-1)
                k = min(remaining, cand_scores.numel())
                top_idx = torch.topk(cand_scores, k=k, largest=True).indices
                top_h = torch.div(top_idx, protected_start, rounding_mode="floor")
                top_slot = top_idx % protected_start
                top_h_list = top_h.tolist()
                top_slot_list = top_slot.tolist()
                for h, slot in zip(top_h_list, top_slot_list):
                    keep_by_head[h].add(slot)
            for h in range(kv_heads):
                if not keep_by_head[h]:
                    keep_by_head[h].add(int(torch.argmax(scores[h]).item()))
                key = cache.keys[layer_idx][h]
                slots = torch.tensor(sorted(keep_by_head[h]), dtype=torch.long, device=key.device)
                self.total_evicted += cache.prune_head(layer_idx, h, slots)
        self.prefill_retained = cache.total_tokens()
        return cache


class RaggedRKVState:
    """R-KV (arXiv 2505.24133), matching the OFFICIAL HuggingFace backend
    exactly (reference/official/R-KV/HuggingFace/rkv/*.py) per explicit
    instruction to match the shipped code's real behavior/defaults, not the
    paper text (which describes a periodic B_buffer-gated compression
    scheme). The real code is simpler: every forward call (prefill's one
    big call, and every subsequent decode step), if a layer's cache_len >=
    budget, compress back down to budget; otherwise leave it alone. See
    rescue/policies/rkv.py's module docstring for the full score formula.

    Ragged, like LAVa's RaggedLaVaState: topk is per-kv_head-independent
    (score.topk(dim=-1) on a [kv_heads, N] tensor), so different heads in
    the same layer generally retain different absolute positions, but
    (unlike LAVa) the same COUNT per head (topk's k is identical across
    heads), which keeps every head's cache length synchronized layer-wide
    even though positions differ -- this is what lets _prune batch all
    kv_heads of a layer into one call instead of a per-head loop.

    UNLIKE LaVa, this needs CONTINUOUS decode-time re-compression: every
    newly generated token gets its own real query, which can change which
    older tokens look "redundant"/important against the latest generation
    context -- not mathematically equivalent to a single prefill-time
    pass. Needs a live per-layer window of the last `rkv_window_size` REAL
    query vectors (full per-head resolution, not kv-head-reduced -- GQA
    reduction happens inside rescue.policies.rkv via MAX, matching the
    official compute_attention_scores(pooling="max")), fed by prefill's
    captured observation-window queries and then by each decode step's
    freshly captured query (see generate_one's push_query calls, mirroring
    exactly how "rfc" feeds its own QueryRingBuffer via _populate_q_cache)."""

    def __init__(self, cfg: BaselineConfig, model):
        self.cfg = cfg
        self.model = model
        self.window_size = int(cfg.rkv_window_size)
        self.mix_lambda = float(cfg.rkv_mix_lambda)
        self.kernel_size = int(cfg.rkv_kernel_size)
        self.retain_ratio = float(cfg.rkv_retain_ratio)
        self.retain_direction = str(cfg.rkv_retain_direction)
        self.query_windows: dict[int, torch.Tensor] = {}  # layer_idx -> [heads, <=window_size, head_dim]
        self.prefill_retained = 0
        self.total_evicted = 0

        self.rescue = None      # set by generate_one for --rfc-recent-style rkv
    def _push_one(self, layer_idx: int, q_new: torch.Tensor) -> None:
        # q_new: [heads, p, head_dim]
        buf = self.query_windows.get(layer_idx)
        buf = q_new if buf is None else torch.cat([buf, q_new.to(buf.device)], dim=1)
        if buf.shape[1] > self.window_size:
            buf = buf[:, -self.window_size:, :]
        self.query_windows[layer_idx] = buf

    def push_query(self, queries: dict[int, torch.Tensor]) -> None:
        """queries[layer_idx]: [heads, p, head_dim] (p new positions -- 1 per
        decode step normally; a bare [heads, head_dim] is unsqueezed)."""
        for layer_idx, q in queries.items():
            if q.dim() == 2:
                q = q.unsqueeze(1)
            self._push_one(layer_idx, q)

    def initialize_after_prefill(self, past_key_values, prefill_queries: dict[int, torch.Tensor], seq_len: int):
        cache = RaggedKVCache.from_past(past_key_values, seq_len)
        for layer_idx in range(len(cache)):
            q = prefill_queries.get(layer_idx)
            if q is None:
                continue
            self._push_one(layer_idx, q[:, -self.window_size:, :])
        self._prune(cache)
        self.prefill_retained = cache.total_tokens()
        return cache

    def update_and_prune_after_decode(self, cache: RaggedKVCache, attentions=None):
        self._prune(cache)
        return cache

    def _prune(self, cache: RaggedKVCache) -> None:
        for layer_idx in range(len(cache)):
            q_win = self.query_windows.get(layer_idx)
            if q_win is None or not cache.keys[layer_idx]:
                continue
            num_kv_heads = cache.layer_head_count(layer_idx)
            n = int(cache.keys[layer_idx][0].shape[2])
            budget = resolve_budget(n, self.cfg)
            if n < budget or n <= self.window_size:
                continue
            key_full = torch.cat(cache.keys[layer_idx], dim=1)  # [1, kv_heads, n, head_dim]
            score = rkv_final_score(
                q_win.to(key_full.device), key_full, self.window_size, self.mix_lambda,
                self.kernel_size, self.retain_ratio, self.retain_direction,
            )  # [kv_heads, n - window_size]
            if self.rescue is not None:
                val_full = torch.cat(cache.values[layer_idx], dim=1)
                pos = torch.arange(n, device=key_full.device).view(1, -1).expand(num_kv_heads, -1)
                score = self.rescue.combine(layer_idx, score, key_full, val_full, pos, n - 1)
            k = max(0, min(budget - self.window_size, score.shape[-1]))
            top_idx = torch.topk(score, k=k, dim=-1, largest=True).indices if k > 0 else score.new_zeros((num_kv_heads, 0), dtype=torch.long)
            tail = torch.arange(n - self.window_size, n, device=score.device).view(1, -1).expand(num_kv_heads, -1)
            keep_all = torch.sort(torch.cat([top_idx, tail], dim=-1), dim=-1).values
            if os.environ.get("RESCUE_KEEP_LOG") and layer_idx < 3:
                # Which tokens actually survive, as a number that can be diffed
                # between two runs. The selector's forced-lambda=1 run scores
                # 34.34 where the same policy run normally scores 35.41, and
                # neither the cache copy nor the probe length explains it -- so
                # compare the kept sets themselves rather than the outputs.
                print(f"[KEEP] L{layer_idx} n={n} k={k} "
                      + " ".join(f"h{h}:{int(keep_all[h].sum())}" for h in range(min(4, num_kv_heads))),
                      flush=True)
            for kv_h in range(num_kv_heads):
                self.total_evicted += cache.prune_head(layer_idx, kv_h, keep_all[kv_h])


class RaggedLlamaLikeDecoder:
    """Batch-1 ragged decode path for Llama/Mistral/Qwen-style decoder layers."""

    def __init__(self, model):
        self.model = model
        self.backbone = getattr(model, "model", model)
        module = importlib.import_module(model.model.layers[0].self_attn.__class__.__module__)
        self.apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb")
        self.repeat_kv = getattr(module, "repeat_kv")

    @torch.inference_mode()
    def decode_one(self, input_ids: torch.Tensor, past_key_values, absolute_pos: int, output_attentions: bool):
        if input_ids.shape[0] != 1 or input_ids.shape[1] != 1:
            raise ValueError("Ragged decode currently supports batch_size=1 and q_len=1 only.")
        hidden_states = self.backbone.embed_tokens(input_ids)
        position_ids = torch.tensor([[int(absolute_pos)]], dtype=torch.long, device=input_ids.device)
        position_embeddings = self.backbone.rotary_emb(hidden_states, position_ids)
        all_attentions = []
        all_queries: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.backbone.layers):
            layer_device = next(layer.parameters()).device
            hidden_states = hidden_states.to(layer_device)
            residual = hidden_states
            attn_input = layer.input_layernorm(hidden_states)
            attn_output, attn_weights, key_new, value_new, query_new = self._attention_one(
                layer,
                layer_idx,
                attn_input,
                past_key_values,
                position_embeddings,
                output_attentions,
            )
            all_queries[layer_idx] = query_new[0]  # strip batch dim -> [num_heads, q_len=1, head_dim], matches _call_with_query_capture's convention
            attn_output = attn_output.to(residual.device)
            if isinstance(past_key_values, RaggedKVCache):
                past_key_values.append_layer(layer_idx, key_new, value_new, absolute_pos=absolute_pos)
            else:
                past_key_values = _append_layer_cache(past_key_values, layer_idx, key_new, value_new)
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            if output_attentions:
                all_attentions.append(attn_weights)
        hidden_states = self.backbone.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits, past_key_values, tuple(all_attentions) if output_attentions else None, all_queries

    def _attention_one(self, layer, layer_idx: int, hidden_states, past_key_values, position_embeddings, output_attentions):
        attn = layer.self_attn
        bsz, q_len, _ = hidden_states.shape
        config = self.model.config
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(attn, "head_dim", config.hidden_size // num_heads))
        query_states = attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        if q_norm is not None:
            query_states = q_norm(query_states)
        if k_norm is not None:
            key_states = k_norm(key_states)
        cos, sin = position_embeddings
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if isinstance(past_key_values, RaggedKVCache):
            return self._ragged_attention_one(
                attn,
                layer_idx,
                query_states,
                key_states,
                value_states,
                past_key_values,
                head_dim,
                output_attentions,
            )
        key_old, value_old = legacy_layers(past_key_values)[layer_idx]
        key_all = torch.cat([key_old, key_states], dim=2)
        value_all = torch.cat([value_old, value_states], dim=2)
        num_groups = num_heads // num_kv_heads
        key_rep = self.repeat_kv(key_all, num_groups)
        value_rep = self.repeat_kv(value_all, num_groups)
        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None, key_states, value_states, query_states

    def _ragged_attention_one(
        self,
        attn,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache: RaggedKVCache,
        head_dim: int,
        output_attentions: bool,
    ):
        bsz, num_heads, q_len, _ = query_states.shape
        num_kv_heads = int(key_states.shape[1])
        group = num_heads // num_kv_heads
        per_q_head: list[torch.Tensor] = [None for _ in range(num_heads)]  # type: ignore[list-item]
        per_kv_attention: list[torch.Tensor] = []
        for kv_h in range(num_kv_heads):
            key_all = torch.cat([cache.keys[layer_idx][kv_h], key_states[:, kv_h : kv_h + 1, :, :]], dim=2)
            value_all = torch.cat([cache.values[layer_idx][kv_h], value_states[:, kv_h : kv_h + 1, :, :]], dim=2)
            q_start = kv_h * group
            q_end = q_start + group
            q_group = query_states[:, q_start:q_end, :, :]
            attn_weights = torch.matmul(q_group, key_all.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            out_group = torch.matmul(attn_weights, value_all)
            for local_h, qh in enumerate(range(q_start, q_end)):
                per_q_head[qh] = out_group[:, local_h : local_h + 1, :, :]
            if output_attentions:
                per_kv_attention.append(attn_weights[0].detach())
        attn_output = torch.cat(per_q_head, dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)
        return attn_output, per_kv_attention if output_attentions else None, key_states, value_states, query_states


class DenseEvictionGenerator:
    """Greedy dense-model generation with KV eviction baselines.

    StreamingLLM/H2O/LaProx can either enforce a strict dynamic cache budget
    during decoding or, for paper-table reproduction, compress once after
    prefill and append generated tokens without further pruning. SnapKV follows
    the prompt-compression setting by default.
    """

    def __init__(self, model, tokenizer, cfg: BaselineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.learned_scorer = None
        self.kvp_paper_scorer = None
        self.foresight_paper_scorer = None
        self.rfc_scorer = None
        self.unified_rfc_scorer = None
        self.external_future_scorer = None
        self.external_future_kind = None
        if cfg.normalized_policy() in {"kvp", "foresightkv", "lookaheadkv", "rfc"}:
            is_oracle = cfg.normalized_policy() == "rfc" and cfg.rfc_objective == "oracle"
            if not cfg.learned_checkpoint and not is_oracle:
                raise ValueError(f"{cfg.normalized_policy()} requires learned_checkpoint")
            if cfg.normalized_policy() == "kvp":
                # See foresightkv_device / rfc_device above: a CPU-resident scorer
                # forces a GPU<->CPU round trip per kv_head/layer/step against the
                # live GPU-resident cache.
                kvp_device = next(model.parameters()).device
                self.kvp_paper_scorer = KVPPaperScorer(cfg.learned_checkpoint, device=kvp_device)
            elif cfg.normalized_policy() == "foresightkv":
                probe = torch.load(cfg.learned_checkpoint, map_location="cpu")
                if "judge_models" in probe and "model_arch" in probe:
                    # genuine ForesightKV mechanism (scripts/train_foresightkv_paper.py
                    # or scripts/train_foresightkv_rl.py output)
                    # Same bug class as rfc's scorer (see rfc_device below): key/value
                    # tensors here come from the live GPU-resident cache, so a
                    # CPU-resident scorer forces a GPU<->CPU round trip on every
                    # layer/step. Measured to dominate cost for rfc (35ms/call ->
                    # 0.40ms/call once moved to the model's device); apply the same
                    # fix here.
                    foresightkv_device = next(model.parameters()).device
                    self.foresight_paper_scorer = ForesightKVJudgeScorer(cfg.learned_checkpoint, device=foresightkv_device)
                else:
                    # pre-fidelity-audit generic MLP proxy checkpoint; kept for
                    # backward compatibility with anything trained against it.
                    self.foresight_paper_scorer = ForesightPaperScorer(cfg.learned_checkpoint, device="cpu")
                del probe
            elif cfg.normalized_policy() == "rfc":
                # Unlike the other learned scorers (which keep their tiny predictor
                # on CPU and briefly move a per-step slice over), rfc's z_cache is
                # maintained as a GPU-resident, per-token running cache across the
                # whole prefill+decode lifetime (spec section 4) -- keeping the
                # scorer itself on CPU forced a GPU<->CPU round trip on literally
                # every layer/step (measured: ~35ms/call, the dominant eviction
                # cost). Put it on the model's own device instead.
                rfc_device = next(model.parameters()).device
                if cfg.rfc_objective == "oracle":
                    head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // model.config.num_attention_heads)
                    self.unified_rfc_scorer = OracleFutureScorer(horizon=256, lambda_mix=float(cfg.rfc_lambda))
                    self._oracle_scaling = float(head_dim) ** -0.5
                elif cfg.rfc_objective == "RESCUE":
                    # Reframed objective: predict C_i = P(Recent-only wrongly
                    # evicts this candidate that Oracle would keep), not raw
                    # future importance -- see rescue.scorer
                    # and rescue_train_export.py. Use with
                    # --rfc-combine-mode share_additive (C_i is already a
                    # well-scaled [0,1] probability).
                    from rescue.scorer import RescueScorer

                    self.unified_rfc_scorer = RescueScorer(cfg.learned_checkpoint, device=rfc_device, lambda_mix=float(cfg.rfc_lambda))
                else:
                    raise ValueError(
                        f"unsupported rfc_objective {cfg.rfc_objective!r}; this "
                        "release implements 'RESCUE' and the 'oracle' ceiling"
                    )
            else:
                self.learned_scorer = LearnedScorer(cfg.learned_checkpoint, device="cpu")
        self.layer_head_gain = None
        self.ragged_decoder = RaggedLlamaLikeDecoder(model) if (
            (cfg.normalized_policy() == "laprox" and cfg.laprox_allocation in {"global_layer", "global_head"})
            or (cfg.normalized_policy() == "rfc" and cfg.rfc_allocation == "global_layer")
            or cfg.normalized_policy() in {"lava", "rkv"}
            # RESCUE on a ragged base runs that base's own state class, so it
            # needs the ragged decode path too -- without this the decode step
            # hands HF a RaggedKVCache it cannot read.
            or (cfg.normalized_policy() == "rfc" and cfg.rfc_recent_style in {"lava", "rkv"})
        ) else None
        self.last_stats = {}
        self.stats_records: list[dict] = []
        self._pristine_cfg = cfg   # fidelity selection rewrites self.cfg per document

    def _select_lambda_by_fidelity(self, dense_past, input_ids, seq_len, device,
                                   build_fn, first_logits, prefill_logits=None):
        """Choose lambda by measuring what each pruned cache does to the model's
        own next-token distribution, instead of predicting it from the prompt.

        Every earlier selector had to decide BEFORE evicting: the entropy gate
        reads the prompt (so it takes the same value whichever base sits
        underneath, yet on trec the right call is opposite per base), and the
        base-score statistics that do vary per base scored the majority-class
        rate. This one evicts first and reads the damage.

        Decode PROBE_LEN tokens against the unpruned cache -- the tokens the
        uncompressed model emits. They are DISCARDED, not emitted: keeping them
        would hand this arm PROBE_LEN dense-quality tokens that no baseline
        gets, which would flatter the comparison rather than test the selector.
        Then, for each candidate lambda, prune a copy of the prompt cache and
        re-run those same tokens against it, and score

            KL( p_dense(.) || p_pruned(.) )   averaged over the probe tokens.

        The smallest KL wins. lambda=0 is always a candidate and under
        scale_additive it is the untouched base, so a document the correction
        would damage keeps the baseline by construction -- the property every
        previous gate had to predict and could not.

        Returns (chosen_lambda, {lambda: KL}, a fresh copy of the dense cache).
        self.cfg is left pointing at the chosen lambda with both gates disabled,
        so the caller's own build runs at that lambda.
        """
        from dataclasses import replace as _dc_replace

        from transformers import DynamicCache

        base_cfg = self._pristine_cfg
        lams = [float(x) for x in str(base_cfg.rfc_fidelity_lambdas).split(",") if x.strip()]
        probe_len = max(1, int(base_cfg.rfc_fidelity_probe_len))
        eos = self.tokenizer.eos_token_id

        def _clone(cache):
            """A new CONTAINER over the SAME tensors -- not a copy of the cache.

            Every write path in this file replaces a list entry with a freshly
            allocated tensor (prune_head and _prune_* via index_select,
            append_layer and _append_layer_cache via cat); nothing writes into a
            K/V tensor in place, and there is no copy_/add_/scatter_ on cache
            storage anywhere. So a trial can share the prompt's tensors: it
            swaps entries in its own list while the original list still holds
            the full ones.

            Copying instead cost two things. Memory: a full prompt cache per
            trial on top of the real one, which is what ran the card out twice
            today -- and it raised the peak of a method whose entire purpose is
            to lower it, since every evict-once baseline already materialises
            this cache once and no more. Numerics: clone() returns contiguous
            tensors, so the selected path handed the pruner a different memory
            layout than the un-selected path would have, a different matmul
            kernel follows, and under greedy decoding a last-bit score
            difference changes which tokens survive. R-KV's lambda=1 documents
            reproduced the gate run only 49 times in 150 under copying."""
            new = DynamicCache()
            for i, (k, v) in enumerate(legacy_layers(cache)):
                new.update(k, v, i, {})
            return new

        def _set_lambda(lam):
            """Two paths read lambda from two different places: the ragged
            bases' RescueInjection reads cfg.rfc_lambda, while the dense path
            reads the SCORER's lambda_mix (fixed at generator construction).
            Both have to move or the trials are all the same eviction."""
            keep_gate = bool(getattr(base_cfg, "rfc_fidelity_with_gate", False))
            self.cfg = _dc_replace(
                base_cfg, rfc_lambda=lam,
                rfc_ppl_gate_tau=(base_cfg.rfc_ppl_gate_tau if keep_gate else 0.0))
            for sc in (self.rfc_scorer, self.unified_rfc_scorer):
                if sc is not None and hasattr(sc, "lambda_mix"):
                    sc.lambda_mix = lam

        # ---- the probe: the dense continuation and its distributions --------
        def _step(cache, token, pos, uniform=False):
            """One decode step.

            The ragged decoder walks the 32 layers in a Python loop, which is
            what the pruned caches need -- global_layer and the ragged bases
            leave every layer a different length and HF's own forward cannot
            mask that. The PROBE does not need it: it runs before any eviction,
            so the cache is still rectangular. Routing the probe through the
            ragged path anyway cost LaProx most of its overhead (+176% against
            SnapKV's +14% for the same algorithm, the two differing only in
            which decode path their allocation forces).
            """
            if self.ragged_decoder is not None and not uniform:
                logits, cache, _, _ = self.ragged_decoder.decode_one(
                    token, cache, absolute_pos=pos, output_attentions=False)
                return logits, cache
            out = self.model(
                input_ids=token, past_key_values=cache,
                position_ids=torch.tensor([[pos]], dtype=torch.long, device=device),
                use_cache=True, return_dict=True)
            return out.logits, out.past_key_values

        def _drop_tail(cache, w, start_pos):
            """Remove the last w prompt positions from a pruned cache.

            The tail is the observation window, which every policy protects, so
            it is present in the pruned cache and would otherwise let the probe
            queries attend to themselves. Dropping it leaves exactly the cache
            those queries would face: sink + whatever this lambda kept."""
            if isinstance(cache, RaggedKVCache):
                for li in range(len(cache)):
                    for h in range(cache.layer_head_count(li)):
                        pos = cache.positions[li][h]
                        keep = (pos.to(pos.device) < start_pos).nonzero(as_tuple=True)[0]
                        cache.prune_head(li, h, keep)
                return cache
            for li, (k, v) in enumerate(legacy_layers(cache)):
                if k is None or int(k.shape[2]) <= w:
                    continue
                cache = _set_layer_cache(cache, li, k[:, :, :-w, :].contiguous(),
                                         v[:, :, :-w, :].contiguous())
            return cache

        probe_mode = str(getattr(base_cfg, "rfc_fidelity_probe", "gen") or "gen")
        if probe_mode == "prompt_tail" and prefill_logits is not None \
                and int(prefill_logits.shape[1]) >= 2:
            # The free variant. Prefill already produced a next-token
            # distribution for each of the last W prompt positions -- the same
            # logits the entropy gate reads -- so the dense reference costs
            # nothing and no token has to be decoded before evicting. The
            # method's three real costs (extra decode work, delayed
            # compression, requests that finish before the probe does) all come
            # from generating the probe, and all three disappear here.
            #
            # What it risks: those are the very queries every base score ranks
            # on, so top-B(base) is close to optimal for them by construction,
            # while the correction is trained against FUTURE queries. A judge
            # that favours lambda=0 would be the failure mode to look for.
            W = min(int(prefill_logits.shape[1]), seq_len - 1)
            start = seq_len - W
            dense_logp = [torch.log_softmax(prefill_logits[0, j].float(), dim=-1)
                          for j in range(W)]
            probe_ids = input_ids[0, start:seq_len].tolist()
            kls = {}
            trial_log = []
            for lam in lams:
                _set_lambda(lam)
                trial_state, pruned = build_fn(_clone(dense_past))
                pruned = _drop_tail(pruned, W, start)
                total = 0.0
                for j, tid in enumerate(probe_ids):
                    cur = torch.tensor([[tid]], dtype=torch.long, device=device)
                    logits, pruned = _step(pruned, cur, start + j)
                    lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                    total += float((dense_logp[j].exp() * (dense_logp[j] - lp)).sum())
                kls[lam] = total / max(1, len(probe_ids))
                trial_log.append((lam, kls[lam]))
                del trial_state, pruned
                torch.cuda.empty_cache()
            best = min(lams, key=lambda l: kls[l])
            _set_lambda(best)
            if os.environ.get("RESCUE_GATE_LOG"):
                print("[FIDELITY/tail] W=" + str(W) + " "
                      + "  ".join(f"lam{l}={k:.3e}" for l, k in trial_log)
                      + f"  -> {best}", flush=True)
            return best, kls, dense_past, None

        def _drop_probe_tail(cache, n):
            """Remove the n probe tokens a trial appended while being scored.

            The trial already built exactly the cache the final build would
            rebuild -- verified by logging the kept sets of both and finding
            them identical on every layer. Rebuilding it costs a second full
            correction pass over every candidate in every layer, which is the
            single most expensive part of the selector."""
            if n <= 0:
                return cache
            if isinstance(cache, RaggedKVCache):
                for li in range(len(cache)):
                    for h in range(cache.layer_head_count(li)):
                        L = int(cache.keys[li][h].shape[2])
                        if L <= n:
                            continue
                        keep = torch.arange(L - n, device=cache.keys[li][h].device)
                        cache.prune_head(li, h, keep)
                return cache
            for li, (k, v) in enumerate(legacy_layers(cache)):
                if k is None or int(k.shape[2]) <= n:
                    continue
                cache = _set_layer_cache(cache, li, k[:, :, :-n, :].contiguous(),
                                         v[:, :, :-n, :].contiguous())
            return cache

        # --- cache-selection control (reviewer item 2a) ----------------------
        # RESCUE_CACHE_SELECT="lookaheadkv,snapkv" turns this selector into a
        # chooser between two POLICIES' caches instead of two lambdas, with the
        # same one-token probe and the same KL. It answers whether the value is
        # in the residual scorer or merely in having a second cache to judge:
        # if picking between an observed and a predicted-future policy scores
        # as well, the scorer is not what the selector needs.
        # Run it with --method lookaheadkv and its checkpoint, so the learned
        # scorer that arm needs is loaded; SnapKV needs none.
        _cand_policies = [c.strip() for c in
                          os.environ.get("RESCUE_CACHE_SELECT", "").split(",") if c.strip()]

        def _set_policy(name):
            self.cfg = _dc_replace(base_cfg, policy=name, rfc_ppl_gate_tau=0.0)

        if _cand_policies:
            cands, apply_cand = _cand_policies, _set_policy
        else:
            cands, apply_cand = lams, _set_lambda

        _tmz = bool(os.environ.get("RESCUE_TIMING"))
        _dev = dense_past if False else None
        def _nw():
            if _tmz:
                torch.cuda.synchronize()
            return time.perf_counter()
        _memlog("selector entry", seq_len)
        _tp0 = _nw()
        # The KL is a sum over the whole vocabulary of p*(log p - log q). In
        # float32 that reduction carries an absolute error around 1e-7, and on
        # Qwen3-8B the true KL falls under it on a large share of documents
        # (62% of PassageRetrieval, 43% of MultiFieldQA-en against 0% and 2% for
        # Llama), so min() over the candidates was reading rounding noise and
        # accepting the correction about half the time. float64 costs one
        # log_softmax per trial and removes the artifact.
        _kl_dtype = torch.float64 if os.environ.get("RESCUE_KL_FLOAT64") else torch.float32
        probe_ids: list[int] = []
        dense_logp: list[torch.Tensor] = []
        work = _clone(dense_past)
        nxt = torch.argmax(first_logits, dim=-1, keepdim=True)
        pos = seq_len
        for _ in range(probe_len):
            tid = int(nxt[0, 0].item())
            if eos is not None and tid == int(eos):
                break
            probe_ids.append(tid)
            _memlog("before probe _step", seq_len)
            logits, work = _step(work, nxt, pos, uniform=True)
            dense_logp.append(torch.log_softmax(logits[:, -1, :].to(_kl_dtype), dim=-1))
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            pos += 1
        del work
        if not probe_ids:
            # the dense model stops immediately; nothing to measure, and with
            # nothing generated the choice cannot matter
            _fallback = cands[0] if _cand_policies else float(base_cfg.rfc_lambda)
            apply_cand(_fallback)
            return _fallback, {}, dense_past, None

        _t_probe = _nw() - _tp0
        _tt0 = _nw()
        # ---- score each lambda on that probe --------------------------------
        kls: dict[float, float] = {}
        trial_log: list[tuple[float, float]] = []
        trials: dict[float, tuple] = {}
        for lam in cands:
            apply_cand(lam)
            trial_state, pruned = build_fn(_clone(dense_past))
            total, pos = 0.0, seq_len
            for j, tid in enumerate(probe_ids):
                cur = torch.tensor([[tid]], dtype=torch.long, device=device)
                logits, pruned = _step(pruned, cur, pos)
                lp = torch.log_softmax(logits[:, -1, :].to(_kl_dtype), dim=-1)
                total += float((dense_logp[j].exp() * (dense_logp[j] - lp)).sum())
                pos += 1
            kls[lam] = total / len(probe_ids)
            trial_log.append((lam, kls[lam]))
            trials[lam] = (trial_state, pruned)
            # empty_cache() synchronises and hands every cached block back to
            # the driver, so the next allocation re-maps: measured at 150-250 ms
            # per document, a quarter of the whole selector. It is only needed
            # where a trial's working set is genuinely large -- R-KV builds an
            # [n, n] redundancy block per layer and ran a card out without it
            # (a 2.16 GiB allocation refused with 985 MiB free). Everywhere else
            # the allocator handles reuse on its own.
            if str(getattr(base_cfg, "rfc_recent_style", "")) in ("rkv", "lava"):
                torch.cuda.empty_cache()
        best = min(cands, key=lambda l: kls[l])
        # Abstention: if every candidate's KL sits under the floor, the probe
        # produced no evidence, and the safe reading of no evidence is to keep
        # the base cache rather than to let the ordering of two noise values
        # decide. RESCUE_KL_FLOOR=0 disables it; the base candidate is the
        # smallest lambda. The default is on: leaving it off made Qwen3-8B lose
        # 6.00 on PassageRetrieval (97.00 -> 91.00) purely on coin flips, and it
        # is a no-op wherever the KL is informative -- Llama's worst cell has
        # 5.5% degenerate documents and its scores do not move.
        _kl_floor = float(os.environ.get("RESCUE_KL_FLOOR", "1e-6") or 0)
        if (_kl_floor > 0 and all(isinstance(l, (int, float)) for l in cands)
                and max(abs(kls[l]) for l in cands) < _kl_floor):
            # cands are lambdas here; the cache-selection control puts policy
            # NAMES in this list, where "the smallest" means nothing.
            best = min(cands)
            if os.environ.get("RESCUE_GATE_LOG"):
                print("[FIDELITY/abstain] all KL < "
                      + f"{_kl_floor:g} -> {best}", flush=True)
        if _tmz:
            print(f"[SELTIME] probe={1000*_t_probe:.1f} trials={1000*(_nw()-_tt0):.1f} "
                  f"steps={len(probe_ids)} cands={len(cands)}", flush=True)
        apply_cand(best)
        # hand back the ORIGINAL cache, not a copy of it. Nothing above mutates
        # dense_past -- the probe and every trial ran on their own clone -- so a
        # further clone only costs a second full-prompt cache at peak (the one
        # OOM seen so far), and it hands the caller tensors with a different
        # memory layout than the un-selected path would have used. Same values,
        # but a different layout can select a different matmul kernel, and under
        # greedy decoding a last-bit difference in the scores is enough to
        # change which tokens survive: R-KV's lambda=1 documents reproduced the
        # gate run only 49 times in 150, while LaProx (159/159) and SnapKV
        # (164/164) reproduced it exactly.
        if os.environ.get("RESCUE_GATE_LOG"):
            print("[FIDELITY] probe=" + str(len(probe_ids)) + " "
                  + "  ".join(f"lam{l}={k:.3e}" for l, k in trial_log)
                  + f"  -> {best}", flush=True)
        won = trials.get(best)
        if won is not None:
            st_, pruned_ = won
            return best, kls, dense_past, (st_, _drop_probe_tail(pruned_, len(probe_ids)))
        return best, kls, dense_past, None

    @torch.inference_mode()
    def generate_one(self, prompt: str, max_new_tokens: int, max_seq_len: int) -> str:
        device = next(self.model.parameters()).device
        cuda_device = device if device.type == "cuda" else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        policy = self.cfg.normalized_policy()
        input_budget = max(1, int(max_seq_len) - int(max_new_tokens))
        if bool(self.cfg.use_chat_template) and hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"]
        if int(input_ids.shape[1]) > input_budget:
            half = input_budget // 2
            tail = input_budget - half
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -tail:]], dim=1)
        input_ids = input_ids.to(device)
        seq_len = int(input_ids.shape[1])
        # ADDED-LOGIC TIMING. Wall-clock per request is not comparable across
        # methods -- it is dominated by how many tokens each happens to emit
        # (ForesightKV's collapsed checkpoint writes 82 words where SnapKV
        # writes 29, and looks 2.5x slower for that reason alone). What IS
        # comparable is what each method's own machinery costs on top of the
        # same prefill, so time the pieces separately.
        _tm = bool(os.environ.get("RESCUE_TIMING"))
        def _now():
            if _tm and cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            return time.perf_counter()
        _t_prefill0 = _now()
        needs_prefill_attention = policy in {"h2o", "snapkv", "laprox", "foresightkv", "lookaheadkv", "rfc", "lava"}
        use_obs_window_prefill = (
            policy in {"snapkv", "laprox", "foresightkv", "lookaheadkv", "rfc", "lava", "rkv"}
            and seq_len > int(self.cfg.snapkv_obs_window)
        )
        # HF returns one [1, seq, hidden] tensor PER LAYER, so asking for hidden
        # states costs num_layers * seq * hidden * 2 bytes -- 16 GiB for a 64K
        # narrativeqa document on this model, and the prefix pass pays it a
        # second time. Only two consumers exist: the residual scorer (via
        # _populate_z_cache_from_hidden_states / append_generated_position, both
        # of which no-op when self.rfc_scorer is None) and learned_scorer's
        # q_group. RESCUE runs on unified_rfc_scorer and reads neither, so it
        # was paying 32 GiB per long document for a tensor it never touched --
        # measured as the single largest allocation when H2O+RESCUE OOM'd on
        # narrativeqa.
        need_hidden_states = (
            policy == "lookaheadkv"
            or (policy == "rfc" and (self.rfc_scorer is not None
                                     or self.learned_scorer is not None))
        )
        rfc_prefill_queries: dict[int, torch.Tensor] = {}
        precomputed_h2o_scores = None
        past_h2o_cache = None   # rfc + h2o base: the cache --method h2o would build
        prompt_entropy = None   # set by the obs-window branch when the ppl gate is on
        prefix = None
        if policy == "h2o":
            past_h2o, last_logits_h2o, precomputed_h2o_scores = _h2o_chunked_prefill(self.model, input_ids, device)
            _memlog("after h2o_prefill(base)", seq_len)
            prefill = SimpleNamespace(past_key_values=past_h2o, logits=last_logits_h2o, attentions=None, hidden_states=None)
        elif use_obs_window_prefill:
            if policy == "rfc" and self.cfg.rfc_recent_style == "h2o":
                # RESCUE ported onto an H2O base (Framing B: portable
                # correction module) needs H2O's OWN real cumulative-attention
                # score as s_recent -- the obs-window 2-pass below can't
                # produce this (its attentions are restricted to the last
                # snapkv_obs_window queries only), so run the expensive
                # chunked full-attention prefill FIRST just to harvest
                # precomputed_h2o_scores. Its own past_key_values/logits are
                # discarded -- the obs-window pass below remains the actual
                # prefill result used downstream, so RESCUE's own feature
                # computation (needs q_ring populated from the observation
                # window) is unaffected. Yes, this means two full prefill
                # passes for this specific combo -- accepted cost for a
                # correct (not window-truncated) H2O base score.
                # Keep this pass's CACHE too, not just the scores. Discarding it
                # and using the obs-window pass's cache instead left rfc_lambda=0
                # scoring 0.38 below --method h2o on both qasper and
                # multifieldqa_en, even though the two retain byte-identical
                # token sets (verified 96/96 layer-documents): same policy, but
                # the caches are built by different attention kernels and chunk
                # splits (this one eager+output_attentions, the obs path sdpa for
                # the prefix), so their K/V differ in the last bits and the
                # generation diverges. Adopting this cache makes lambda=0 the
                # same computation --method h2o runs, the way it already is for
                # laprox and snapkv.
                past_h2o_cache, _, precomputed_h2o_scores = _h2o_chunked_prefill(self.model, input_ids, device)
                _memlog("after h2o_chunked_prefill", seq_len)
            obs = max(1, min(int(self.cfg.snapkv_obs_window), seq_len))
            prefix_ids = input_ids[:, :-obs]
            obs_ids = input_ids[:, -obs:]
            # This prefix pass needs no attention weights back (output_attentions=
            # False) and can be arbitrarily long once max_seq_len isn't truncating
            # documents down to a few thousand tokens -- but the model's config is
            # globally pinned to attn_implementation="eager" (needed for the obs
            # window call right below, which DOES need weights), and every layer
            # re-reads config._attn_implementation on every forward call rather
            # than caching it at construction, so eager wouldn't otherwise be
            # avoidable here. Materializing eager's full [heads, seq, seq]
            # attention matrix for a 20k+ token prefix OOMs (measured: 36GB for a
            # single layer's softmax on a ~17k-token qasper document). Swap to
            # sdpa (fused, no full materialization) just for this call.
            self.model.config._attn_implementation = "sdpa"
            try:
                prefix = self.model(
                    input_ids=prefix_ids,
                    attention_mask=torch.ones_like(prefix_ids, device=device),
                    use_cache=True,
                    output_attentions=False,
                    # "rfc" needs hidden_states for EVERY currently-cached prompt
                    # position (spec section 4's per-token cache), not just the
                    # observation window below -- unlike lookaheadkv, which only
                    # reads hidden_states from the final forward.
                    output_hidden_states=policy == "rfc",
                    return_dict=True,
                )
            finally:
                self.model.config._attn_implementation = "eager"
            # PROMPT PREDICTABILITY, for the lambda gate. This prefix pass has
            # already produced a next-token distribution for (almost) every
            # prompt position, so the mean entropy of those distributions is
            # free -- no extra forward, nothing to load. It is a property of
            # the TEXT, not a task label.
            #
            # Why this and not a statistic of the correction: measured today,
            # whether lambda helps is decided by the strength of the BASE, not
            # by how precise the correction is (trec has the worst precision of
            # three tasks, 0.150, and still gains +6.50; lcc sits between at
            # 0.197 and loses -4.11). S_recent here is LaProx, a locality
            # score, so it is already near-optimal exactly where the next token
            # follows from nearby context -- which is what entropy measures.
            # Per-document over 5 tasks x 20 documents this separates them
            # completely: lcc spans 0.413-1.289 and every gaining task's
            # documents sit above (gov_report 1.317, qasper 1.414, trec 1.777),
            # an empty band no threshold has to be tuned inside.
            if float(getattr(self.cfg, "rfc_ppl_gate_tau", 0.0) or 0.0) > 0.0:
                lg = getattr(prefix, "logits", None)
                if lg is not None and lg.shape[1] > 0:
                    with torch.no_grad():
                        n_pos = int(lg.shape[1])
                        step = max(1, n_pos // 4096)   # a mean needs a sample, not every position
                        tot, cnt = 0.0, 0
                        for s in range(0, n_pos, 512 * step):
                            sl = lg[0, s:s + 512 * step:step].float()
                            lp = torch.log_softmax(sl, dim=-1)
                            tot += float(-(lp.exp() * lp).sum(dim=-1).sum())
                            cnt += int(sl.shape[0])
                            del sl, lp
                        prompt_entropy = tot / max(1, cnt)
                    if os.environ.get("RESCUE_GATE_LOG"):
                        print(f"[PPLGATE] n_pos={n_pos} step={step} "
                              f"prompt_entropy={prompt_entropy:.4f}", flush=True)
            # prefix.logits is [1, seq_len - obs, vocab] -- 15 GiB at 64K tokens
            # on a 128K-vocab model. Nothing downstream reads it except the
            # perplexity gate handled just above, and that gate is off by
            # default, so drop the reference before the obs pass allocates on
            # top of it. past_key_values is what the next call actually needs.
            if float(getattr(self.cfg, "rfc_ppl_gate_tau", 0.0) or 0.0) <= 0.0:
                try:
                    prefix.logits = None
                except Exception:
                    pass

            position_ids = torch.arange(seq_len - obs, seq_len, dtype=torch.long, device=device).view(1, -1)
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

            def _obs_prefill_call():
                return self.model(
                    input_ids=obs_ids,
                    attention_mask=attention_mask,
                    past_key_values=prefix.past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=need_hidden_states,
                    return_dict=True,
                )

            if policy in ("rfc", "rkv"):
                # obs (>= rfc_scorer.recent_window / rkv_window_size in every
                # configuration this project uses) always covers the initial
                # recent-query window on its own -- the prefix pass's queries
                # are never needed for q_cache, so only this call is worth
                # capturing.
                prefill, rfc_prefill_queries = _call_with_query_capture(self.model, _obs_prefill_call)
            else:
                prefill = _obs_prefill_call()
        else:
            def _plain_prefill_call():
                return self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids, device=device),
                    use_cache=True,
                    output_attentions=needs_prefill_attention,
                    output_hidden_states=need_hidden_states,
                    return_dict=True,
                )

            # This branch serves policies that don't need attention weights
            # from the prefill at all (full, kvp, streamingllm; rfc only lands
            # here for seq_len <= snapkv_obs_window, tiny) --
            # needs_prefill_attention is already False for them, but the
            # model's config is still globally pinned to attn_implementation=
            # "eager" (other policies need it), so eager's full [heads, seq,
            # seq] softmax still gets computed and OOMs once documents aren't
            # truncated down to a few thousand tokens (same issue as the
            # use_obs_window_prefill branch above, fixed there already).
            if not needs_prefill_attention:
                self.model.config._attn_implementation = "sdpa"
            try:
                if policy in ("rfc", "rkv"):
                    prefill, rfc_prefill_queries = _call_with_query_capture(self.model, _plain_prefill_call)
                else:
                    prefill = _plain_prefill_call()
            finally:
                self.model.config._attn_implementation = "eager"
        past = prefill.past_key_values
        _t_prefill = _now() - _t_prefill0
        if policy == "rfc" and self.cfg.rfc_recent_style == "h2o" and past_h2o_cache is not None:
            # The obs-window pass still ran, for its queries (q_ring) and its
            # logits (the perplexity gate); only its cache is replaced.
            del past
            past = past_h2o_cache
            past_h2o_cache = None
        full_hidden_states = getattr(prefill, "hidden_states", None)
        if policy == "rfc" and prefix is not None and full_hidden_states is not None:
            # Two separate forward calls each only cover the tokens THEY
            # processed; concatenate per layer along the sequence dim so
            # z_cache can be built for the whole prompt at once.
            prefix_hidden = getattr(prefix, "hidden_states", None)
            if prefix_hidden is not None:
                full_hidden_states = tuple(
                    torch.cat([ph, fh], dim=1) for ph, fh in zip(prefix_hidden, full_hidden_states)
                )
        # RESCUE on a ragged base: run that base's OWN state class (so its budget
        # allocation is the method's, not a reimplementation) and hand it the
        # correction to fold into its score. rfc_recent_style names the base.
        ragged_base = self.cfg.rfc_recent_style if policy == "rfc" else None
        if ragged_base not in ("lava", "rkv"):
            ragged_base = None

        # --- training-free future-signal control (RESCUE_PROBE_SNAPKV=N) ------
        # Decode N tokens off the DENSE prompt cache and keep their attention
        # over the prompt. That is the same probe RESCUE's selector pays for,
        # minus the scorer and minus the two candidate caches. Feeding it into
        # SnapKV's own vote is the "do you need the scorer at all?" baseline.
        # The decode runs on a container-level clone, so the real cache is
        # untouched and the emitted tokens are discarded -- this arm must not
        # get dense-quality output tokens no other arm gets.
        _probe_attn = None
        _n_probe = int(os.environ.get("RESCUE_PROBE_SNAPKV", "0") or 0)
        _probe_recent_w = float(os.environ.get("RESCUE_PROBE_IN_RECENT", "0") or 0)
        if _n_probe > 0 and (policy == "snapkv"
                             or (_probe_recent_w != 0.0 and ragged_base is None
                                 and policy == "rfc"
                                 and self.cfg.rfc_recent_style == "snapkv")):
            from transformers import DynamicCache as _DC
            _pc = _DC()
            for _i, (_k, _v) in enumerate(legacy_layers(past)):
                _pc.update(_k, _v, _i, {})
            _tok = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
            _pos = seq_len
            _acc = None
            with torch.no_grad():
                for _ in range(_n_probe):
                    _o = self.model(input_ids=_tok, past_key_values=_pc,
                                    position_ids=torch.tensor([[_pos]], device=device),
                                    use_cache=True, output_attentions=True, return_dict=True)
                    _pc = _o.past_key_values
                    # keep only the columns over the ORIGINAL prompt
                    _rows = [a[:, :, :, :seq_len].detach() for a in _o.attentions]
                    _acc = _rows if _acc is None else [torch.cat([x, y], dim=2) for x, y in zip(_acc, _rows)]
                    _tok = torch.argmax(_o.logits[:, -1, :], dim=-1, keepdim=True)
                    _pos += 1
            _probe_attn = _acc
            del _pc
        # "probe + RESCUE": the same probe rows also join RESCUE's own s_recent,
        # so the combined arm is the control's cache with the scorer on top
        # rather than two independent corrections of the base.
        if _probe_attn is not None and _probe_recent_w != 0.0:
            from rescue.policies.scoring import set_probe_rows
            set_probe_rows(_probe_attn, _probe_recent_w)

        def _build_state_and_prune(past_in):
            """Build the policy's state object and run its one-shot prefill
            eviction on past_in.

            This used to be a straight line here. It is a function so the
            fidelity selector below can run it once per candidate lambda --
            every branch reads self.cfg, which the selector swaps between
            calls, so the lambda reaches the scorer without threading an
            override through five state classes.
            """
            if policy == "laprox" and self.cfg.laprox_allocation == "global_head":
                st = RaggedLaProxState(self.cfg, self.model)
                return st, st.initialize_after_prefill(past_in, prefill.attentions, seq_len)
            if policy == "lava" or ragged_base == "lava":
                st = RaggedLaVaState(self.cfg, self.model)
                if ragged_base == "lava":
                    st.rescue = RescueInjection(
                        self.cfg, self.model, self.unified_rfc_scorer or self.rfc_scorer,
                        rfc_prefill_queries or {}, prompt_entropy)
                return st, st.initialize_after_prefill(past_in, prefill.attentions, seq_len)
            if policy == "rkv" or ragged_base == "rkv":
                st = RaggedRKVState(self.cfg, self.model)
                if ragged_base == "rkv":
                    st.rescue = RescueInjection(
                        self.cfg, self.model, self.unified_rfc_scorer or self.rfc_scorer,
                        rfc_prefill_queries or {}, prompt_entropy)
                return st, st.initialize_after_prefill(past_in, rfc_prefill_queries or {}, seq_len)
            st = DynamicEvictionState(self.cfg, self.model, learned_scorer=self.learned_scorer)
            st.kvp_paper_scorer = self.kvp_paper_scorer
            st.foresight_paper_scorer = self.foresight_paper_scorer
            st.rfc_scorer = self.rfc_scorer
            st.unified_rfc_scorer = self.unified_rfc_scorer
            st.external_future_scorer = self.external_future_scorer
            st.external_future_kind = self.external_future_kind
            st.layer_head_gain = self.layer_head_gain
            st.prompt_entropy = prompt_entropy
            if policy == "rfc" and rfc_prefill_queries:
                st._populate_q_cache(rfc_prefill_queries, self.model.config.num_key_value_heads)
            if (policy == "rfc" and self.cfg.rfc_objective == "RESCUE") or policy == "laprox":
                # Llama-3's tokenizer places every chat-template/control token
                # (<|begin_of_text|>, <|start_header_id|>, <|end_header_id|>,
                # <|eot_id|>, reserved slots, ...) contiguously at/above
                # vocab_size -- see rescue_eviction_inspect.py's finding.
                special_id_threshold = int(self.tokenizer.vocab_size)
                prompt_ids_flat = input_ids[0].tolist()
                st.special_positions = {
                    i for i, tid in enumerate(prompt_ids_flat) if tid >= special_id_threshold
                }
                if os.environ.get("RESCUE_DEBUG_SPECIAL"):
                    print(f"[RESCUE_DEBUG] special_positions n={len(st.special_positions)} sample={sorted(st.special_positions)[:10]}", flush=True)
            return st, st.initialize_after_prefill(
                past_in, prefill.attentions, hidden_states=full_hidden_states,
                precomputed_h2o_scores=precomputed_h2o_scores,
                probe_attentions=_probe_attn,
            )

        chosen_lambda = None
        fidelity_kls = None
        reuse = None   # set only when the fidelity selector hands back its winning trial
        _t_select = 0.0
        # RESCUE_CACHE_SELECT turns the same probe+KL selector into a chooser
        # between two policies' caches, which is a control on the scorer rather
        # than an arm of RESCUE, so it must fire for a non-rfc base policy too.
        # LookaheadKV cannot be a candidate: it has its own decoder path and
        # never reaches this eviction machinery (see the rfc_objective comment
        # above), so the predicted-future side of the comparison is ForesightKV.
        if (policy == "rfc" and str(getattr(self.cfg, "rfc_lambda_select", "") or "") == "fidelity") \
                or os.environ.get("RESCUE_CACHE_SELECT"):
            _t0 = _now()
            chosen_lambda, fidelity_kls, past, reuse = self._select_lambda_by_fidelity(
                past, input_ids, seq_len, device, _build_state_and_prune,
                prefill.logits[:, -1, :], getattr(prefill, "logits", None))
            _t_select = _now() - _t0
        _t_evict0 = _now()
        if reuse is not None:
            state, past = reuse
        else:
            state, past = _build_state_and_prune(past)
        _t_evict = _now() - _t_evict0
        if _probe_attn is not None and _probe_recent_w != 0.0:
            from rescue.policies.scoring import clear_probe_rows
            clear_probe_rows()
        if _tm:
            print(f"[TIMING] len={seq_len} prefill={1000*_t_prefill:.1f} "
                  f"select={1000*_t_select:.1f} evict={1000*_t_evict:.1f}", flush=True)
        # Start of decode. Everything before this is prompt processing plus the
        # method's own eviction/selection; everything after is generation on the
        # compressed cache. Splitting here is what lets the fixed eviction cost be
        # amortised against generation length instead of reported on its own.
        if _tm and cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        _t_decode0 = _now()
        prefill_peak = torch.cuda.max_memory_allocated(cuda_device) if cuda_device is not None else None
        prefill_cache_bytes = _cache_bytes(past)
        prefill_cache_lengths = _cache_lengths(past)
        prefill_cache_lengths_by_head = _cache_head_lengths(past)
        prefill_kv_slots_total = _cache_kv_slots_total(past)
        generated: List[int] = []
        next_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        eos = self.tokenizer.eos_token_id
        absolute_pos = seq_len
        for step in range(int(max_new_tokens)):
            token_id = int(next_token[0, 0].item())
            if step > 0 or token_id != eos:
                generated.append(token_id)
            if eos is not None and token_id == int(eos):
                break
            position_ids = torch.tensor([[absolute_pos]], dtype=torch.long, device=device)
            if self.ragged_decoder is not None:
                logits, past, attentions, decode_queries = self.ragged_decoder.decode_one(
                    next_token,
                    past,
                    absolute_pos=absolute_pos,
                    output_attentions=bool(self.cfg.evict_during_decode),
                )
                if policy == "rfc" and hasattr(state, "_populate_q_cache"):
                    state._populate_q_cache(decode_queries, self.model.config.num_key_value_heads)
                elif policy == "rkv" or (policy == "rfc" and self.cfg.rfc_recent_style == "rkv"):
                    # R-KV keeps its own short query window AND, when RESCUE is
                    # bolted on, the correction's longer one.
                    state.push_query(decode_queries)
                    if getattr(state, "rescue", None) is not None:
                        state.rescue.push_queries(decode_queries)
                elif policy == "rfc" and getattr(state, "rescue", None) is not None:
                    state.rescue.push_queries(decode_queries)
            else:
                # Under evict-once nothing after prefill reads the decode-time
                # hidden states or the query ring buffer: decode_hidden_states is
                # consumed only inside the evict_during_decode branch below, and
                # _populate_q_cache exists to feed the NEXT eviction, which never
                # comes. Returning 32 layers of hidden states and capturing every
                # query per token cost a measured 2.90 ms/token against the same
                # base policy at the same cache size, on 100% of documents.
                _rfc_decode_work = policy == "rfc" and bool(self.cfg.evict_during_decode)

                def _decode_call():
                    return self.model(
                        input_ids=next_token,
                        past_key_values=past,
                        position_ids=position_ids,
                        use_cache=True,
                        output_attentions=bool(self.cfg.evict_during_decode) and policy in {"h2o", "laprox", "foresightkv", "rfc"},
                        output_hidden_states=_rfc_decode_work,
                        return_dict=True,
                    )

                if _rfc_decode_work:
                    out, decode_queries = _call_with_query_capture(self.model, _decode_call)
                    state._populate_q_cache(decode_queries, self.model.config.num_key_value_heads)
                else:
                    out = _decode_call()
                past = out.past_key_values
                logits = out.logits
                attentions = out.attentions
            decode_hidden_states = getattr(out, "hidden_states", None) if self.ragged_decoder is None else None
            if bool(self.cfg.evict_during_decode):
                if isinstance(past, RaggedKVCache):
                    past = state.update_and_prune_after_decode(past, attentions)
                else:
                    state.append_generated_position(past, absolute_pos, hidden_states=decode_hidden_states)
                    past = state.update_and_prune_after_decode(past, attentions, hidden_states=decode_hidden_states)
            elif not isinstance(past, RaggedKVCache):
                state.append_generated_position(past, absolute_pos, hidden_states=decode_hidden_states)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            absolute_pos += 1
        self.last_stats = {
            "policy": self.cfg.normalized_policy(),
            "config": asdict(self.cfg),
            # what the fidelity selector decided for THIS document, so a caller
            # can read the margin without parsing the log
            "fidelity_lambda": chosen_lambda,
            "fidelity_kls": fidelity_kls,
            "prefill_tokens": seq_len,
            "retained_tokens": _cache_len(past),
            "retained_tokens_total_layers": _cache_tokens_total(past),
            "retained_tokens_by_layer": _cache_lengths(past),
            "retained_tokens_by_layer_head": _cache_head_lengths(past),
            "retained_kv_slots_total": _cache_kv_slots_total(past),
            "kv_cache_bytes": _cache_bytes(past),
            "prefill_kv_cache_bytes": prefill_cache_bytes,
            "prefill_retained_tokens_by_layer": prefill_cache_lengths,
            "prefill_retained_tokens_by_layer_head": prefill_cache_lengths_by_head,
            "prefill_retained_kv_slots_total": prefill_kv_slots_total,
            "prefill_retained_tokens": state.prefill_retained,
            "evicted_tokens": state.total_evicted,
            "generated_tokens": len(generated),
            "cuda_prefill_peak_allocated_bytes": int(prefill_peak) if prefill_peak is not None else None,
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device is not None else None
            ),
            "cuda_allocated_bytes": (
                int(torch.cuda.memory_allocated(cuda_device)) if cuda_device is not None else None
            ),
        }
        if _tm:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            _t_decode = _now() - _t_decode0
            _t_added = _t_select + _t_evict
            _n_gen = max(1, len(generated))
            print(f"[E2E] len={seq_len} gen={len(generated)} "
                  f"prefill={1000*_t_prefill:.1f} added={1000*_t_added:.1f} "
                  f"decode={1000*_t_decode:.1f} total={1000*(_t_prefill+_t_added+_t_decode):.1f} "
                  f"per_tok={1000*_t_decode/_n_gen:.2f} "
                  f"kv_bytes={_cache_bytes(past)} "
                  f"peak={int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device is not None else -1}",
                  flush=True)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate(self, inputs: list[str], max_new_tokens: int, max_seq_len: int) -> list[str]:
        self.stats_records = []
        outputs = []
        for text in inputs:
            if isinstance(self.unified_rfc_scorer, OracleFutureScorer):
                self._prime_oracle_reference(text, max_new_tokens, max_seq_len)
            outputs.append(self.generate_one(text, max_new_tokens=max_new_tokens, max_seq_len=max_seq_len))
            self.stats_records.append(dict(self.last_stats))
        return outputs

    @torch.inference_mode()
    def _prime_oracle_reference(self, prompt: str, max_new_tokens: int, max_seq_len: int) -> None:
        """Runs a no-eviction reference generation for this SAME prompt (via
        the model's own .generate(), no cache pruning at all) and captures
        real Q/K over (prompt + reference continuation), so
        OracleFutureScorer.score() has a ground-truth future to look up
        during the REAL (budgeted) generation that follows. Must exactly
        replicate generate_one's own tokenization/truncation so its absolute
        positions line up with what OracleFutureScorer will be queried at."""
        device = next(self.model.parameters()).device
        input_budget = max(1, int(max_seq_len) - int(max_new_tokens))
        if bool(self.cfg.use_chat_template) and hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )
            enc = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"]
        if int(input_ids.shape[1]) > input_budget:
            half = input_budget // 2
            tail = input_budget - half
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -tail:]], dim=1)
        input_ids = input_ids.to(device)
        # No eviction happens in this reference pass (cache grows unbounded
        # over up to max_new_tokens decode steps), and this .generate() call
        # never reads attn_weights -- but since the model is loaded globally
        # with attn_implementation="eager", every forward call still
        # materializes full O(seq^2) eager softmax attention regardless of
        # output_attentions, which OOMs on long qasper documents (same class
        # of bug fixed this session in generate_one's "prefix" pass and
        # kv_cache_eviction_hf.py's _generate_lookaheadkv). Swap to sdpa for
        # just this call.
        prev_attn_impl = self.model.config._attn_implementation
        try:
            self.model.config._attn_implementation = "sdpa"
            ref_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=int(max_new_tokens),
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
            )
        finally:
            self.model.config._attn_implementation = prev_attn_impl
        self.unified_rfc_scorer.set_reference(ref_ids, self.model.config.model_type, self.model, self._oracle_scaling)
