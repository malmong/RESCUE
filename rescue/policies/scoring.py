from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


OBSERVED_POLICIES = {"streamingllm", "h2o", "snapkv", "laprox", "lava"}
LEARNED_POLICIES = {"kvp", "foresightkv", "lookaheadkv", "rfc", "rkv"}
ALL_POLICIES = {"full"} | OBSERVED_POLICIES | LEARNED_POLICIES


@dataclass(frozen=True)
class BaselineConfig:
    policy: str
    budget_ratio: float = 0.3
    budget_tokens: int | None = None
    sink_tokens: int = 4
    recent_tokens: int = 64
    # ForesightKV hardcodes a 128-token recency-protected window of its own
    # (reference/official/ForesightKV/.../foresightkv.py:84, `self.length = 128`).
    # At a 128-token TOTAL budget that window consumes the whole budget, the
    # judge gets zero slots and no sink survives -- the method is not runnable
    # as published at this operating point. Lowering this to the repo's shared
    # recent_tokens (64) makes it runnable and MUST be reported as a deviation.
    foresight_recent_window: int = 128
    snapkv_obs_window: int = 32
    snapkv_kernel_size: int = 7
    snapkv_pooling: str = "avgpool"
    laprox_allocation: str = "global_layer"
    evict_during_decode: bool = True
    use_chat_template: bool = False
    learned_checkpoint: str | None = None
    rfc_lambda: float = 1.0
    rfc_recent_weight: float = 1.0
    rfc_use_impact: bool = True
    rfc_impact_mode: str = "learned"  # "learned" (impact_predictor MLP) | "vwo" (non-learned ||V_i@Wo|| norm)
    rfc_objective: str = "legacy"  # "legacy" (Q/K FutureQueryPredictor) | "A"/"B"/"C"/"D"/"G" (rescue/future_contrib/train_unified.py checkpoints) | "E" (hybrid: legacy reuse x objective-C impact)
    rfc_allocation: str = "per_head"  # "per_head" (default: fixed budget per kv_head, per layer) | "global_layer" (matches LaProx's own default: heads within a layer share one retained-position set via score averaging, AND budget is pooled/ragged across layers)
    rfc_recent_style: str = "contribution"  # "contribution" (default: Gram-matrix true output-contribution, rescue.future_contrib.contribution.recent_contribution_from_attn) | "laprox" (LaProx's own exact ||attn||_2 * ||V W_O|| formula + its snapkv_obs_window protected-window convention -- lambda=0 + rfc_allocation="global_layer" + this reproduces LaProx's own score exactly, a verification baseline)
    rfc_kslot: int = 16   # rfc_combine_mode="kslot" only: how many of the B slots the
                          # correction may fill. The base policy keeps B - rfc_kslot
                          # outright, so the correction can add entries but never
                          # displace ones the base ranked into its own top-(B-k).
    rfc_combine_mode: str = "additive"  # "additive" (default: S_recent + lambda*S_future) | "rank_gate" (priority 7: percentile-rank multiplicative gate, see dense_eviction_hf._rank_gate_combine) | "rank_additive" (percentile-rank(S_recent) + lambda*percentile-rank(S_future), see dense_eviction_hf._rank_additive_combine)
    protect_special: bool = True    # RESCUE: force-keep Llama-3 chat-template control tokens (<|begin_of_text|>, <|start_header_id|>, <|eot_id|>, ...), which _prune_laprox_global_layer does NOT do -- the sole reason rfc_lambda=0 does not reproduce pure LaProx. False makes the two paths identical at lambda=0.
    rfc_ppl_gate_tau: float = 0.0    # >0 enables the PERPLEXITY GATE: lambda drops to exactly 0 on any document whose mean prompt next-token entropy (computed free from the prefill logits) is BELOW tau. A predictable prompt is one where the recent-window base score is already near-optimal, so the correction can only perturb it. Measured per document over 5 tasks: lcc spans 0.413-1.289, every gaining task sits above (gov_report 1.317, qasper 1.414, trec 1.777) -- an empty band, so tau is not tuned inside a contested region. 0.0 = disabled.
    rfc_lambda_select: str = ""       # "" = lambda is whatever --rfc-lambda says (optionally zeroed by a gate). "fidelity" = pick it per document AFTER evicting: decode rfc_fidelity_probe_len tokens against the unpruned cache, re-run them against each candidate lambda's pruned cache, keep the lambda with the smallest KL to the dense distribution. Unlike every prompt-derived gate this varies with the BASE, because the pruned cache does; and lambda=0 is always a candidate, so a document the correction would damage keeps the baseline by construction. Disables both gates when active.
    rfc_fidelity_lambdas: str = "0,1"   # candidate lambdas for rfc_lambda_select="fidelity", cheapest-first; each costs one probe-length decode against a 128-token cache.
    rfc_fidelity_probe: str = "gen"    # "gen" = decode rfc_fidelity_probe_len tokens against the unpruned cache and score the pruned ones on those. "prompt_tail" = score on the last observation-window PROMPT positions instead, whose dense distributions prefill already produced -- nothing is decoded, compression is not delayed, and a request that finishes in a few tokens is unaffected. The risk is the judge: those queries are exactly what every base score ranks on, while the correction is trained against future ones.
    rfc_fidelity_with_gate: bool = False  # keep the entropy gate ACTIVE inside the fidelity trials. With it on, a document below tau has its lambda=1 trial gated to 0, so both trials are the same eviction and the tie resolves to 0 -- i.e. the two signals must AGREE before the correction is applied. The gate reads the text and the fidelity reads the damage, so this is an AND of two independent readings of the same perplexity.
    rfc_fidelity_probe_len: int = 16    # how many dense-decoded tokens the KL is averaged over. They are decoded against the unpruned cache and then DISCARDED -- emitting them would give this arm dense-quality output tokens no baseline gets.
    rfc_rescue_floor_frac: float = 0.0  # rfc_objective="RESCUE" + rfc_allocation="global_layer" only: fraction of the (global, pooled) budget reserved for recent-only's OWN top ranking, bypassing the combined score entirely -- bounds how much of the budget a miscalibrated correction can swap away from recent-only, capping the "blast radius" of a bad C_i decision on any one document (see rescue_eviction_inspect.py's repetition-collapse finding). 0.0 = no floor (combined score competes for the whole budget, current default behavior).
    rkv_window_size: int = 8       # R-KV's own official-code default (R1KV.__init__): trailing real queries always kept + used as the attention-cache observation window
    rkv_mix_lambda: float = 0.07   # official code default (paper text says 0.1 -- this project matches the shipped code per explicit instruction, not the paper text)
    rkv_kernel_size: int = 7
    rkv_retain_ratio: float = 0.1
    rkv_retain_direction: str = "last"  # official default; "first"|"last_percent"|"first_percent" also supported (see rescue.policies.rkv)

    def normalized_policy(self) -> str:
        return str(self.policy or "full").lower()


def supported_policies() -> set[str]:
    return set(ALL_POLICIES)


def resolve_budget(total_tokens: int, cfg: BaselineConfig) -> int:
    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        return 0
    if cfg.budget_tokens is not None:
        budget = int(cfg.budget_tokens)
    else:
        budget = int(round(total_tokens * float(cfg.budget_ratio)))
    return max(1, min(total_tokens, budget))


def protected_positions(total_tokens: int, cfg: BaselineConfig) -> set[int]:
    sink = max(0, int(cfg.sink_tokens))
    recent = max(0, int(cfg.recent_tokens))
    keep: set[int] = set(range(min(sink, total_tokens)))
    if recent:
        keep.update(range(max(0, total_tokens - recent), total_tokens))
    return keep


def resolve_recent_budget(total_budget: int, cfg: BaselineConfig) -> int:
    return max(0, min(int(cfg.recent_tokens), max(0, int(total_budget))))


def query_to_kv_attention(layer_attn: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Aggregate query-head attention into KV-head groups.

    Returns a tensor with shape [kv_heads, q_len, k_len]. For MHA,
    num_kv_heads == num_query_heads. For GQA, query heads are summed inside the
    corresponding KV group, matching how each KV head is repeated.
    """
    attn = layer_attn.detach().float()
    if attn.ndim != 4:
        raise ValueError(f"Expected attention [batch,q_heads,q_len,k_len], got {tuple(attn.shape)}")
    attn = attn[0]
    q_heads, q_len, k_len = attn.shape
    num_kv_heads = int(num_kv_heads)
    if num_kv_heads <= 0 or q_heads % num_kv_heads != 0:
        return attn.mean(dim=0, keepdim=True).expand(max(1, num_kv_heads), q_len, k_len)
    group = q_heads // num_kv_heads
    return attn.view(num_kv_heads, group, q_len, k_len).sum(dim=1)


def select_topk_indices(
    scores: torch.Tensor,
    positions: torch.Tensor,
    budget: int,
    protected_abs: Iterable[int] = (),
) -> torch.Tensor:
    """Select cache slot indices per KV head.

    scores/positions shape: [kv_heads, cache_len]. Returned indices have shape
    [kv_heads, budget_for_this_cache] and are sorted by cache-slot order within
    each head so the retained cache remains chronological.
    """
    scores = scores.detach().float()
    positions = positions.detach().long()
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(scores.shape[0], -1)
    kv_heads, cache_len = scores.shape
    budget = max(0, min(int(budget), int(cache_len)))
    if budget >= cache_len:
        return torch.arange(cache_len, device=positions.device).view(1, -1).expand(kv_heads, -1)
    if budget <= 0:
        return torch.empty((kv_heads, 0), dtype=torch.long, device=positions.device)

    protected_abs_set = {int(p) for p in protected_abs}
    protected_tensor = (
        torch.tensor(sorted(protected_abs_set), dtype=torch.long, device=positions.device)
        if protected_abs_set else None
    )
    selected: list[torch.Tensor] = []
    for h in range(kv_heads):
        if protected_tensor is not None:
            # Vectorized membership test instead of one .eq() kernel launch per
            # protected position (was O(len(protected_abs) * kv_heads) tiny GPU
            # ops per call -- measured to dominate rfc's per-step eviction cost;
            # same result, just not one kernel launch per protected position).
            protected_mask = torch.isin(positions[h], protected_tensor)
        else:
            protected_mask = torch.zeros(cache_len, dtype=torch.bool, device=positions.device)
        protected_idx = protected_mask.nonzero(as_tuple=False).flatten()
        if protected_idx.numel() >= budget:
            keep = protected_idx[-budget:]
            selected.append(keep.sort().values)
            continue
        candidate_idx = (~protected_mask).nonzero(as_tuple=False).flatten()
        need = budget - int(protected_idx.numel())
        if need > 0 and candidate_idx.numel() > 0:
            cand_scores = scores[h, candidate_idx]
            top_rel = torch.topk(cand_scores, k=min(need, int(candidate_idx.numel())), largest=True).indices
            keep = torch.cat([protected_idx, candidate_idx[top_rel]])
        else:
            keep = protected_idx
        selected.append(keep.sort().values)
    return torch.stack(selected, dim=0)


def select_common_topk_indices(
    scores: torch.Tensor,
    positions: torch.Tensor,
    budget: int,
    protected_abs: Iterable[int] = (),
) -> torch.Tensor:
    """Select one chronological token set shared by all KV heads in a layer."""
    scores = scores.detach().float()
    positions = positions.detach().long()
    if scores.ndim == 2:
        scores = scores.mean(dim=0)
    if positions.ndim == 2:
        ref_positions = positions[0]
        kv_heads = positions.shape[0]
    else:
        ref_positions = positions
        kv_heads = 1
    cache_len = int(ref_positions.numel())
    budget = max(0, min(int(budget), cache_len))
    if budget >= cache_len:
        keep = torch.arange(cache_len, device=ref_positions.device)
    elif budget <= 0:
        keep = torch.empty((0,), dtype=torch.long, device=ref_positions.device)
    else:
        protected_abs_set = {int(p) for p in protected_abs}
        protected_mask = torch.zeros(cache_len, dtype=torch.bool, device=ref_positions.device)
        for p in protected_abs_set:
            protected_mask |= ref_positions.eq(p)
        protected_idx = protected_mask.nonzero(as_tuple=False).flatten()
        if protected_idx.numel() >= budget:
            keep = protected_idx[-budget:]
        else:
            candidate_idx = (~protected_mask).nonzero(as_tuple=False).flatten()
            need = budget - int(protected_idx.numel())
            top_rel = torch.topk(scores[candidate_idx], k=min(need, int(candidate_idx.numel())), largest=True).indices
            keep = torch.cat([protected_idx, candidate_idx[top_rel]])
        keep = keep.sort().values
    return keep.view(1, -1).expand(kv_heads, -1)


def start_recent_indices(positions: torch.Tensor, budget: int, sink_tokens: int) -> torch.Tensor:
    # Fully vectorized (no per-element Python loop / per-scalar GPU tensor
    # allocation): decode-time eviction calls this once per layer on EVERY
    # generated token, so a Python-side loop over cache-slot indices here was
    # measured to dominate streamingllm/laprox/kvp's whole decode step
    # (~0.2s/step from ~30k individual torch.tensor(...) allocations across
    # kv_heads x layers x budget) -- fine for short-output QA tasks, but a
    # 5-8x wall-clock blowup on long-output tasks (gov_report/qmsum at 512
    # decode steps vs qasper's 32).
    positions = positions.detach().long()
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    kv_heads, cache_len = positions.shape
    budget = max(0, min(int(budget), cache_len))
    if budget >= cache_len:
        return torch.arange(cache_len, device=positions.device).view(1, -1).expand(kv_heads, -1)
    sink_tokens = max(0, min(int(sink_tokens), budget))

    sink_mask = positions < sink_tokens  # [kv_heads, cache_len], the sink slots present in THIS cache
    sink_count = sink_mask.sum(dim=1).clamp(max=sink_tokens)  # [kv_heads] -- matches the old per-row start[:sink_tokens] cap
    recent_need = (budget - sink_count).clamp(min=0)  # [kv_heads]

    # Recency ranking over the non-sink slots only (sink slots are already
    # guaranteed a seat, so they must never win a "recent" slot too).
    neg_inf = torch.iinfo(positions.dtype).min
    tail_scores = torch.where(sink_mask, torch.full_like(positions, neg_inf), positions)
    max_need = int(recent_need.max().item()) if kv_heads else 0
    keep_mask = sink_mask.clone()
    if max_need > 0:
        k = min(max_need, cache_len)
        tail_idx = torch.topk(tail_scores, k=k, dim=1, largest=True).indices  # [kv_heads, k]
        # Row h only actually wants recent_need[h] (<= k) of these; a
        # per-row arange-vs-recent_need compare drops the extras without
        # any Python-level per-row loop.
        rank = torch.arange(k, device=positions.device).view(1, -1).expand(kv_heads, -1)
        tail_valid = rank < recent_need.view(-1, 1)
        keep_mask.scatter_(1, tail_idx, tail_valid | keep_mask.gather(1, tail_idx))

    # keep_mask now has exactly `budget` True entries per row (sink_count[h] +
    # recent_need[h] == budget by construction); turn it into sorted absolute
    # cache-slot indices per row without a Python loop.
    order = torch.argsort(~keep_mask, dim=1, stable=True)[:, :budget]
    return order.sort(dim=1).values


def select_topk_positions(scores: torch.Tensor, budget: int, protected: Iterable[int]) -> list[int]:
    scores = scores.detach().float().cpu()
    n = int(scores.numel())
    budget = max(0, min(int(budget), n))
    protected_set = {int(p) for p in protected if 0 <= int(p) < n}
    if budget >= n:
        return list(range(n))
    if len(protected_set) >= budget:
        return sorted(protected_set)[-budget:]
    candidates = [p for p in range(n) if p not in protected_set]
    need = budget - len(protected_set)
    if need <= 0:
        return sorted(protected_set)
    cand_scores = scores[candidates]
    top = torch.topk(cand_scores, k=min(need, len(candidates)), largest=True).indices.tolist()
    keep = set(protected_set)
    keep.update(candidates[int(idx)] for idx in top)
    return sorted(keep)


def attention_mean_scores(attentions: tuple[torch.Tensor, ...] | None, seq_len: int) -> torch.Tensor:
    if not attentions:
        return torch.ones(seq_len, dtype=torch.float32)
    total = torch.zeros(seq_len, dtype=torch.float32)
    count = 0
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        # [batch, q_heads, q_len, k_len]
        attn = layer_attn.detach().float()
        if attn.ndim != 4:
            continue
        attn = attn[0, :, :, :seq_len]
        total += attn.sum(dim=(0, 1)).cpu()
        count += int(attn.shape[0] * attn.shape[1])
    if count == 0:
        return torch.ones(seq_len, dtype=torch.float32)
    return total / float(count)


def snapkv_scores(attentions: tuple[torch.Tensor, ...] | None, seq_len: int, obs_window: int) -> torch.Tensor:
    if not attentions:
        return torch.ones(seq_len, dtype=torch.float32)
    obs_window = max(1, min(int(obs_window), seq_len))
    total = torch.zeros(seq_len, dtype=torch.float32)
    count = 0
    start = max(0, seq_len - obs_window)
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        attn = layer_attn.detach().float()
        if attn.ndim != 4:
            continue
        obs = attn[0, :, start:seq_len, :seq_len]
        total += obs.sum(dim=(0, 1)).cpu()
        count += int(obs.shape[0] * obs.shape[1])
    if count == 0:
        return torch.ones(seq_len, dtype=torch.float32)
    return total / float(count)


def laprox_scores(
    attentions: tuple[torch.Tensor, ...] | None,
    past_key_values,
    seq_len: int,
) -> torch.Tensor:
    # Approximate LaProx: observed attention mass times value-vector norm.
    attn_score = attention_mean_scores(attentions, seq_len)
    value_norm = torch.zeros(seq_len, dtype=torch.float32)
    count = 0
    layers = legacy_layers(past_key_values)
    for _, value in layers:
        if value is None:
            continue
        val = value.detach().float()
        # [batch, kv_heads, seq, dim]
        if val.ndim != 4:
            continue
        value_norm += val[0, :, :seq_len, :].norm(p=2, dim=-1).mean(dim=0).cpu()
        count += 1
    if count > 0:
        value_norm /= float(count)
    else:
        value_norm += 1.0
    return attn_score * value_norm


def h2o_prefill_scores(attentions: tuple[torch.Tensor, ...] | None, past_key_values) -> list[torch.Tensor]:
    if not attentions:
        return []
    layers = legacy_layers(past_key_values)
    out: list[torch.Tensor] = []
    for layer_attn, (key, _) in zip(attentions, layers):
        if layer_attn is None or key is None:
            out.append(torch.empty(0))
            continue
        kv_heads = int(key.shape[1])
        grouped = query_to_kv_attention(layer_attn, kv_heads)
        out.append(grouped.sum(dim=1)[:, : key.shape[2]].to(key.device))
    return out


def h2o_decode_scores(layer_attn: torch.Tensor, kv_heads: int, device) -> torch.Tensor:
    grouped = query_to_kv_attention(layer_attn, kv_heads)
    return grouped.sum(dim=1).to(device)


def _pool_scores(vote: torch.Tensor, cfg: BaselineConfig) -> torch.Tensor:
    """Shared SnapKV-style smoothing ("average pooling with a kernel size of
    7 to mitigate information fragmentation", LaProx paper Appendix A -- this
    applies to SnapKV AND LaProx itself, per Table 2's method list; only SLLM
    is exempted). vote: [kv_heads, N], pooled along the token axis N."""
    kernel = max(1, int(cfg.snapkv_kernel_size))
    if kernel <= 1:
        return vote
    n = vote.shape[-1]
    x = vote.unsqueeze(1)
    if cfg.snapkv_pooling.lower() in {"avg", "avgpool", "mean"}:
        pooled = F.avg_pool1d(x, kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(1)
    elif cfg.snapkv_pooling.lower() in {"max", "maxpool"}:
        pooled = F.max_pool1d(x, kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(1)
    else:
        raise ValueError(f"Unsupported pooling={cfg.snapkv_pooling}")
    return pooled[:, :n]


def snapkv_prefill_indices(
    layer_attn: torch.Tensor,
    key: torch.Tensor,
    budget: int,
    cfg: BaselineConfig,
    probe_attn: torch.Tensor | None = None,
    probe_weight: float = 1.0,
) -> torch.Tensor:
    """probe_attn is the reviewer's training-free control: RESCUE decodes a
    probe token off the dense prompt cache anyway, so its attention over the
    prompt is a future signal that costs one decode step and no scorer. Adding
    those rows to SnapKV's own vote asks whether the trained residual scorer
    buys anything the probe does not. The window and prefix_len below stay
    derived from the PROMPT rows, so the protected window and the candidate set
    are bit-identical to plain SnapKV -- only the vote changes."""
    cache_len = int(key.shape[2])
    kv_heads = int(key.shape[1])
    budget = max(1, min(int(budget), cache_len))
    if cache_len <= budget:
        return torch.arange(cache_len, device=key.device).view(1, -1).expand(kv_heads, -1)
    window = max(1, min(int(cfg.snapkv_obs_window), cache_len, budget - 1 if budget > 1 else 1))
    prefix_len = cache_len - window
    grouped = query_to_kv_attention(layer_attn, kv_heads).to(key.device)
    vote = grouped[:, -window:, :prefix_len].sum(dim=1)
    if probe_attn is not None and float(probe_weight) != 0.0:
        pg = query_to_kv_attention(probe_attn, kv_heads).to(key.device)
        vote = vote + float(probe_weight) * pg[:, :, :prefix_len].sum(dim=1)
    vote = _pool_scores(vote, cfg)
    keep_prefix = max(0, budget - window)
    if keep_prefix:
        prefix_idx = torch.topk(vote, k=min(keep_prefix, prefix_len), dim=-1, largest=True).indices
        obs_idx = torch.arange(prefix_len, cache_len, device=key.device).view(1, -1).expand(kv_heads, -1)
        keep = torch.cat([prefix_idx, obs_idx], dim=1)
    else:
        keep = torch.arange(cache_len - budget, cache_len, device=key.device).view(1, -1).expand(kv_heads, -1)
    return keep.sort(dim=1).values


def _layer_o_proj_blocks(model, layer_idx: int, q_heads: int, head_dim: int) -> Sequence[torch.Tensor]:
    # o_proj.weight is a static model parameter -- the old version re-sliced
    # AND re-cast-to-fp32 AND re-transposed it on every call (every layer,
    # every decode step: laprox always requests decode-time attention, so
    # this runs unconditionally each generated token). Cache the fp32+
    # transposed per-head blocks on the model object once; same fix pattern
    # already applied to rfc's o_proj conversion (dense_eviction_hf.py's
    # _rfc_o_proj_fp32) for the identical reason.
    cache = getattr(model, "_laprox_o_proj_blocks_cache", None)
    if cache is None:
        cache = {}
        model._laprox_o_proj_blocks_cache = cache
    blocks = cache.get(layer_idx)
    if blocks is None:
        layer = getattr(getattr(model, "model", model), "layers", [])[layer_idx]
        weight = layer.self_attn.o_proj.weight.detach().float()
        blocks = [weight[:, h * head_dim : (h + 1) * head_dim].t().contiguous() for h in range(q_heads)]
        cache[layer_idx] = blocks
    return blocks


def laprox_vwo_norm(model, layer_idx: int, value: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Compute ||V W_O^h|| per KV head/token, including GQA repetitions."""
    val = value.detach().float()[0]
    kv_heads, cache_len, head_dim = val.shape
    if num_query_heads % kv_heads != 0:
        return val.norm(p=2, dim=-1)
    group = num_query_heads // kv_heads
    blocks = _layer_o_proj_blocks(model, layer_idx, num_query_heads, head_dim)  # each block already fp32, transposed, contiguous
    scores = torch.zeros(kv_heads, cache_len, dtype=torch.float32, device=value.device)
    for kv_h in range(kv_heads):
        for r in range(group):
            qh = kv_h * group + r
            projected = val[kv_h] @ blocks[qh]
            scores[kv_h] += projected.norm(p=2, dim=-1)
    return scores / float(group)


def laprox_indices(
    model,
    layer_idx: int,
    layer_attn: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    budget: int,
    cfg: BaselineConfig,
) -> torch.Tensor:
    kv_heads = int(value.shape[1])
    grouped = query_to_kv_attention(layer_attn, kv_heads).to(value.device)
    attn_norm = grouped.norm(p=2, dim=1)[:, : value.shape[2]]
    attn_norm = _pool_scores(attn_norm, cfg)
    vwo = laprox_vwo_norm(model, layer_idx, value, int(layer_attn.shape[1]))
    score = attn_norm * vwo[:, : attn_norm.shape[1]]
    recent_abs = set()
    recent = max(0, min(int(cfg.snapkv_obs_window), int(positions.shape[1])))
    if recent:
        for h in range(positions.shape[0]):
            recent_abs.update(int(x) for x in positions[h, -recent:].tolist())
    return select_topk_indices(score, positions, budget, recent_abs)


def laprox_layer_token_scores(
    model,
    layer_idx: int,
    layer_attn: torch.Tensor,
    value: torch.Tensor,
    cfg: BaselineConfig,
) -> torch.Tensor:
    kv_heads = int(value.shape[1])
    grouped = query_to_kv_attention(layer_attn, kv_heads).to(value.device)
    attn_norm = grouped.norm(p=2, dim=1)[:, : value.shape[2]]
    attn_norm = _pool_scores(attn_norm, cfg)
    vwo = laprox_vwo_norm(model, layer_idx, value, int(layer_attn.shape[1]))
    head_scores = attn_norm * vwo[:, : attn_norm.shape[1]]
    return head_scores.mean(dim=0)


def laprox_head_token_scores(
    model,
    layer_idx: int,
    layer_attn: torch.Tensor,
    value: torch.Tensor,
    cfg: BaselineConfig,
) -> torch.Tensor:
    """Compute LaProx scores for every KV-head/token slot in one layer.

    Applies the SAME average-pooling(kernel=7)-over-the-token-axis smoothing
    the LaProx paper's Appendix A says is used for every baseline except SLLM
    (SnapKV's own "mitigate information fragmentation" convention, reused for
    LaProx's ||A[:,i]||_2 term) -- this was previously MISSING here (only
    snapkv_prefill_indices applied it), a real, non-inert reproduction gap
    found by comparing this function against the paper's Algorithm 1 +
    Appendix A line by line."""
    kv_heads = int(value.shape[1])
    grouped = query_to_kv_attention(layer_attn, kv_heads).to(value.device)
    attn_norm = grouped.norm(p=2, dim=1)[:, : value.shape[2]]
    attn_norm = _pool_scores(attn_norm, cfg)
    vwo = laprox_vwo_norm(model, layer_idx, value, int(layer_attn.shape[1]))
    return attn_norm * vwo[:, : attn_norm.shape[1]]


# --- probe rows for the "probe + RESCUE" combined arm -----------------------
# RESCUE's s_recent is computed deep inside the state classes, which the probe
# attention cannot reach without threading an override through five of them
# (the same reason the fidelity selector swaps self.cfg instead). The caller in
# dense_eviction_hf sets these around its prune call and clears them after.
_PROBE_ROWS = None
_PROBE_WEIGHT = 0.0


def set_probe_rows(rows, weight):
    global _PROBE_ROWS, _PROBE_WEIGHT
    _PROBE_ROWS, _PROBE_WEIGHT = rows, float(weight)


def clear_probe_rows():
    global _PROBE_ROWS, _PROBE_WEIGHT
    _PROBE_ROWS, _PROBE_WEIGHT = None, 0.0


def snapkv_head_token_scores(
    layer_idx: int,
    layer_attn: torch.Tensor,
    value: torch.Tensor,
    cfg: BaselineConfig,
) -> torch.Tensor:
    """SnapKV's own per-(kv_head, token) score for every slot in one layer --
    parallel to laprox_head_token_scores, for --rfc-recent-style snapkv (RESCUE
    ported onto a SnapKV base instead of LaProx): SUM (not L2-norm) of
    observation-window attention, NO value-projection weighting, same
    kernel=7 smoothing pool (snapkv_prefill_indices's own vote/_pool_scores
    convention). layer_idx accepted only for call-site symmetry with the
    laprox_* functions (unused: SnapKV's formula needs no per-layer weights)."""
    kv_heads = int(value.shape[1])
    grouped = query_to_kv_attention(layer_attn, kv_heads).to(value.device)
    cache_len = int(value.shape[2])
    # Match snapkv_prefill_indices exactly: it votes over the PREFIX only
    # ([:, -window:, :prefix_len]) and pools that slice, so the kernel-7 average
    # near the prefix/observation-window boundary sees zero padding rather than
    # obs-window attention. Pooling the full length instead shifts scores right
    # at that boundary, which is enough to stop rfc_lambda=0 from reproducing
    # SnapKV (measured 24.91 vs 25.15 on qasper). Observation-window positions
    # are protected by the caller either way, so their score is never used --
    # they are filled with the prefix maximum purely to keep the vector's shape
    # and ordering sane for any caller that ranks the whole thing.
    window = max(1, min(int(cfg.snapkv_obs_window), cache_len,
                        grouped.shape[1] if grouped.shape[1] > 0 else 1))
    prefix_len = max(0, cache_len - window)
    if prefix_len == 0:
        return _pool_scores(grouped.sum(dim=1)[:, :cache_len], cfg)
    vote = grouped[:, -window:, :prefix_len].sum(dim=1)
    if _PROBE_ROWS is not None and _PROBE_WEIGHT != 0.0 and layer_idx < len(_PROBE_ROWS):
        pg = query_to_kv_attention(_PROBE_ROWS[layer_idx], kv_heads).to(value.device)
        vote = vote + _PROBE_WEIGHT * pg[:, :, :prefix_len].sum(dim=1)
    pooled = _pool_scores(vote, cfg)
    out = pooled.new_empty((kv_heads, cache_len))
    out[:, :prefix_len] = pooled
    out[:, prefix_len:] = pooled.max(dim=-1, keepdim=True).values
    return out


def lava_head_token_scores(
    layer_attn: torch.Tensor,
    value: torch.Tensor,
    window: int,
    cfg: BaselineConfig,
) -> torch.Tensor:
    """LAVa's own per-(kv_head, token) score (paper Eq 5):
        s[h, i] = (max_k ||V[h, k]||_1 / w) * maxpool_7(sum of the last w
                   real queries' attention onto candidate i)
    Two deliberate divergences from every other baseline in this file:
    - GQA is reduced by MAX across the query group (paper Appendix A.2),
      not sum (query_to_kv_attention) or mean (laprox's global_layer
      pooling) -- confirmed against the paper text, not inferred.
    - Pooling is ALWAYS maxpool kernel=7, regardless of cfg.snapkv_pooling
      (LAVa's own fixed convention, not this project's shared
      --snapkv-pooling default of avgpool).
    No sink-token protection is applied here or by the caller -- LAVa's
    paper protects only the last `w` positions, confirmed deliberate (not
    an oversight) per user instruction to implement LAVa exactly as
    published, diverging from every other baseline here which also
    protects cfg.sink_tokens.
    """
    attn = layer_attn.detach().float()
    if attn.ndim == 4:
        attn = attn[0]
    q_heads, q_len, k_len = attn.shape
    kv_heads = int(value.shape[1])
    group = max(1, q_heads // kv_heads)
    grouped = attn.view(kv_heads, group, q_len, k_len).amax(dim=1)  # [kv_heads, q_len, k_len] -- MAX, not sum/mean
    cache_len = int(value.shape[2])
    w = max(1, min(int(window), q_len))
    vote = grouped[:, -w:, :cache_len].sum(dim=1)  # [kv_heads, cache_len]
    kernel = max(1, int(cfg.snapkv_kernel_size))
    pooled = F.max_pool1d(vote.unsqueeze(1), kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(1)[:, :cache_len]
    val = value.detach().float()[0, :, :cache_len, :]  # [kv_heads, cache_len, head_dim]
    v_l1 = val.norm(p=1, dim=-1)  # [kv_heads, cache_len]
    v_max_scalar = v_l1.max(dim=-1, keepdim=True).values.clamp_min(1e-12)  # [kv_heads, 1]
    return (v_max_scalar / float(w)) * pooled


def recent_absolute_positions(positions: torch.Tensor, window: int) -> set[int]:
    positions = positions.detach().long()
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    recent = max(0, min(int(window), int(positions.shape[1])))
    if recent == 0:
        return set()
    out: set[int] = set()
    for h in range(positions.shape[0]):
        out.update(int(x) for x in positions[h, -recent:].tolist())
    return out


def legacy_layers(past_key_values):
    if past_key_values is None:
        return []
    if isinstance(past_key_values, (tuple, list)):
        return [(layer[0], layer[1]) for layer in past_key_values]
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return list(zip(key_cache, value_cache))
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        out = []
        for layer in layers:
            out.append((getattr(layer, "keys", None), getattr(layer, "values", None)))
        return out
    return []


def compute_keep_positions(
    cfg: BaselineConfig,
    attentions: tuple[torch.Tensor, ...] | None,
    past_key_values,
    seq_len: int,
) -> list[int]:
    policy = cfg.normalized_policy()
    if policy == "full":
        return list(range(seq_len))
    if policy in LEARNED_POLICIES:
        if not cfg.learned_checkpoint:
            raise ValueError(f"{policy} requires learned_checkpoint before OpenCompass evaluation")
        raise NotImplementedError(f"{policy} inference hook is prepared, but predictor loading is not implemented yet")
    budget = resolve_budget(seq_len, cfg)
    protected = protected_positions(seq_len, cfg)
    if policy == "streamingllm":
        keep = sorted(protected)
        if len(keep) < budget:
            tail = list(range(max(0, seq_len - (budget - len(keep))), seq_len))
            keep = sorted(set(keep) | set(tail))
        return keep[-budget:] if len(keep) > budget else keep
    if policy == "h2o":
        scores = attention_mean_scores(attentions, seq_len)
    elif policy == "snapkv":
        scores = snapkv_scores(attentions, seq_len, cfg.snapkv_obs_window)
    elif policy == "laprox":
        scores = laprox_scores(attentions, past_key_values, seq_len)
    else:
        raise ValueError(f"Unsupported KV eviction policy: {policy}")
    return select_topk_positions(scores, budget, protected)
