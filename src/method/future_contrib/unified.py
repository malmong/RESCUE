"""Shared feature basis + tiny model heads for the 4-way future-formulation
comparison (objectives A/B/C/D). All four objectives consume the IDENTICAL
6-dim per-candidate feature vector and a same-shaped tiny MLP -- only the
training TARGET and LOSS differ between them, so any score difference at
eval time is attributable to objective/target choice, not architecture or
feature choice.

Feature vector per candidate i (see compute_unified_features_per_head):
    x_i = [ q_last . k_i * scale,   q_mean . k_i * scale,
            ||k_i||,                ||v_i||,
            ||W_O v_i|| (kv-head-averaged),
            log1p(cur_t - pos_i) ]

- A (FutureAttn/KL): per-kv-head score -> softmax over candidates -> KL
  against the real future softmax-attention distribution. Uses
  NormalizedDirectScorer, one instance per (layer), applied independently
  per kv-head (features already carry a kv-head axis).
- B (DirectContribution): position-level (kv-head-averaged features) ->
  Huber regression against log1p(mean future output-contribution). Uses
  NormalizedDirectScorer.
- C (Factorized reuse x impact): position-level -> (reuse_rate in [0,1],
  impact in log1p-domain), reusing src.method.future_contrib.labels'
  build_future_labels targets exactly. Uses NormalizedFactorizedScorer
  (SeparateFuturePredictor under the hood).
- D (Ranking): position-level -> same raw target as B (mean future
  contribution, pre-log1p) but trained with a pairwise ranking loss instead
  of regression. Uses NormalizedDirectScorer.
"""
from __future__ import annotations

import torch
from torch import nn

from src.method.future_contrib.model import DirectFutureScorer, SeparateFuturePredictor

OBJECTIVES = ("A", "B", "C", "D")
FEATURE_DIM = 6


def vwo_norm_candidates(v_cand: torch.Tensor, o_proj_weight: torch.Tensor, kv_repeat: int) -> torch.Tensor:
    """v_cand: [num_kv_heads, num_candidates, head_dim] (real cached values,
    NOT GQA-repeated). Returns ||V_i @ W_O||, kv-head-averaged: [num_candidates].
    Identical math to src.method.rfc_paper.RFCScorer.score_vwo_impact,
    duplicated here (not imported) so this module has no dependency on a
    loaded RFCScorer checkpoint -- this is a pure function of K/V/weights."""
    head_dim = v_cand.shape[-1]
    num_heads = o_proj_weight.shape[1] // head_dim
    v_rep = v_cand.float().repeat_interleave(kv_repeat, dim=0)  # [num_heads, N, head_dim]
    wo = o_proj_weight.float()
    wo_blocks = wo.view(wo.shape[0], num_heads, head_dim).permute(1, 2, 0)  # [num_heads, head_dim, hidden]
    g_all = torch.bmm(v_rep, wo_blocks)  # [num_heads, N, hidden]
    return g_all.norm(dim=-1).mean(dim=0)  # [N]


def compute_unified_features_per_head(
    q_last: torch.Tensor,
    q_mean: torch.Tensor,
    k_cand: torch.Tensor,
    v_cand: torch.Tensor,
    cand_positions: torch.Tensor,
    cur_t: int,
    wo_norm: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """q_last/q_mean: [num_kv_heads, head_dim]. k_cand/v_cand: [num_kv_heads,
    num_candidates, head_dim]. cand_positions: [num_candidates] (shared
    across heads) OR [num_kv_heads, num_candidates] (per-head, e.g. once
    eviction has diverged per kv-head at eval time) long tensor. wo_norm:
    [num_candidates] (see vwo_norm_candidates). Returns
    [num_kv_heads, num_candidates, 6]."""
    dot_last = torch.einsum("hd,hnd->hn", q_last.float(), k_cand.float()) * scaling
    dot_mean = torch.einsum("hd,hnd->hn", q_mean.float(), k_cand.float()) * scaling
    k_norm = k_cand.float().norm(dim=-1)
    v_norm = v_cand.float().norm(dim=-1)
    num_kv_heads = k_cand.shape[0]
    pos = cand_positions.to(k_cand.device).float()
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).expand(num_kv_heads, -1)
    rel_pos = torch.log1p((cur_t - pos).clamp_min(0.0))  # [H, N]
    wo_b = wo_norm.to(k_cand.device).float().unsqueeze(0).expand(num_kv_heads, -1)
    return torch.stack([dot_last, dot_mean, k_norm, v_norm, wo_b, rel_pos], dim=-1)  # [H, N, 6]


def compute_activepattern_features_per_head(
    q_last: torch.Tensor,
    q_mean: torch.Tensor,
    k_cand: torch.Tensor,
    v_cand: torch.Tensor,
    cand_positions: torch.Tensor,
    cur_t: int,
    wo_norm: torch.Tensor,
    scaling: float,
    recency: torch.Tensor,
    freq: torch.Tensor,
    streak: torch.Tensor,
) -> torch.Tensor:
    """Objective H: the same 6 Q/K-compatibility features as A, plus 3
    observed-activity-history features (recency/freq/streak, see
    src.method.future_contrib.active_pattern) -- NOT predicted, directly
    measured from what actually happened. recency/freq/streak: [num_kv_heads,
    num_candidates] each. Returns [num_kv_heads, num_candidates, 9]."""
    base = compute_unified_features_per_head(q_last, q_mean, k_cand, v_cand, cand_positions, cur_t, wo_norm, scaling)
    extra = torch.stack(
        [torch.log1p(recency.float().clamp_min(0.0)), freq.float(), torch.log1p(streak.float().clamp_min(0.0))],
        dim=-1,
    ).to(base.device)
    return torch.cat([base, extra], dim=-1)  # [H, N, 9]


def to_position_level(feat_per_head: torch.Tensor) -> torch.Tensor:
    """[num_kv_heads, num_candidates, 6] -> [num_candidates, 6], averaging the
    (already kv-head-resolved) dot-product/norm features over kv-heads --
    matches this project's existing convention of treating contribution
    magnitude as a position-level (not per-head) property (see
    RFCScorer.score_vwo_impact / build_future_labels, both head-independent)."""
    return feat_per_head.mean(dim=0)


class NormalizedDirectScorer(nn.Module):
    """LayerNorm(6) -> DirectFutureScorer(6). Used for objectives A, B, D --
    a single opaque scalar per candidate, target/loss decide the semantics."""

    def __init__(self, hidden_dim: int = 32, task_hidden_dim: int = 16, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.core = DirectFutureScorer(input_dim=feature_dim, shared_hidden_dim=hidden_dim, task_hidden_dim=task_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(self.norm(x))


class NormalizedFactorizedScorer(nn.Module):
    """LayerNorm(6) -> SeparateFuturePredictor(6). Used for objective C:
    returns (reuse_hat in [0,1], impact_log_hat unconstrained)."""

    def __init__(self, hidden_dim: int = 32, task_hidden_dim: int = 16, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.core = SeparateFuturePredictor(input_dim=feature_dim, shared_hidden_dim=hidden_dim, task_hidden_dim=task_hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core(self.norm(x))


RICH_REUSE_M = 4  # number of individual recent queries dotted against candidate keys for objective G


def compute_richreuse_features_per_head(
    q_last_m: torch.Tensor,
    k_cand: torch.Tensor,
    v_cand: torch.Tensor,
    cand_positions: torch.Tensor,
    cur_t: int,
    wo_norm: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Objective G: like compute_unified_features_per_head, but instead of
    collapsing the recent-query window to (last, mean) (2 dot-product
    features), keeps RICH_REUSE_M individual recent queries as separate
    dot-product features -- a middle ground between A's 2-feature collapse
    and legacy's full flatten+learned-projection, testing whether temporal
    richness in the reuse feature (not the KL objective) is what closes the
    gap to legacy. q_last_m: [num_kv_heads, RICH_REUSE_M, head_dim].
    Returns [num_kv_heads, num_candidates, RICH_REUSE_M + 4]."""
    dots = torch.einsum("hmd,hnd->hmn", q_last_m.float(), k_cand.float()) * scaling  # [H, M, N]
    num_kv_heads = k_cand.shape[0]
    k_norm = k_cand.float().norm(dim=-1)
    v_norm = v_cand.float().norm(dim=-1)
    pos = cand_positions.to(k_cand.device).float()
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).expand(num_kv_heads, -1)
    rel_pos = torch.log1p((cur_t - pos).clamp_min(0.0))
    wo_b = wo_norm.to(k_cand.device).float().unsqueeze(0).expand(num_kv_heads, -1)
    static = torch.stack([k_norm, v_norm, wo_b, rel_pos], dim=-1)  # [H, N, 4]
    dots_t = dots.permute(0, 2, 1)  # [H, N, M]
    return torch.cat([dots_t, static], dim=-1)  # [H, N, M+4]


def build_predictor(objective: str, hidden_dim: int = 32, task_hidden_dim: int = 16) -> nn.Module:
    if objective == "C":
        return NormalizedFactorizedScorer(hidden_dim, task_hidden_dim)
    if objective in ("A", "B", "D", "S", "R"):
        return NormalizedDirectScorer(hidden_dim, task_hidden_dim)
    if objective == "G":
        return NormalizedDirectScorer(hidden_dim, task_hidden_dim, feature_dim=RICH_REUSE_M + 4)
    if objective == "H":
        return NormalizedDirectScorer(hidden_dim, task_hidden_dim, feature_dim=FEATURE_DIM + 3)
    raise ValueError(f"Unknown objective: {objective}")
