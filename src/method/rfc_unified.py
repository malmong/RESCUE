"""Eval-time scorers for the future-formulation experiments (objectives
A/B/C/D/G, see src/method/future_contrib/train_unified.py and
src/method/future_contrib/unified.py), plus HybridReuseImpactScorer (objective
E: proven legacy Q/K reuse x a separately-trained objective-C impact head).
Mirrors src.method.rfc_paper.RFCScorer's role: loads a checkpoint and
serves S_future at eviction decisions, using the SAME feature-extraction
functions training used (no duplicated feature logic between train/eval,
same discipline as the Q/K FutureQueryPredictor before it)."""
from __future__ import annotations

from pathlib import Path

import torch

from src.method.future_contrib.labels import inverse_log1p_transform
from src.method.future_contrib.query_state import QueryRingBuffer
from src.method.future_contrib.unified import (
    RICH_REUSE_M,
    build_predictor,
    compute_activepattern_features_per_head,
    compute_richreuse_features_per_head,
    compute_unified_features_per_head,
    to_position_level,
    vwo_norm_candidates,
)


class UnifiedRFCScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu", lambda_mix: float = 1.0):
        payload = torch.load(str(checkpoint), map_location="cpu")
        required = {"objective", "predictors", "model_arch", "recent_window"}
        if not required.issubset(payload.keys()):
            raise ValueError(
                f"{checkpoint} is not a train_unified.py checkpoint (expected keys {sorted(required)})"
            )
        self.objective = payload["objective"]
        self.device = torch.device(device)
        self.lambda_mix = float(lambda_mix)
        self.recent_window = int(payload["recent_window"])
        self.alpha = float(payload.get("alpha", 1.0))
        arch = payload["model_arch"]
        self.head_dim = int(arch["head_dim"])
        self.scaling = self.head_dim ** -0.5
        hidden_dim = int(payload.get("hidden_dim", 32))
        task_hidden_dim = int(payload.get("task_hidden_dim", 16))

        self.predictors: list[torch.nn.Module] = []
        for state_dict in payload["predictors"]:
            m = build_predictor(self.objective, hidden_dim, task_hidden_dim).to(self.device)
            m.load_state_dict(state_dict)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.predictors.append(m)

    @torch.no_grad()
    def score(
        self,
        q_ring: QueryRingBuffer,
        k_cand: torch.Tensor,
        v_cand: torch.Tensor,
        cand_positions: torch.Tensor,
        cur_t: int,
        o_proj_weight: torch.Tensor,
        kv_repeat: int,
        layer_idx: int,
        activity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """k_cand/v_cand: [num_kv_heads, num_candidates, head_dim] (real cached
        K/V, NOT GQA-repeated). activity: (recency, freq, streak), each
        [num_kv_heads, num_candidates] -- objective H only, see
        src.method.future_contrib.active_pattern; ignored by every other
        objective. Returns S_future: [num_kv_heads, num_candidates]
        (objectives A/G/H are genuinely per-kv-head; B/C/D are position-level,
        broadcast across kv_heads to keep the same return shape rfc's
        eviction branch already expects)."""
        num_kv_heads = k_cand.shape[0]
        if layer_idx >= len(self.predictors) or k_cand.shape[1] == 0:
            return k_cand.new_zeros((num_kv_heads, k_cand.shape[1]))
        pred = self.predictors[layer_idx]
        wo_norm = vwo_norm_candidates(v_cand.to(self.device), o_proj_weight.to(self.device), kv_repeat)

        if self.objective == "G":
            q_last_m = q_ring.last_m(RICH_REUSE_M).to(self.device)
            feat_rich = compute_richreuse_features_per_head(
                q_last_m, k_cand.to(self.device), v_cand.to(self.device), cand_positions, cur_t, wo_norm, self.scaling,
            )
            logits = pred(feat_rich).squeeze(-1)  # [H, N]
            return torch.softmax(logits, dim=-1).to(k_cand.device)

        if self.objective == "H":
            if activity is None:
                return k_cand.new_zeros((num_kv_heads, k_cand.shape[1]))
            recency, freq, streak = activity
            q_last, q_mean = q_ring.last_and_mean()
            feat_h = compute_activepattern_features_per_head(
                q_last.to(self.device), q_mean.to(self.device), k_cand.to(self.device), v_cand.to(self.device),
                cand_positions, cur_t, wo_norm, self.scaling,
                recency.to(self.device), freq.to(self.device), streak.to(self.device),
            )
            logits = pred(feat_h).squeeze(-1)  # [H, N]
            return torch.softmax(logits, dim=-1).to(k_cand.device)

        q_last, q_mean = q_ring.last_and_mean()
        feat_per_head = compute_unified_features_per_head(
            q_last.to(self.device), q_mean.to(self.device), k_cand.to(self.device), v_cand.to(self.device),
            cand_positions, cur_t, wo_norm, self.scaling,
        )
        if self.objective == "A":
            logits = pred(feat_per_head).squeeze(-1)  # [H, N]
            return torch.softmax(logits, dim=-1).to(k_cand.device)

        feat_pos = to_position_level(feat_per_head)  # [N, 6]
        if self.objective == "B" or self.objective == "R":
            # R (eviction regret) uses the exact same log1p-domain regression
            # head/transform as B -- only the training TARGET differs
            # (gradient-based regret vs future output-contribution).
            y_log_hat = pred(feat_pos).squeeze(-1)
            val = inverse_log1p_transform(y_log_hat, alpha=self.alpha)
        elif self.objective == "S":
            # Survival classification: pred is a logit, sigmoid gives a
            # [0,1] "probability this candidate survives the real future
            # top-budget set" -- used directly as S_future, same scale
            # convention as objective C's reuse_hat.
            logit = pred(feat_pos).squeeze(-1)
            val = torch.sigmoid(logit)
        elif self.objective == "D":
            raw = pred(feat_pos).squeeze(-1)  # trained to preserve order only, not calibrated units
            # Diagnostic (2026-08-22, 4-way comparison): raw D score has no
            # anchored scale, and adding it directly into S_total = S_recent +
            # lambda*S_future let an arbitrary learned scale dominate/distort
            # S_recent even at small lambda (score got monotonically WORSE as
            # lambda grew, 28.82 -> 25.6). Z-score across this step's own
            # candidate pool anchors it to the same rough scale S_recent lives
            # on before mixing.
            val = (raw - raw.mean()) / (raw.std() + 1e-6)
        else:  # C
            reuse_hat, impact_log_hat = pred(feat_pos)
            impact_hat = inverse_log1p_transform(impact_log_hat, alpha=self.alpha)
            val = reuse_hat * impact_hat
        return val.unsqueeze(0).expand(num_kv_heads, -1).to(k_cand.device)

    @torch.no_grad()
    def score_impact_only(
        self,
        q_ring: QueryRingBuffer,
        k_cand: torch.Tensor,
        v_cand: torch.Tensor,
        cand_positions: torch.Tensor,
        cur_t: int,
        o_proj_weight: torch.Tensor,
        kv_repeat: int,
        layer_idx: int,
    ) -> torch.Tensor:
        """Objective-C-only: returns impact_hat alone (no reuse multiply),
        [num_candidates], for HybridReuseImpactScorer to pair with a
        different (better) reuse signal."""
        assert self.objective == "C", "score_impact_only is only defined for objective C checkpoints"
        if layer_idx >= len(self.predictors) or k_cand.shape[1] == 0:
            return k_cand.new_zeros((k_cand.shape[1],))
        pred = self.predictors[layer_idx]
        wo_norm = vwo_norm_candidates(v_cand.to(self.device), o_proj_weight.to(self.device), kv_repeat)
        q_last, q_mean = q_ring.last_and_mean()
        feat_per_head = compute_unified_features_per_head(
            q_last.to(self.device), q_mean.to(self.device), k_cand.to(self.device), v_cand.to(self.device),
            cand_positions, cur_t, wo_norm, self.scaling,
        )
        feat_pos = to_position_level(feat_per_head)
        _, impact_log_hat = pred(feat_pos)
        return inverse_log1p_transform(impact_log_hat, alpha=self.alpha).to(k_cand.device)


class HybridReuseImpactScorer:
    """Objective E: pairs the PROVEN legacy Q/K FutureQueryPredictor's reuse
    branch (32.19 qasper alone, this project's best result) with a
    separately-trained objective-C impact head (6-feature factorized
    reuse/impact predictor) instead of the original hidden-state-projection
    ImpactPredictor (found near-zero correlated with true impact). The two
    are trained completely independently -- the impact head's training
    target (conditional_impact) never depends on which reuse branch it will
    later be multiplied with, so reusing an already-trained objective-C
    checkpoint's impact head here is exact, not an approximation."""

    def __init__(
        self,
        legacy_checkpoint: str | Path,
        impact_checkpoint: str | Path,
        device: torch.device | str = "cpu",
        lambda_mix: float = 1.0,
    ):
        from src.method.rfc_paper import RFCScorer  # local import: avoid a hard cycle at module load time

        self.legacy = RFCScorer(legacy_checkpoint, device=device, lambda_mix=lambda_mix)
        self.impact_scorer = UnifiedRFCScorer(impact_checkpoint, device=device, lambda_mix=lambda_mix)
        if self.impact_scorer.objective != "C":
            raise ValueError(f"HybridReuseImpactScorer requires an objective-C impact checkpoint, got {self.impact_scorer.objective}")
        self.device = torch.device(device)
        self.lambda_mix = float(lambda_mix)
        self.recent_window = self.legacy.recent_window

    @torch.no_grad()
    def score(
        self,
        q_ring: QueryRingBuffer,
        k_cand: torch.Tensor,
        v_cand: torch.Tensor,
        cand_positions: torch.Tensor,
        cur_t: int,
        o_proj_weight: torch.Tensor,
        kv_repeat: int,
        layer_idx: int,
        activity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        recent_q_flat = q_ring.flatten().to(self.device)
        reuse_hat = self.legacy.score_reuse(recent_q_flat, k_cand, layer_idx)  # [H, N]
        impact_hat = self.impact_scorer.score_impact_only(
            q_ring, k_cand, v_cand, cand_positions, cur_t, o_proj_weight, kv_repeat, layer_idx,
        )  # [N]
        return (reuse_hat * impact_hat.unsqueeze(0).to(reuse_hat.device)).to(k_cand.device)
