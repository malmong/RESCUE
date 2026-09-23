"""Loads src.method.future_contrib checkpoints (produced by
src/method/future_contrib/train.py) and serves S_future = R_hat * M_hat at
eviction decisions. Naming follows this project's other *_paper.py scorers
(kvp_paper.py, foresightkv_paper.py).

S_recent is NOT computed here -- it's derived from whatever attention HF's
forward pass already produced (src.method.future_contrib.contribution.
recent_contribution_from_attn), since it needs no learned checkpoint. This
class owns the frozen projection + per-layer impact predictors + per-layer
future-query predictors + the S_total mixing weight lambda.

Reuse (R_hat) is no longer a per-candidate hidden-state-projection MLP
output: this project's own diagnostics found that feature basis carried
almost no signal about true future reuse (Pearson ~0.03-0.09, even in-sample
on the model's own training data), while switching to the real future
softmax-attention distribution over candidate KEYS raised it to ~0.87-0.98,
and a small query predictor trained via KL divergence against that
distribution reached ~0.74-0.86 on held-out documents. See
src/method/future_contrib/train.py's module docstring for the full diagnostic
history. Impact (M_hat) keeps its original hidden-state-projection feature
basis unchanged -- a quick check found ||V_i @ W_O|| alone only weakly
explains conditional impact (corr 0.02-0.17), so there was no clear basis
for replacing it too.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.method.future_contrib.features import build_candidate_feature
from src.method.future_contrib.labels import inverse_log1p_transform
from src.method.future_contrib.model import FutureQueryPredictor, ImpactPredictor
from src.method.future_contrib.projection import FrozenRandomProjection


class RFCScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu", lambda_mix: float = 1.0):
        payload = torch.load(str(checkpoint), map_location="cpu")
        required = {"query_predictors", "impact_predictors", "projection", "model_arch", "recent_window"}
        if not required.issubset(payload.keys()):
            raise ValueError(
                f"{checkpoint} is not a future_contrib checkpoint in the current (Q/K reuse) "
                f"format (expected keys {sorted(required)} -- produced by src/method/future_contrib/train.py)"
            )
        self.device = torch.device(device)
        self.lambda_mix = float(lambda_mix)
        train_args = payload.get("args", {}) or {}
        self.alpha = float(train_args.get("alpha", 1.0))
        self.recent_window = int(payload["recent_window"])
        arch = payload["model_arch"]
        self.head_dim = int(arch["head_dim"])
        self.scaling = self.head_dim ** -0.5

        proj_sd = payload["projection"]
        self.proj_dim = int(proj_sd["proj_dim"])
        self.projection = FrozenRandomProjection.from_state_dict(proj_sd, device=self.device, dtype=torch.float32)

        shared_hidden_dim = int(train_args.get("shared_hidden_dim", 64))
        task_hidden_dim = int(train_args.get("task_hidden_dim", 16))
        activation = str(train_args.get("activation", "silu"))
        query_hidden_dim = int(payload.get("query_hidden_dim", train_args.get("query_hidden_dim", 128)))

        self.impact_predictors: list[ImpactPredictor] = []
        for state_dict in payload["impact_predictors"]:
            m = ImpactPredictor(
                input_dim=3 * self.proj_dim, shared_hidden_dim=shared_hidden_dim,
                task_hidden_dim=task_hidden_dim, activation=activation,
            ).to(self.device)
            m.load_state_dict(state_dict)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.impact_predictors.append(m)

        self.query_predictors: list[FutureQueryPredictor] = []
        for state_dict in payload["query_predictors"]:
            m = FutureQueryPredictor(self.recent_window, self.head_dim, hidden_dim=query_hidden_dim).to(self.device)
            m.load_state_dict(state_dict)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.query_predictors.append(m)

    @torch.no_grad()
    def project(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., hidden_size] -> [..., proj_dim], on this scorer's device."""
        return self.projection.project(h.to(self.device))

    @torch.no_grad()
    def score_reuse(self, recent_q_flat: torch.Tensor, k_cand: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """recent_q_flat: [num_kv_heads, recent_window*head_dim] (ring-buffer
        state, see src.method.future_contrib.query_state.QueryRingBuffer). k_cand:
        [num_kv_heads, num_candidates, head_dim] (real cached keys, NOT
        projected -- this is the whole point of the Q/K redesign). Returns
        R_hat, the predicted future-reuse probability distribution per
        kv-head (sums to 1 over candidates): [num_kv_heads, num_candidates]."""
        if layer_idx >= len(self.query_predictors) or k_cand.shape[1] == 0:
            return k_cand.new_zeros((k_cand.shape[0], k_cand.shape[1]))
        q_hat = self.query_predictors[layer_idx](recent_q_flat.to(self.device))  # [num_kv_heads, head_dim]
        k_cand = k_cand.to(self.device).float()
        logits = torch.bmm(k_cand, q_hat.float().unsqueeze(-1)).squeeze(-1) * self.scaling  # [num_kv_heads, N]
        return torch.softmax(logits, dim=-1).to(k_cand.device)

    @torch.no_grad()
    def score_impact(self, z_cand: torch.Tensor, c_t: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """z_cand: [num_candidates, proj_dim], c_t: [proj_dim] (hidden-state-
        projection features, unchanged from the original design). Returns
        M_hat: [num_candidates]."""
        x = build_candidate_feature(z_cand.to(self.device), c_t.to(self.device))  # [N, 3*proj_dim]
        impact_log_hat = self.impact_predictors[layer_idx](x)
        return inverse_log1p_transform(impact_log_hat, alpha=self.alpha)

    @torch.no_grad()
    def score_vwo_impact(self, v_cand: torch.Tensor, o_proj_weight: torch.Tensor, kv_repeat: int) -> torch.Tensor:
        """Non-learned alternative to score_impact: ||V_i @ W_O||, averaged
        over query-group heads. A diagnostic found the LEARNED impact_predictor
        correlates near-zero with true conditional impact (Pearson 0.01-0.10,
        some layers negative -- same failure mode the original hidden-state
        reuse feature had), while reuse_hat * ||V_i @ W_O|| correlates far
        better (0.17-0.54 across layers) without any additional training --
        essentially LaProx's own ||A||*||V W_O|| contribution proxy, with our
        predicted FUTURE reuse standing in for LaProx's raw (past) attention
        norm. v_cand: [num_kv_heads, num_candidates, head_dim] (real cached
        values, not GQA-repeated). Returns [num_candidates] (kv-head-averaged,
        matching score_impact's shape so callers don't need to branch)."""
        head_dim = v_cand.shape[-1]
        num_heads = o_proj_weight.shape[1] // head_dim
        v_rep = v_cand.to(self.device).float().repeat_interleave(kv_repeat, dim=0)  # [num_heads, N, head_dim]
        wo = o_proj_weight.to(self.device).float()
        wo_blocks = wo.view(wo.shape[0], num_heads, head_dim).permute(1, 2, 0)  # [num_heads, head_dim, hidden]
        g_all = torch.bmm(v_rep, wo_blocks)  # [num_heads, N, hidden]
        return g_all.norm(dim=-1).mean(dim=0)  # [N]

    @torch.no_grad()
    def score_future(
        self,
        recent_q_flat: torch.Tensor,
        k_cand: torch.Tensor,
        z_cand: torch.Tensor,
        c_t: torch.Tensor,
        layer_idx: int,
        use_impact: bool = True,
    ) -> torch.Tensor:
        """Returns S_future: [num_kv_heads, num_candidates]. S_future = R_hat *
        M_hat by default (use_impact=True); with use_impact=False, returns
        R_hat alone (an ablation isolating the reuse/future-attention signal
        from the impact/contribution-magnitude signal)."""
        reuse_hat = self.score_reuse(recent_q_flat, k_cand, layer_idx)  # [num_kv_heads, N]
        if not use_impact:
            return reuse_hat
        impact_hat = self.score_impact(z_cand, c_t, layer_idx)  # [N]
        return reuse_hat * impact_hat.unsqueeze(0).to(reuse_hat.device)
