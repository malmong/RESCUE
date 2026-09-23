"""Eval-time scorer for scripts/train_query_contribution.py's checkpoint
(rfc_objective="QC"): predicts a future query from recent-query history
(identical architecture/input to legacy's FutureQueryPredictor), then scores
each candidate as softmax(q_hat . k_cand) * ||V_cand @ W_O|| -- the same
formula structure as real output-contribution, substituting predicted
attention for real attention -- instead of legacy's raw reuse-probability
distribution. Same generic .score(...) interface as UnifiedRFCScorer/
MultiHeadRFCScorer, so it plugs into rfc's "unified_rfc_scorer" slot in
dense_eviction_hf.py without new eviction-loop code."""
from __future__ import annotations

from pathlib import Path

import torch

from src.method.future_contrib.model import FutureQueryPredictor
from src.method.future_contrib.query_state import QueryRingBuffer
from src.method.future_contrib.unified import vwo_norm_candidates


class QueryContributionScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu", lambda_mix: float = 1.0):
        payload = torch.load(str(checkpoint), map_location="cpu")
        required = {"query_predictors", "model_arch", "recent_window"}
        if not required.issubset(payload.keys()):
            raise ValueError(f"{checkpoint} is not a train_query_contribution.py checkpoint (expected keys {sorted(required)})")
        self.device = torch.device(device)
        self.lambda_mix = float(lambda_mix)
        self.recent_window = int(payload["recent_window"])
        arch = payload["model_arch"]
        self.head_dim = int(arch["head_dim"])
        self.scaling = self.head_dim ** -0.5
        query_hidden_dim = int(payload.get("query_hidden_dim", 128))

        self.query_predictors: list[torch.nn.Module] = []
        for state_dict in payload["query_predictors"]:
            m = FutureQueryPredictor(self.recent_window, self.head_dim, hidden_dim=query_hidden_dim).to(self.device)
            m.load_state_dict(state_dict)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.query_predictors.append(m)

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
        """Returns S_future: [1, num_candidates] (position-level -- the
        contribution formula's V-W_O norm is already kv-head-averaged, same
        convention as objective S/R's contribution-derived targets), which
        broadcasts against a per-kv-head S_recent term same as those
        objectives' position-level scores already do."""
        if layer_idx >= len(self.query_predictors) or k_cand.shape[1] == 0:
            return k_cand.new_zeros((1, k_cand.shape[1]))
        recent_q_flat = q_ring.flatten().to(self.device)
        q_hat = self.query_predictors[layer_idx](recent_q_flat)  # [num_kv_heads, head_dim]
        k_cand_dev = k_cand.to(self.device).float()  # [num_kv_heads, N, head_dim]
        v_cand_dev = v_cand.to(self.device).float()
        logits = torch.bmm(k_cand_dev, q_hat.float().unsqueeze(-1)).squeeze(-1) * self.scaling  # [num_kv_heads, N]
        pred_attn = torch.softmax(logits, dim=-1)
        vwo = vwo_norm_candidates(v_cand_dev, o_proj_weight.to(self.device), kv_repeat)  # [N]
        pred_contribution = (pred_attn * vwo.unsqueeze(0)).mean(dim=0, keepdim=True)  # [1, N]
        return pred_contribution.to(k_cand.device)
