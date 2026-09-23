"""Eval-time scorer for the multi-head future-query predictor (priorities 4
and 5, see src/method/future_contrib/model.py's MultiHeadFutureQueryPredictor
and src/method/future_contrib/train_multi.py). Same generic .score(...)
interface as UnifiedRFCScorer/OracleFutureScorer, so it plugs into rfc's
"unified_rfc_scorer" slot in dense_eviction_hf.py without new eviction-loop
code."""
from __future__ import annotations

from pathlib import Path

import torch

from src.method.future_contrib.model import MultiHeadFutureQueryPredictor
from src.method.future_contrib.query_state import QueryRingBuffer


class MultiHeadRFCScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu", lambda_mix: float = 1.0):
        payload = torch.load(str(checkpoint), map_location="cpu")
        required = {"variant", "query_predictors", "model_arch", "recent_window", "num_prototypes"}
        if not required.issubset(payload.keys()):
            raise ValueError(f"{checkpoint} is not a train_multi.py checkpoint (expected keys {sorted(required)})")
        self.variant = payload["variant"]  # "horizon" (priority 4) | "prototype" (priority 5)
        self.device = torch.device(device)
        self.lambda_mix = float(lambda_mix)
        self.recent_window = int(payload["recent_window"])
        self.num_prototypes = int(payload["num_prototypes"])
        self.horizons = payload.get("horizons")  # variant=="horizon" only
        # variant=="horizon": which prototype's distribution to serve at eval
        # time -- defaults to the SHORTEST trained horizon (index 0, by
        # construction the closest match to real short-generation downstream
        # tasks like qasper, ~30 tokens -- the horizon-mismatch this project
        # already diagnosed as one real contributor). Override to blend or
        # pick a different index via eval_horizon_idx / eval_blend.
        self.eval_horizon_idx = 0
        self.eval_blend = False
        arch = payload["model_arch"]
        self.head_dim = int(arch["head_dim"])
        self.scaling = self.head_dim ** -0.5
        query_hidden_dim = int(payload.get("query_hidden_dim", 128))

        self.query_predictors: list[torch.nn.Module] = []
        for state_dict in payload["query_predictors"]:
            m = MultiHeadFutureQueryPredictor(self.recent_window, self.head_dim, self.num_prototypes, hidden_dim=query_hidden_dim).to(self.device)
            m.load_state_dict(state_dict)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.query_predictors.append(m)

    @torch.no_grad()
    def _prototype_forward(
        self,
        q_ring: QueryRingBuffer,
        k_cand: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Shared computation core for .score()/.score_with_confidence() --
        returns (dists [H,K,N], gate [H,K], mixed [H,N]) or None on the same
        early-out condition .score() uses. Kept as one code path so
        score_with_confidence() is guaranteed byte-identical to score() for
        the mixed/F_i output (future_fusion="fixed" correctness check)."""
        if layer_idx >= len(self.query_predictors) or k_cand.shape[1] == 0:
            return None
        recent_q_flat = q_ring.flatten().to(self.device)
        protos, gate_logits = self.query_predictors[layer_idx](recent_q_flat)  # protos: [H, K, d], gate: [H, K]
        k_cand_dev = k_cand.to(self.device).float()  # [H, N, d]
        # logits_k[h, n] = k_cand[h, n] . protos[h, k]
        logits = torch.einsum("hnd,hkd->hkn", k_cand_dev, protos.float()) * self.scaling  # [H, K, N]
        dists = torch.softmax(logits, dim=-1)  # [H, K, N], each k a valid distribution over candidates
        gate = torch.softmax(gate_logits.float(), dim=-1)  # [H, K]
        mixed = torch.einsum("hk,hkn->hn", gate, dists)  # [H, N]
        return dists, gate, mixed

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
        """Returns S_future: [num_kv_heads, num_candidates], same convention
        as RFCScorer.score_reuse / UnifiedRFCScorer.score."""
        if self.variant == "prototype":
            out = self._prototype_forward(q_ring, k_cand, layer_idx)
            if out is None:
                return k_cand.new_zeros((k_cand.shape[0], k_cand.shape[1]))
            _dists, _gate, mixed = out
            return mixed.to(k_cand.device)

        if layer_idx >= len(self.query_predictors) or k_cand.shape[1] == 0:
            return k_cand.new_zeros((k_cand.shape[0], k_cand.shape[1]))
        recent_q_flat = q_ring.flatten().to(self.device)
        protos, gate_logits = self.query_predictors[layer_idx](recent_q_flat)  # protos: [H, K, d], gate: [H, K]
        k_cand_dev = k_cand.to(self.device).float()  # [H, N, d]
        logits = torch.einsum("hnd,hkd->hkn", k_cand_dev, protos.float()) * self.scaling  # [H, K, N]
        dists = torch.softmax(logits, dim=-1)  # [H, K, N], each k a valid distribution over candidates

        # variant == "horizon"
        if self.eval_blend:
            gate = torch.softmax(gate_logits.float(), dim=-1)  # gate_head is unused/untrained for this
            # variant but harmless; blend uniformly across horizons instead,
            # short-to-long, weighted toward the shorter (more realistic)
            # horizons.
            weights = torch.linspace(1.0, 0.3, steps=self.num_prototypes, device=dists.device)
            weights = weights / weights.sum()
            mixed = torch.einsum("k,hkn->hn", weights, dists)
            return mixed.to(k_cand.device)
        idx = min(self.eval_horizon_idx, self.num_prototypes - 1)
        return dists[:, idx, :].to(k_cand.device)

    @torch.no_grad()
    def score_with_confidence(
        self,
        q_ring: QueryRingBuffer,
        k_cand: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """variant="prototype" only (MP objective). Returns (mixed [H,N],
        dists [H,K,N], gate [H,K]) -- the same F_i=mixed .score() returns,
        plus the raw per-prototype distributions/gate needed to compute
        prototype disagreement D_i and concentration Q for adaptive-lambda
        fusion (see dense_eviction_hf._adaptive_lambda_from_confidence).
        Returns None on the same early-out .score() uses (caller should
        fall back exactly as .score()'s zero-tensor case does)."""
        if self.variant != "prototype":
            raise ValueError("score_with_confidence is only defined for variant='prototype' (MP objective)")
        out = self._prototype_forward(q_ring, k_cand, layer_idx)
        if out is None:
            return None
        dists, gate, mixed = out
        return mixed.to(k_cand.device), dists.to(k_cand.device), gate.to(k_cand.device)
