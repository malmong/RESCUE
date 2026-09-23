"""Oracle-future eviction: not a learned predictor at all -- at each eviction
decision, S_future is the REAL future attention distribution, computed
exactly as training labels are (rescue.future_contrib.train.
future_attention_distribution), except the "future" comes from a reference
continuation generated once up front (policy="full", no eviction) rather
than from real generated-so-far tokens. This measures a ceiling: "if we knew
the true future perfectly, how good could future-attention-based eviction
be" -- separating "our predictor is imperfect" from "future attention isn't
even the right eviction objective" (see the diagnostic priority list this
was built for).

Same caveat as every label-construction routine already in this project:
the reference continuation is generated WITHOUT eviction, so if the budgeted
run's own generation diverges from it (a real possibility under an
aggressive budget), positions past the divergence point are scored against
"what would have been attended to if generation had continued along the
reference trajectory" rather than the true realized future. This is the
same non-self-consistent-oracle approximation used throughout this
project's training-label pipeline; a fully self-consistent oracle would
require re-generating the reference after every eviction decision, which is
intractable.
"""
from __future__ import annotations

import torch

from rescue.future_contrib.extract import capture_sequence_chunked


def _oracle_head_distribution(
    q_full: torch.Tensor, k_full: torch.Tensor, cand_h: torch.Tensor,
    horizon_start: int, horizon_end: int, head_group: int, kv_repeat: int, scaling: float,
) -> torch.Tensor:
    """Real causal softmax attention that queries in [horizon_start,
    horizon_end) give to cand_h, for ONE kv-head group -- same math as
    rescue.future_contrib.train.future_attention_distribution's inner loop,
    extracted per-head so candidates can differ by head (eviction can
    diverge per kv-head once budget is reached, same as every other learned
    scorer in this project). Returns [len(cand_h)], summing to 1."""
    device = q_full.device
    cand_h = cand_h.to(device)
    abs_q_pos = torch.arange(horizon_start, horizon_end, device=device).unsqueeze(1)
    key_pos_full = torch.arange(horizon_end, device=device).unsqueeze(0)
    causal_mask = key_pos_full > abs_q_pos
    k_g = k_full[head_group, :horizon_end, :].float()
    acc = torch.zeros(horizon_end - horizon_start, cand_h.numel(), device=device)
    for h_local in range(kv_repeat):
        h = head_group * kv_repeat + h_local
        q_h = q_full[h, horizon_start:horizon_end, :].float()
        scores = torch.matmul(q_h, k_g.transpose(0, 1)) * scaling
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        acc += attn.index_select(1, cand_h)
    dist = (acc / kv_repeat).mean(dim=0)
    return dist / dist.sum().clamp_min(1e-12)


class OracleFutureScorer:
    """Drop-in replacement for UnifiedRFCScorer/RFCScorer at eviction time --
    same .score(q_ring, k_cand, v_cand, cand_positions, cur_t, o_proj_weight,
    kv_repeat, layer_idx) interface, so it plugs into dense_eviction_hf.py's
    existing "using_unified" rfc branch with zero changes there. q_ring is
    accepted but ignored (oracle needs no query history -- it looks up the
    true future directly)."""

    def __init__(self, horizon: int = 256, lambda_mix: float = 1.0):
        self.horizon = int(horizon)
        self.lambda_mix = float(lambda_mix)
        self.recent_window = 1  # dummy: _populate_q_cache reads this but oracle never uses q_ring's contents
        self._cap = None
        self._full_len = 0
        self._scaling = None

    def set_reference(self, full_ids: torch.Tensor, model_type: str, model, scaling: float) -> None:
        """Call once per document, BEFORE generate_one runs, with the
        reference (prompt + no-eviction-generated continuation) token ids.
        Captures real (rotary-applied) Q/K for every layer over the whole
        reference sequence via one teacher-forced pass."""
        with torch.no_grad():
            # Chunked (not plain capture_sequence): a single one-shot forward
            # over the full reference sequence would need eager attention's
            # full O(seq^2) matrix, which OOMs on qasper's longer outlier
            # documents now that max_seq_len is uncapped (see
            # capture_sequence_chunked's docstring).
            # keep_hidden=False: score() reads q/k only, and the hidden
            # accumulation is what exhausts a 95GB card on long documents.
            self._cap = capture_sequence_chunked(model, full_ids, model_type, keep_hidden=False)
        self._full_len = int(full_ids.shape[1])
        self._scaling = float(scaling)

    @torch.no_grad()
    def score(
        self,
        q_ring,
        k_cand: torch.Tensor,
        v_cand: torch.Tensor,
        cand_positions: torch.Tensor,
        cur_t: int,
        o_proj_weight: torch.Tensor,
        kv_repeat: int,
        layer_idx: int,
        activity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        base_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # base_score is accepted and ignored. RESCUE's scorer conditions on the
        # base policy's own score; the oracle does not -- it reads the true
        # future attention directly, which is what makes it a ceiling rather
        # than a method. The argument exists only so this class stays a
        # drop-in for the same call site.
        num_kv_heads = k_cand.shape[0]
        num_candidates = k_cand.shape[1]
        if self._cap is None or layer_idx not in self._cap.qkv:
            return k_cand.new_zeros((num_kv_heads, num_candidates))
        horizon_start = cur_t + 1
        horizon_end = min(horizon_start + self.horizon, self._full_len)
        if horizon_end <= horizon_start or num_candidates == 0:
            return k_cand.new_zeros((num_kv_heads, num_candidates))
        q_full, k_full, _ = self._cap.qkv[layer_idx]
        pos = cand_positions
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(num_kv_heads, -1)
        out = k_cand.new_zeros((num_kv_heads, num_candidates), dtype=torch.float32)
        for h in range(num_kv_heads):
            out[h] = _oracle_head_distribution(
                q_full, k_full, pos[h], horizon_start, horizon_end, h, kv_repeat, self._scaling,
            ).to(k_cand.device)
        return out.to(k_cand.device)
