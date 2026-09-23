"""Eval-time scorer for the reframed "rescue correction" objective
(rfc_objective="RESCUE"): instead of regressing raw future-importance F_i
directly (MP/MH/QC etc, all still ~19 points short of the oracle ceiling on
qasper), this predicts C_i = P(a Recent-only policy would WRONGLY evict this
candidate that an Oracle policy would keep) from cheap, already-available
recent-window Q/K/V statistics -- validated offline (see
the original single-file experiments) at grouped AUC 0.90-0.97 on
held-out qasper documents, using ONLY non-tautological features (excludes the
production S_recent score itself).

Same generic .score(...) interface as MultiHeadRFCScorer/UnifiedRFCScorer, so
it plugs into rfc's "unified_rfc_scorer" slot in dense_eviction_hf.py without
new eviction-loop code. Combine with S_recent via --rfc-combine-mode
share_additive (C_i is already a well-scaled [0,1] probability, matching the
share_additive convention other scorers rely on).
"""
from __future__ import annotations

import os

from pathlib import Path

import torch

from rescue.future_contrib.query_state import QueryRingBuffer

# Feature order MUST match rescue_train_export.py's ALL_FEATURES exactly:
# 9 raw + 9 within-candidate-pool z-score + 9 within-candidate-pool rank.
_RAW_ORDER = ["current_qk", "mean_qk", "max_qk", "var_qk", "slope_qk",
              "k_norm", "v_norm", "vwo_norm", "age"]


class RescueScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu", lambda_mix: float = 1.0):
        payload = torch.load(str(checkpoint), map_location="cpu")
        kind = payload.get("kind")
        if kind not in ("rescue_scorer_v1", "rescue_scorer_ensemble_v1",
                        "rescue_scorer_layergroup_v1"):
            raise ValueError(f"{checkpoint} is not a rescue_train_export.py checkpoint")
        self.device = torch.device(device)
        self.lambda_mix = float(lambda_mix)
        self.recent_window = int(payload["recent_window"])
        assert payload["raw_features"] == _RAW_ORDER, "feature order drift between export and scorer"
        # all_features: which of the 27 (9 raw + 9 z + 9 rank) columns this
        # checkpoint was actually trained on, in the exact order the scaler/
        # MLP expect -- older checkpoints omit this key and always meant the
        # full 27, so default preserves that behavior.
        self.all_features = payload.get(
            "all_features",
            _RAW_ORDER + [f"z_{n}" for n in _RAW_ORDER] + [f"rk_{n}" for n in _RAW_ORDER],
        )
        self.scaler_mean = payload["scaler_mean"].to(self.device)
        self.scaler_scale = payload["scaler_scale"].to(self.device)
        # rescue_scorer_layergroup_v1: one MLP + scaler per LAYER GROUP rather
        # than one shared by all 32 layers. Measured motivation
        # (compare_oracle_vs_learned.py): a shared MLP recovers the tokens
        # LaProx misses equally well on every task in layers 0-4 (qasper 0.448
        # vs lcc 0.450) but only half as well off its training domain in layers
        # 12-31 (qasper 0.384 vs lcc 0.215, trec 0.156) -- the right rule
        # differs by depth, and one parameter set cannot hold all of them.
        # layer_idx is known for free at inference and is not task detection.
        self.layer_groups = None
        if kind == "rescue_scorer_layergroup_v1":
            groups = payload["groups"]
            self.layer_groups = [(int(g["lo"]), int(g["hi"])) for g in groups]
            self.group_scaler_mean = [m.to(self.device) for m in payload["group_scaler_mean"]]
            self.group_scaler_scale = [s.to(self.device) for s in payload["group_scaler_scale"]]
            self.group_mlp_weights = [[w.to(self.device) for w in ws] for ws in payload["group_mlp_weights"]]
            self.group_mlp_biases = [[b.to(self.device) for b in bs] for bs in payload["group_mlp_biases"]]
        if kind == "rescue_scorer_ensemble_v1":
            # members: list of {"mlp_weights": [...], "mlp_biases": [...]}, all
            # sharing this checkpoint's scaler/all_features (must have been
            # trained on the identical feature set -- only the random seed/
            # init differs across members). Predictions are averaged in
            # PROBABILITY space (post-sigmoid), the standard way to ensemble
            # classifiers, to reduce the ~1-1.5pt run-to-run seed variance
            # measured on single-seed checkpoints of this exact architecture.
            self.is_ensemble = True
            self.ensemble_members = [
                ([w.to(self.device) for w in m["mlp_weights"]], [b.to(self.device) for b in m["mlp_biases"]])
                for m in payload["members"]
            ]
            self.mlp_weights = self.ensemble_members[0][0]  # for backward-compat callers of _mlp_forward
            self.mlp_biases = self.ensemble_members[0][1]
        else:
            self.is_ensemble = False
            self.mlp_weights = [w.to(self.device) for w in payload["mlp_weights"]]
            self.mlp_biases = [b.to(self.device) for b in payload["mlp_biases"]]
        # score_mode="sigmoid" (default, v1/v2 checkpoints): C_i = sigmoid(logit),
        # independently share-normalized downstream by share_additive -- found
        # (rescue_scale_check.py) to be nearly UNIFORM across the full
        # candidate population (entropy_ratio 0.996, top_B_mass 2.5% vs
        # recent_share's 49%), since the MLP was only trained to discriminate
        # WITHIN the boundary population, not to push confident 0/1 values
        # across the much broader real eviction-time population. score_mode=
        # "softmax": C_i = softmax(logit/temperature) over the candidate pool
        # directly -- properly peaked/concentrated by construction, closing
        # the shape mismatch with recent_share.
        self.score_mode = payload.get("score_mode", "sigmoid")
        self.softmax_temperature = float(payload.get("softmax_temperature", 1.0))
        self._o_proj_block_cache: dict[int, torch.Tensor] = {}  # layer_idx -> [num_kv_heads, head_dim, hidden]

    def _o_proj_blocks(self, layer_idx: int, o_proj_weight: torch.Tensor, num_kv_heads: int, kv_repeat: int, head_dim: int) -> torch.Tensor:
        cached = self._o_proj_block_cache.get(layer_idx)
        if cached is not None:
            return cached
        out_features = o_proj_weight.shape[0]
        num_heads_total = num_kv_heads * kv_repeat
        # weight: [out_features, in_features], in_features = concatenated per-(query-)head V-projection slices
        blocks = o_proj_weight[:, : num_heads_total * head_dim].view(out_features, num_heads_total, head_dim)
        blocks = blocks.permute(1, 2, 0)  # [num_heads_total, head_dim, out_features]
        blocks = blocks.view(num_kv_heads, kv_repeat, head_dim, out_features).mean(dim=1)  # [num_kv_heads, head_dim, out_features]
        blocks = blocks.to(self.device)
        self._o_proj_block_cache[layer_idx] = blocks
        return blocks

    @staticmethod
    def _run_member(x: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> torch.Tensor:
        h = x
        n_layers = len(weights)
        for i, (w, b) in enumerate(zip(weights, biases)):
            h = torch.nn.functional.linear(h, w, b)
            if i < n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)

    def _mlp_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self._run_member(x, self.mlp_weights, self.mlp_biases)

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid(logit), averaged across ensemble members in PROBABILITY
        space when self.is_ensemble -- kept for backward compat / diagnostics
        that want a per-candidate probability independent of the candidate
        pool."""
        if self.is_ensemble:
            probs = [torch.sigmoid(self._run_member(x, w, b)) for w, b in self.ensemble_members]
            return torch.stack(probs, dim=0).mean(dim=0)
        return torch.sigmoid(self._mlp_logit(x))

    @staticmethod
    def _zscore_and_rank(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [N] -> (z [N], rank-in-[0,1] [N]), matching
        rescue_train_export.py's per-(doc,layer) population normalization
        (here the population is this call's own full candidate pool)."""
        n = x.shape[-1]
        mu = x.mean()
        sd = x.std(unbiased=False) + 1e-8
        z = (x - mu) / sd
        if n <= 1:
            rank = torch.zeros_like(x)
        else:
            order = x.argsort(dim=-1)
            rank = torch.empty_like(x)
            rank[order] = torch.arange(n, device=x.device, dtype=x.dtype)
            rank = rank / (n - 1)
        return z, rank

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
        base_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns C_i broadcast to [num_kv_heads, num_candidates] (head-shared,
        matching how the offline classifier was trained on kv-head-averaged
        features -- see rescue_feature_collect.py's feat_mean)."""
        num_kv_heads, n_cand, head_dim = k_cand.shape
        if n_cand == 0:
            return k_cand.new_zeros((num_kv_heads, 0))
        window = min(self.recent_window, q_ring.recent_window)
        q_win = q_ring.last_m(window).to(self.device).float()  # [H, W, d]
        k_dev = k_cand.to(self.device).float()  # [H, N, d]
        v_dev = v_cand.to(self.device).float()  # [H, N, d]
        scaling = float(head_dim) ** -0.5

        logits = torch.einsum("hwd,hnd->hwn", q_win, k_dev) * scaling  # [H, W, N]
        current_qk = logits[:, -1, :]
        mean_qk = logits.mean(dim=1)
        max_qk = logits.max(dim=1).values
        var_qk = logits.var(dim=1, unbiased=False)
        w = logits.shape[1]
        xw = torch.arange(w, device=self.device, dtype=torch.float32) - (w - 1) / 2.0
        xw_ss = (xw * xw).sum().clamp_min(1e-6)
        slope_qk = (logits * xw.view(1, w, 1)).sum(dim=1) / xw_ss  # [H, N]

        k_norm = k_dev.norm(p=2, dim=-1)  # [H, N]
        v_norm = v_dev.norm(p=2, dim=-1)  # [H, N]
        blocks = self._o_proj_blocks(layer_idx, o_proj_weight.to(self.device).float(), num_kv_heads, kv_repeat, head_dim)
        # Only the ROW NORMS of V @ W_O are used, so the [H, N, out_features]
        # product never has to exist all at once. out_features is the model dim
        # (4096), which made this the single largest allocation in the scorer:
        # at narrativeqa lengths it asked for ~7.8 GiB on top of the dense cache
        # the fidelity selector is still holding, and OOM'd. Chunking over N is
        # exact -- each row's norm depends only on that row.
        vwo_chunk = max(1, (256 * 1024 * 1024) // max(1, 4 * num_kv_heads * int(blocks.shape[-1])))
        if v_dev.shape[1] <= vwo_chunk:
            vwo_norm = torch.einsum("hnd,hdm->hnm", v_dev, blocks).norm(p=2, dim=-1)  # [H, N]
        else:
            vwo_norm = torch.empty(v_dev.shape[0], v_dev.shape[1],
                                   device=v_dev.device, dtype=v_dev.dtype)
            for st_ in range(0, v_dev.shape[1], vwo_chunk):
                en_ = min(st_ + vwo_chunk, v_dev.shape[1])
                vwo_norm[:, st_:en_] = torch.einsum(
                    "hnd,hdm->hnm", v_dev[:, st_:en_, :], blocks).norm(p=2, dim=-1)

        pos = cand_positions if cand_positions.dim() == 1 else cand_positions[0]
        pos = pos.to(self.device).float()
        if pos.shape[-1] != n_cand:
            pos = pos[-n_cand:]
        age = (float(cur_t) - pos).unsqueeze(0).expand(num_kv_heads, -1)  # [H, N]

        raw = {
            "current_qk": current_qk, "mean_qk": mean_qk, "max_qk": max_qk, "var_qk": var_qk,
            "slope_qk": slope_qk, "k_norm": k_norm, "v_norm": v_norm, "vwo_norm": vwo_norm, "age": age,
        }
        # head-shared reduction (mean over kv_heads), matching training's feat_mean convention
        shared = {k: v.mean(dim=0) for k, v in raw.items()}  # each [N]

        zscore = {}
        rank = {}
        for name in _RAW_ORDER:
            z, rk = self._zscore_and_rank(shared[name])
            zscore[name] = z
            rank[name] = rk
        by_column_name = dict(shared)
        by_column_name.update({f"z_{name}": zscore[name] for name in _RAW_ORDER})
        by_column_name.update({f"rk_{name}": rank[name] for name in _RAW_ORDER})
        # base_score: TRIED AND REJECTED 2026-09-16, plumbing kept inert.
        # The motivation was sound -- C_i's target is defined against the base
        # ("tokens the oracle keeps and the base drops") yet none of the 18
        # features told it what the base was about to drop, so one shared rule
        # had no way to behave differently under a different base. It was not a
        # tautology either: corr(base_score, rescue) ~ 0 and a base-rank
        # threshold reaches only 5% precision on the rescue set.
        # But it FIT better and SCORED worse: val loss 6.53 -> 6.35 while LaProx
        # collapsed (qasper 34.82 -> 25.76, under its own 27.85 baseline;
        # multifieldqa_en 53.47 -> 45.61). Since the rule and the training
        # corpus have to be shared across bases, a feature that breaks LaProx is
        # unusable whatever it does for the others. No checkpoint declares these
        # columns, so this branch never fires; it is left so the next attempt
        # starts from the measurement rather than the idea.
        if any(f in ("base_score", "rk_base_score") for f in self.all_features):
            if base_score is None:
                raise RuntimeError(
                    "this checkpoint was trained with base_score features but the "
                    "caller passed none; the correction would be ranking on a "
                    "zero-filled column")
            bs = base_score.to(self.device).float()
            if bs.dim() == 2:
                bs = bs.mean(dim=0)
            if bs.shape[-1] > n_cand:
                bs = bs[-n_cand:]
            elif bs.shape[-1] < n_cand:
                # R-KV ranks only [0, n - window_size); the trailing window is
                # protected and never scored. Pad it with the maximum so those
                # positions read as "the base wants to keep this", which is what
                # protection means -- truncating instead would misalign every
                # other feature column.
                pad = bs.new_full((n_cand - bs.shape[-1],), float(bs.max()))
                bs = torch.cat([bs, pad], dim=-1)
            bz, brk = self._zscore_and_rank(bs)
            by_column_name["base_score"] = bs
            by_column_name["rk_base_score"] = brk
        feat_cols = [by_column_name[name] for name in self.all_features]
        X = torch.stack(feat_cols, dim=-1)  # [N, len(self.all_features)]
        if self.layer_groups is not None:
            gi = next((i for i, (lo, hi) in enumerate(self.layer_groups)
                       if lo <= layer_idx <= hi), len(self.layer_groups) - 1)
            X = (X - self.group_scaler_mean[gi]) / self.group_scaler_scale[gi]
            logit = self._run_member(X, self.group_mlp_weights[gi], self.group_mlp_biases[gi])
            c = (torch.softmax(logit / self.softmax_temperature, dim=-1)
                 if self.score_mode == "softmax" else torch.sigmoid(logit))
            return c.unsqueeze(0).expand(num_kv_heads, -1).to(k_cand.device)
        if os.environ.get("RESCUE_FEAT_DEBUG"):
            # Compare the feature distribution produced at inference against
            # the scaler fitted during training. If the two drift apart the
            # normalisation stops meaning anything, and how far they drift is
            # model-dependent.
            _m = X.float().mean(0); _s = X.float().std(0)
            _z = ((_m - self.scaler_mean.float()) / self.scaler_scale.float())
            print("[FEATDBG] L%02d " % layer_idx
                  + " ".join(f"{n}:inf={_m[j]:.2f}/{_s[j]:.2f} ckpt={self.scaler_mean[j]:.2f}/{self.scaler_scale[j]:.2f} z={_z[j]:+.2f}"
                             for j, n in enumerate(self.all_features[:9])), flush=True)
        X = (X - self.scaler_mean) / self.scaler_scale
        if os.environ.get("RESCUE_PROB_DEBUG"):
            _lg = self._run_member(X, self.mlp_weights, self.mlp_biases)
            _pp = torch.softmax(_lg / self.softmax_temperature, dim=-1)
            _N = _pp.numel()
            _ent = float(-(_pp * _pp.clamp_min(1e-12).log()).sum())
            _eff = float(torch.exp(torch.tensor(_ent)))
            _k = min(124, _N)
            print(f"[PROBDBG] L{layer_idx:02d} N={_N} eff={_eff:.1f} ({100*_eff/max(_N,1):.1f}%) "
                  f"maxN={float(_pp.max())*_N:.1f} top{_k}mass={float(_pp.topk(_k).values.sum()):.3f}",
                  flush=True)
        if self.score_mode == "softmax":
            # Ensembling in softmax mode: average the post-softmax
            # distributions across members (same "combine in probability
            # space" convention as the sigmoid ensemble in _mlp_forward), NOT
            # the pre-softmax logits -- averaging logits first would let one
            # member's out-of-distribution outlier logit (candidates outside
            # the boundary population this MLP was trained on are never seen
            # in training, unlike softmax's amplification of a single raw
            # logit) dominate the shared softmax before the other members'
            # disagreement can damp it down.
            if self.is_ensemble:
                dists = [torch.softmax(self._run_member(X, w, b) / self.softmax_temperature, dim=-1)
                          for w, b in self.ensemble_members]
                c = torch.stack(dists, dim=0).mean(dim=0)  # [N]
            else:
                logit = self._mlp_logit(X)  # [N]
                c = torch.softmax(logit / self.softmax_temperature, dim=-1)  # [N], sums to 1, properly peaked
        else:
            c = self._mlp_forward(X)  # [N], in [0,1], sigmoid -- near-uniform once share-normalized
        return c.unsqueeze(0).expand(num_kv_heads, -1).to(k_cand.device)
