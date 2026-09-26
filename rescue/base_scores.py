#!/usr/bin/env python
"""Each base ranker's own per-candidate score, in ONE place.

These two functions were nested closures inside the feature-caching script.
The bodies below are unchanged; what they used to close over (the model, the
GQA repeat factor, the attention scaling, the R-KV window) is passed in.

They were lifted so the Section 5.3 mechanism analysis scores the bases with
exactly the implementation the training targets were built from. A second copy
is the easiest way for the analysis to stop describing the runs it explains.
"""
from __future__ import annotations

import torch

from rescue import features as ORIGMOD
from rescue.features import OBS_WINDOW

# BaselineConfig's R-KV defaults, which the official code ships with. Kept here
# rather than imported from the caching script so this module has no dependency
# back on it.
CFG_RKV_MIX = 0.07
CFG_RKV_RETAIN = 0.1


def get_o_proj_block(layer_idx: int, kv_h: int, *, model, num_heads: int,
                     kv_repeat: int, head_dim: int, o_cache: dict) -> torch.Tensor:
    if layer_idx not in o_cache:
        w = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().float()
        o_cache[layer_idx] = [w[:, h * head_dim:(h + 1) * head_dim].t().contiguous()
                              for h in range(num_heads)]
    g = o_cache[layer_idx][kv_h * kv_repeat:(kv_h + 1) * kv_repeat]
    return torch.stack(g, dim=0).mean(dim=0)


def base_head_score(base, q_full, k_full, v_full, cand, cur_t, kv_h, o_block, *,
                    kv_repeat: int, scaling: float, cfg_rkv_window: int):
    """The base ranker's own per-candidate score, matching exactly what the
    eval path computes for --rfc-recent-style <base> (dense_eviction_hf.py
    lines 1155-1181):
      laprox  ||softmax(obs-window qk)||_2 * ||V W_O||      (value-weighted)
      snapkv  SUM of obs-window attention, no value weighting, avg-pool k=7
      h2o     cumulative attention from EVERY query position, not just the
              observation window
    """
    if base == "laprox":
        return ORIGMOD.recent_laprox_score(q_full, k_full, v_full, cand, cur_t,
                                           OBS_WINDOW, kv_h, kv_repeat, scaling, o_block)
    if base == "snapkv":
        logits = ORIGMOD.recent_qk_logits(q_full, k_full, cand, cur_t, OBS_WINDOW,
                                          kv_h, kv_repeat, scaling)
        vote = torch.softmax(logits, dim=-1).sum(dim=0)          # [n_cand]
        pooled = torch.nn.functional.avg_pool1d(
            vote.view(1, 1, -1), kernel_size=7, padding=3, stride=1).view(-1)
        return pooled
    if base == "lava":
        # lava_head_token_scores, rebuilt from q/k/v: per kv-head group the
        # query heads are combined with amax (not sum/mean), the window's
        # attention is summed, MAXpooled at kernel 7 (always maxpool, whatever
        # --snapkv-pooling says), and scaled by that head's largest ||V||_1.
        dev = q_full.device
        cand_d = cand.to(dev)
        k_g = k_full[kv_h, :, :].float()
        start = max(0, cur_t - OBS_WINDOW + 1)
        # Causal mask: the observation-window queries sit at positions
        # start..cur_t and must not attend to keys after themselves. HF's
        # eager attention masks this; without it the softmax normalises over
        # a larger set and the scores drift (measured: top-92 agreement with
        # the online lava_head_token_scores rose from 86/92 to near-exact
        # once this was added).
        kpos = torch.arange(cur_t + 1, device=dev).unsqueeze(0)
        qpos = torch.arange(start, cur_t + 1, device=dev).unsqueeze(1)
        causal = kpos > qpos
        per_h = []
        for h_local in range(kv_repeat):
            h = kv_h * kv_repeat + h_local
            q_h = q_full[h, start:cur_t + 1, :].float()
            sc = torch.matmul(q_h, k_g[: cur_t + 1].transpose(0, 1)) * scaling
            sc = sc.masked_fill(causal, float("-inf"))
            per_h.append(torch.softmax(sc, dim=-1))
        grouped = torch.stack(per_h, 0).amax(dim=0)          # [w, cur_t+1]
        vote = grouped.sum(dim=0)                             # [cur_t+1]
        pooled = torch.nn.functional.max_pool1d(
            vote.view(1, 1, -1), kernel_size=7, padding=3, stride=1).view(-1)
        v_l1 = v_full[kv_h, : cur_t + 1, :].float().norm(p=1, dim=-1)
        v_max = v_l1.max().clamp_min(1e-12)
        w = max(1, cur_t + 1 - start)
        return ((v_max / float(w)) * pooled).index_select(0, cand_d)
    if base == "rkv":
        # R-KV's own mix: mix_lambda * attention-cache - (1-mix_lambda) *
        # redundancy, using the library's real functions so the offline
        # rescue target matches what --method rkv ranks on.
        from rescue.policies.rkv import rkv_final_score
        dev = q_full.device
        start = max(0, cur_t - int(cfg_rkv_window) + 1)
        q_win = q_full[:, start:cur_t + 1, :]                  # [heads, w, d]
        k_st = k_full[:, : cur_t + 1, :].unsqueeze(0)          # [1, kv_heads, n, d]
        sc = rkv_final_score(q_win, k_st, window_size=int(cfg_rkv_window),
                             mix_lambda=CFG_RKV_MIX, kernel_size=7,
                             retain_ratio=CFG_RKV_RETAIN, retain_direction="last")
        # rkv_final_score drops the trailing window_size positions; every
        # candidate sits before the protected observation window (32 > 8),
        # so indexing by absolute position is still valid.
        return sc[kv_h].index_select(0, cand.to(dev))
    if base == "h2o":
        # every query position 0..cur_t attends over the whole prefix; H2O
        # accumulates that mass per key. Chunked over queries so a 40k
        # prompt never materialises one [cur_t, cur_t] matrix.
        dev = q_full.device
        cand_d = cand.to(dev)
        k_g = k_full[kv_h, : cur_t + 1, :].float()
        acc = torch.zeros(cand_d.numel(), device=dev)
        step = 2048
        for qs in range(0, cur_t + 1, step):
            qe = min(qs + step, cur_t + 1)
            kpos = torch.arange(cur_t + 1, device=dev).unsqueeze(0)
            qpos = torch.arange(qs, qe, device=dev).unsqueeze(1)
            causal = kpos > qpos
            for h_local in range(kv_repeat):
                h = kv_h * kv_repeat + h_local
                q_h = q_full[h, qs:qe, :].float()
                sc = torch.matmul(q_h, k_g.transpose(0, 1)) * scaling
                sc = sc.masked_fill(causal, float("-inf"))
                acc += torch.softmax(sc, dim=-1).index_select(1, cand_d).sum(dim=0)
            del sc
        return acc / kv_repeat
    raise ValueError(base)
