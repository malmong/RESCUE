"""Reuse-rate / conditional-impact label construction. Spec ref: sections 14-17.

Everything here consumes a single C[i, tau] contribution matrix (one
eviction step, one layer) and a scalar threshold epsilon_l, and returns the
supervised targets. The R*M == mean(r*C) identity (section 17/37) is checked
by build_future_labels itself in debug mode, and by test_labels.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FutureLabels:
    reuse_rate: torch.Tensor          # R_i in [0,1], shape [num_candidates]
    conditional_impact: torch.Tensor  # M_i >= 0, shape [num_candidates] (0 where mask is False)
    impact_mask: torch.Tensor         # bool, True where reuse_count > 0 (section 16)
    reuse_count: torch.Tensor         # N_i^reuse, shape [num_candidates]
    future_utility: torch.Tensor      # F_i = R_i * M_i == mean_tau(r_i,tau * C_i,tau)
    horizon_len: int                  # H_t actually used (post end-of-sequence clipping)


def build_future_labels(c: torch.Tensor, epsilon: float, assert_identity: bool = False) -> FutureLabels:
    """c: [num_candidates, horizon_len] contribution magnitudes for one eviction step.

    epsilon is layer-wise calibrated (see calibrate_reuse_threshold), a single
    scalar broadcast across all candidates/steps in this layer.
    """
    num_candidates, horizon_len = c.shape
    if horizon_len == 0:
        z = c.new_zeros(num_candidates)
        return FutureLabels(z, z, torch.zeros(num_candidates, dtype=torch.bool, device=c.device), z, z, 0)

    reuse_events = (c > epsilon).float()  # r_{i,tau}, section 14
    reuse_count = reuse_events.sum(dim=-1)  # N_i^reuse
    reuse_rate = reuse_events.mean(dim=-1)  # R_i, section 15 (H_t normalization is exactly /horizon_len here)

    impact_mask = reuse_count > 0
    weighted_sum = (reuse_events * c).sum(dim=-1)
    safe_count = reuse_count.clamp_min(1.0)
    conditional_impact = weighted_sum / safe_count  # M_i, section 16
    conditional_impact = torch.where(impact_mask, conditional_impact, torch.zeros_like(conditional_impact))

    future_utility = weighted_sum / horizon_len  # F_i = R_i*M_i identically (section 17), computed directly to avoid 0*undefined when N=0

    if assert_identity:
        product = reuse_rate * conditional_impact
        # Where impact_mask is False, conditional_impact is defined as 0, so the
        # product is 0 there too -- matches future_utility (also 0, since no
        # reuse events contributed to weighted_sum).
        if not torch.allclose(product, future_utility, atol=1e-5, rtol=1e-4):
            bad = (product - future_utility).abs().max().item()
            raise AssertionError(f"R*M != F identity violated, max abs diff={bad}")

    return FutureLabels(reuse_rate, conditional_impact, impact_mask, reuse_count, future_utility, horizon_len)


_QUANTILE_MAX_ELEMENTS = 10_000_000  # torch.quantile hard-errors past 16,777,216 elements


def calibrate_reuse_threshold(contribution_samples: torch.Tensor, percentile: float = 80.0, generator: torch.Generator | None = None) -> float:
    """Layer-wise epsilon_l = Q_{percentile/100}(C_l) over a calibration split
    (spec section 14). contribution_samples: any-shape tensor of pooled C
    values collected across many (candidate, tau) pairs on calibration-only
    sequences -- never touch test/benchmark data here.

    Pooling many sequences * steps * candidates * horizon easily exceeds
    torch.quantile's hard 16,777,216-element limit, so we random-subsample
    down to _QUANTILE_MAX_ELEMENTS first -- a percentile estimate doesn't
    need every point, so this trades a little estimation noise for actually
    running.
    """
    flat = contribution_samples.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return 0.0
    if flat.numel() > _QUANTILE_MAX_ELEMENTS:
        idx = torch.randperm(flat.numel(), generator=generator)[:_QUANTILE_MAX_ELEMENTS]
        flat = flat[idx]
    q = torch.quantile(flat, float(percentile) / 100.0)
    return float(q.item())


def log1p_transform(m: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Training target for the impact head (section 9): log(1 + alpha*M)."""
    return torch.log1p(alpha * m.clamp_min(0.0))


def inverse_log1p_transform(m_tilde: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Inference-time inverse of log1p_transform.

    The impact head is an unconstrained linear regressor (spec section 9), so
    at eval time -- on inputs the head never saw in training -- it can
    extrapolate to wild outputs (observed: log-domain output of ~13 on a real
    eval run, expm1'd into ~487,000 against a training-target scale of
    roughly [0, 1]). Since M is a real training target the head is only ever
    supervised to be >= 0 for (log1p(alpha*M) is itself >= 0 whenever M >= 0),
    any negative or exponentially-huge output here is purely a regression
    failure mode, not a legitimate value -- clamp the log-domain input before
    exponentiating (bounding the blowup) and clamp the result to be
    non-negative (M can never truly be negative).
    """
    # +-10 (expm1(10) ~= 22,000) still let a single wild extrapolation swamp
    # every other candidate's score by 5+ orders of magnitude on a real eval
    # run, against a typical trained-scale M of ~0.001-0.06 (this project's
    # calibration thresholds top out around 0.01-0.02). +-2 (expm1(2) ~= 6.4)
    # stays >100x above the legitimate range while still bounding the blowup.
    m_tilde = m_tilde.clamp(min=-2.0, max=2.0)
    return torch.expm1(m_tilde).clamp_min(0.0) / alpha
