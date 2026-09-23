"""Supervised regression losses. Spec ref: section 18-19.

No pairwise ranking loss, no RL -- calibrated magnitude is the point (section
1's rationale, reiterated in section 19).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ReuseLoss(nn.Module):
    """Huber regression on the continuous reuse rate (never BCE -- section 18.1:
    the target is a rate in [0,1], not a binary label)."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, reuse_hat: torch.Tensor, reuse_target: torch.Tensor) -> torch.Tensor:
        if reuse_hat.numel() == 0:
            return reuse_hat.sum() * 0.0
        return F.huber_loss(reuse_hat, reuse_target, delta=self.delta)


class ImpactLoss(nn.Module):
    """Masked Huber regression in log1p-domain (section 18.2). Candidates with
    reuse_count == 0 (impact_mask False) are excluded, not zero-labeled."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, impact_log_hat: torch.Tensor, impact_log_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return impact_log_hat.sum() * 0.0
        pred = impact_log_hat[mask]
        target = impact_log_target[mask]
        return F.huber_loss(pred, target, delta=self.delta)


class FutureConsistencyLoss(nn.Module):
    """Optional auxiliary loss (section 19): the R*M product should also match
    the ground-truth future utility F, evaluated in a second log1p-domain
    (alpha_F, independent of the impact head's own alpha)."""

    def __init__(self, alpha_f: float = 1.0, delta: float = 1.0):
        super().__init__()
        self.alpha_f = alpha_f
        self.delta = delta

    def forward(self, reuse_hat: torch.Tensor, impact_hat: torch.Tensor, future_utility_target: torch.Tensor) -> torch.Tensor:
        f_hat = reuse_hat * impact_hat
        pred_log = torch.log1p(self.alpha_f * f_hat.clamp_min(0.0))
        target_log = torch.log1p(self.alpha_f * future_utility_target.clamp_min(0.0))
        return F.huber_loss(pred_log, target_log, delta=self.delta)


class FutureContribLossBundle(nn.Module):
    def __init__(self, lambda_r: float = 1.0, lambda_m: float = 1.0, lambda_f: float = 0.25, alpha: float = 1.0, alpha_f: float = 1.0):
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_m = lambda_m
        self.lambda_f = lambda_f
        self.alpha = alpha
        self.reuse_loss = ReuseLoss()
        self.impact_loss = ImpactLoss()
        self.consistency_loss = FutureConsistencyLoss(alpha_f=alpha_f)

    def forward(
        self,
        reuse_hat: torch.Tensor,
        impact_log_hat: torch.Tensor,
        reuse_target: torch.Tensor,
        impact_log_target: torch.Tensor,
        impact_mask: torch.Tensor,
        future_utility_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_r = self.reuse_loss(reuse_hat, reuse_target)
        l_m = self.impact_loss(impact_log_hat, impact_log_target, impact_mask)
        total = self.lambda_r * l_r + self.lambda_m * l_m
        out = {"loss_reuse": l_r, "loss_impact": l_m}
        if self.lambda_f > 0 and future_utility_target is not None:
            impact_hat = torch.expm1(impact_log_hat) / self.alpha
            l_f = self.consistency_loss(reuse_hat, impact_hat, future_utility_target)
            total = total + self.lambda_f * l_f
            out["loss_consistency"] = l_f
        out["loss_total"] = total
        return out
