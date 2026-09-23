"""Future-predictor input feature construction. Spec ref: section 6.

x_{i,t} = [z_i ; c_t ; z_i (elementwise*) c_t], z_i/c_t both proj_dim-d.
Deliberately excludes raw K/V, attention statistics, and absolute position --
see spec section 1's explicit exclusion list.
"""
from __future__ import annotations

import torch


def build_candidate_feature(z_i: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
    """z_i: [..., num_candidates, proj_dim], c_t: [..., proj_dim] (broadcasts
    over the candidate dim). Returns [..., num_candidates, 3*proj_dim]."""
    c_t_b = c_t.unsqueeze(-2).expand_as(z_i)
    interaction = z_i * c_t_b
    return torch.cat([z_i, c_t_b, interaction], dim=-1)
