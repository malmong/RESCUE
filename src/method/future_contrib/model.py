"""Per-layer shared-trunk two-head future predictor. Spec ref: section 7-9, 20.

One predictor instance per transformer layer (V1 choice, section 20). Trunk
outputs a latent vector, not a scalar; reuse/impact heads branch off it.
"""
from __future__ import annotations

import torch
from torch import nn


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class SharedFuturePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 96,
        shared_hidden_dim: int = 64,
        task_hidden_dim: int = 16,
        activation: str = "silu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        trunk: list[nn.Module] = [nn.Linear(input_dim, shared_hidden_dim), _act(activation)]
        if use_layernorm:
            trunk.append(nn.LayerNorm(shared_hidden_dim))
        self.shared_trunk = nn.Sequential(*trunk)

        self.reuse_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, task_hidden_dim), _act(activation), nn.Linear(task_hidden_dim, 1)
        )
        self.impact_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, task_hidden_dim), _act(activation), nn.Linear(task_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [..., input_dim]. Returns (reuse_rate_hat in [0,1], impact_log_hat unconstrained)."""
        u = self.shared_trunk(x)
        reuse_logit = self.reuse_head(u).squeeze(-1)
        reuse_hat = torch.sigmoid(reuse_logit)
        impact_log_hat = self.impact_head(u).squeeze(-1)
        return reuse_hat, impact_log_hat


class SeparateFuturePredictor(nn.Module):
    """Ablation (section 34): fully independent reuse/impact towers, no shared trunk."""

    def __init__(
        self,
        input_dim: int = 96,
        shared_hidden_dim: int = 64,
        task_hidden_dim: int = 16,
        activation: str = "silu",
    ):
        super().__init__()
        self.reuse_tower = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim), _act(activation),
            nn.Linear(shared_hidden_dim, task_hidden_dim), _act(activation),
            nn.Linear(task_hidden_dim, 1),
        )
        self.impact_tower = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim), _act(activation),
            nn.Linear(shared_hidden_dim, task_hidden_dim), _act(activation),
            nn.Linear(task_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reuse_hat = torch.sigmoid(self.reuse_tower(x).squeeze(-1))
        impact_log_hat = self.impact_tower(x).squeeze(-1)
        return reuse_hat, impact_log_hat


class DirectFutureScorer(nn.Module):
    """Ablation (section 33): single opaque future-importance scalar, no R/M decomposition."""

    def __init__(
        self,
        input_dim: int = 96,
        shared_hidden_dim: int = 64,
        task_hidden_dim: int = 16,
        activation: str = "silu",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim), _act(activation),
            nn.Linear(shared_hidden_dim, task_hidden_dim), _act(activation),
            nn.Linear(task_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns F_hat directly, in log1p-domain (same target transform as impact)."""
        return self.net(x).squeeze(-1)


class ImpactPredictor(nn.Module):
    """Conditional-impact-only predictor -- split out from SharedFuturePredictor
    once the reuse branch moved to an entirely different feature basis
    (Q/K interaction over recent queries, see FutureQueryPredictor) that no
    longer shares a trunk with impact's hidden-state-projection features.
    A quick diagnostic found ||V_i @ W_O|| alone only weakly explains
    conditional impact (corr 0.02-0.17 across layers), so impact's original
    feature basis (candidate/current-state projection interaction) is kept
    unchanged here rather than switched to a V/W_O-only feature."""

    def __init__(self, input_dim: int = 96, shared_hidden_dim: int = 64, task_hidden_dim: int = 16, activation: str = "silu"):
        super().__init__()
        self.shared_trunk = nn.Sequential(nn.Linear(input_dim, shared_hidden_dim), _act(activation))
        self.impact_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, task_hidden_dim), _act(activation), nn.Linear(task_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.shared_trunk(x)
        return self.impact_head(u).squeeze(-1)


class FutureQueryPredictor(nn.Module):
    """Predicts a per-kv-head future-query prototype from recent query
    history. Trained via KL divergence against the REAL future softmax-
    attention distribution -- never a mean-query regression, since
    softmax(E[q]K^T) != E[softmax(qK^T)] (verified experimentally: raw
    future-Q.K logit correlates only ~0.3-0.45 with true future reuse while
    the real future softmax-attention distribution correlates ~0.87-0.98,
    and a query predictor trained this way reaches ~0.74-0.86 Pearson on
    held-out documents -- see the diagnostic history in this package's
    training script for the full gate sequence that led here).

    recent_window*head_dim input (flattened, grouped to kv-heads exactly
    like the candidate keys it will be dotted against) -> head_dim output,
    one shared-weight predictor per layer applied independently per
    kv-head (Model A from the gate experiments: a single query prototype,
    not multiple)."""

    def __init__(self, recent_window: int, head_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(recent_window * head_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, head_dim),
        )

    def forward(self, recent_q_flat: torch.Tensor) -> torch.Tensor:
        """recent_q_flat: [..., recent_window*head_dim] -> [..., head_dim]."""
        return self.net(recent_q_flat)


class MultiHeadFutureQueryPredictor(nn.Module):
    """Priority 4 (multi-horizon) / priority 5 (multi-prototype), sharing one
    implementation: same shared-trunk architecture as FutureQueryPredictor,
    but the output head is widened to K separate head_dim-sized query
    prototypes sharing one trunk, instead of collapsing to a single query
    vector. What differs between the two priorities is purely the TRAINING
    TARGET each prototype is supervised against (see train_multi.py):
    multi-horizon assigns prototype k its own horizon-chunk's real future
    distribution (a fixed, non-learned assignment); multi-prototype
    supervises a GATED MIXTURE of all K prototypes' distributions against
    the single standard-horizon target (a learned soft assignment, gate
    also produced by this same trunk). Both eval-time scorers live in
    src.method.rfc_multi.MultiHeadRFCScorer."""

    def __init__(self, recent_window: int, head_dim: int, num_prototypes: int, hidden_dim: int = 128):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.head_dim = head_dim
        self.trunk = nn.Sequential(
            nn.Linear(recent_window * head_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.proto_head = nn.Linear(hidden_dim, num_prototypes * head_dim)
        self.gate_head = nn.Linear(hidden_dim, num_prototypes)

    def forward(self, recent_q_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """recent_q_flat: [..., recent_window*head_dim]. Returns
        (prototypes [..., K, head_dim], gate_logits [..., K])."""
        u = self.trunk(recent_q_flat)
        protos = self.proto_head(u).view(*u.shape[:-1], self.num_prototypes, self.head_dim)
        gate_logits = self.gate_head(u)
        return protos, gate_logits
